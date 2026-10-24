"""Regressões das otimizações de contexto, ferramentas e limite de quota."""
from __future__ import annotations

import copy
import json
from types import SimpleNamespace as NS

import pytest
from banco_agil import bootstrap
from banco_agil.agents import AGENTS
from banco_agil.exchange import ExchangeService
from banco_agil.flow import FlowController
from banco_agil.llm import LlmRetryState, _build_payload, _estimate_request_tokens, _tool_schemas, call_llm
from banco_agil.orchestrator import Orchestrator
from banco_agil.session import Session
from banco_agil.statuses import STATUS_APROVADO, STATUS_REPROVADO

from fakes import FakeLLM, assert_history_valid, reply, ScriptedClient, RateLimited, ToolUseFailed



SCORE_ANSWERS = {
    "renda_mensal": 6000,
    "tipo_emprego": "formal",
    "despesas_fixas": 2000,
    "num_dependentes": 1,
    "tem_dividas": False,
}


def _interview_session() -> Session:
    session = Session(active_agent="entrevista", authenticated=True, cpf="52998224725", nome="Bruno Lima")
    FlowController.start_interview(session)
    return session


def test_context_keeps_two_latest_customer_turns_and_complete_tool_pairs():
    session = Session()
    session.history = [
        {"role": "assistant", "content": "Saudação"},
        {"role": "user", "content": "turno antigo"},
        {"role": "assistant", "content": "resposta antiga"},
        {"role": "user", "content": "penúltimo turno"},
        {"role": "assistant", "content": None, "tool_calls": [{
            "id": "call-1", "type": "function",
            "function": {"name": "consultar_limite", "arguments": "{}"},
        }]},
        {"role": "tool", "tool_call_id": "call-1", "name": "consultar_limite", "content": "{}"},
        {"role": "assistant", "content": "Seu limite é ..."},
        {"role": "user", "content": "turno atual"},
    ]
    original = copy.deepcopy(session.history)

    payload = _build_payload(AGENTS["credito"], session)

    assert payload[2:] == original[3:]
    assert payload[3]["tool_calls"][0]["id"] == payload[4]["tool_call_id"] == "call-1"
    assert session.history == original


def test_interview_sends_only_tools_possible_in_current_state():
    session = _interview_session()
    incomplete_names = {tool["function"]["name"] for tool in _tool_schemas(AGENTS["entrevista"], session)}
    assert "registrar_respostas_entrevista" in incomplete_names
    assert "registrar_entrevista_e_calcular_score" not in incomplete_names

    session.flow.interview_answers = dict(SCORE_ANSWERS)
    complete_names = {tool["function"]["name"] for tool in _tool_schemas(AGENTS["entrevista"], session)}
    assert "registrar_entrevista_e_calcular_score" in complete_names
    assert "registrar_respostas_entrevista" not in complete_names


def test_pending_interview_summary_omits_all_tool_schemas_without_delaying_normal_reply(monkeypatch):
    session = _interview_session()
    session.flow.pending_interview_answers = {"renda_mensal": 6000}
    session.history.append({"role": "user", "content": "Ganho seis mil por mês."})
    client = ScriptedClient([reply("Resumo: renda mensal de R$ 6.000. Confirma?")])
    monkeypatch.setattr("banco_agil.llm.time.sleep", lambda _seconds: (_ for _ in ()).throw(AssertionError("espera inesperada")))

    message = call_llm(client, "fake", AGENTS["entrevista"], session)

    assert message.content.startswith("Resumo")
    names = {t["function"]["name"] for t in client.requests[0]["tools"]}
    assert "registrar_entrevista_e_calcular_score" not in names
    assert "registrar_respostas_entrevista" not in names
    assert {"encerrar_atendimento", "transferir_para_agente"}.issubset(names)
    assert session.flow.pending_interview_answers == {"renda_mensal": 6000}


def test_429_without_header_recovers_once_and_preserves_interview_state(repo, monkeypatch):
    monkeypatch.setattr("banco_agil.llm.time.sleep", lambda _: None)
    client = ScriptedClient([RateLimited(), RateLimited()])
    orch = Orchestrator(repo, ExchangeService(), client, "fake", max_llm_retries=1)
    session = _interview_session()
    session.flow.interview_answers = {"renda_mensal": 6000}
    session.flow.pending_interview_answers = {"tipo_emprego": "formal"}
    clients_before = repo.clients_path.read_bytes()
    requests_before = repo.requests_path.read_bytes()

    response = orch.handle(session, "pode continuar a entrevista?")

    assert len(client.requests) == 2
    assert "alta demanda" in response.lower()
    assert "429" not in response
    assert session.last_failure_category == "http_429"
    assert session.flow.interview_answers == {"renda_mensal": 6000}
    assert session.flow.pending_interview_answers == {"tipo_emprego": "formal"}
    assert repo.clients_path.read_bytes() == clients_before
    assert repo.requests_path.read_bytes() == requests_before


def test_429_above_configured_wait_limit_falls_back_without_waiting(monkeypatch):
    session = Session()
    session.history.append({"role": "user", "content": "olá"})
    client = ScriptedClient([RateLimited("80")])
    waits = []
    monkeypatch.setattr("banco_agil.llm.time.sleep", waits.append)

    message = call_llm(
        client, "fake", AGENTS["triagem"], session, max_retries=1,
        max_rate_limit_wait_seconds=7,
    )

    assert message is None
    assert len(client.requests) == 1
    assert waits == []
    assert session.last_failure_category == "http_429"


@pytest.mark.parametrize("retry_after", [None, "invalido", "-1", "0.1", "8"])
def test_429_invalid_or_short_retry_after_uses_minimum_wait(monkeypatch, retry_after):
    session = Session()
    session.history.append({"role": "user", "content": "olá"})
    client = ScriptedClient([RateLimited(retry_after), reply("Olá!")])
    waits = []
    monkeypatch.setattr("banco_agil.llm.time.sleep", waits.append)

    message = call_llm(client, "fake", AGENTS["triagem"], session, max_retries=3)

    assert message.content == "Olá!"
    assert len(client.requests) == 2
    assert waits == [20.0]


def test_second_429_in_the_same_turn_does_not_retry_again(monkeypatch):
    session = Session()
    session.history.append({"role": "user", "content": "olá"})
    client = ScriptedClient([RateLimited("0"), RateLimited("0"), reply("não deve ser usada")])
    waits = []
    monkeypatch.setattr("banco_agil.llm.time.sleep", waits.append)

    message = call_llm(client, "fake", AGENTS["triagem"], session, max_retries=1)

    assert message is None
    assert len(client.requests) == 2
    assert waits == [20.0]
    assert session.last_failure_category == "http_429"


def test_rate_limit_retry_state_prevents_a_second_retry_across_pipeline_steps(monkeypatch):
    state = LlmRetryState()
    first_session = Session()
    first_session.history.append({"role": "user", "content": "primeiro passo"})
    second_session = Session()
    second_session.history.append({"role": "user", "content": "segundo passo"})
    first_client = ScriptedClient([RateLimited("0"), reply("primeiro concluído")])
    second_client = ScriptedClient([RateLimited("0"), reply("não deve ser usada")])
    waits = []
    monkeypatch.setattr("banco_agil.llm.time.sleep", waits.append)

    assert call_llm(first_client, "fake", AGENTS["triagem"], first_session, max_retries=1,
                    retry_state=state).content == "primeiro concluído"
    assert call_llm(second_client, "fake", AGENTS["triagem"], second_session, max_retries=1,
                    retry_state=state) is None

    assert len(first_client.requests) == 2
    assert len(second_client.requests) == 1
    assert waits == [20.0]


def test_tool_use_failed_retries_once_when_explicitly_enabled_without_mutating_history():
    session = Session()
    session.history.append({"role": "user", "content": "olá"})
    original_history = copy.deepcopy(session.history)
    client = ScriptedClient([ToolUseFailed(), reply("Olá!")])

    message = call_llm(
        client, "fake", AGENTS["triagem"], session,
        max_retries=0,
    )

    assert message.content == "Olá!"
    assert len(client.requests) == 2
    assert client.requests[0]["tools"] == client.requests[1]["tools"]
    assert client.requests[1]["messages"][0]["content"].startswith(client.requests[0]["messages"][0]["content"])
    assert session.history == original_history
    assert session.last_failure_category is None


@pytest.mark.parametrize("error", [ToolUseFailed("invalid_tool"), ToolUseFailed(None)])
def test_other_or_unstructured_http_400_does_not_retry(error):
    session = Session()
    session.history.append({"role": "user", "content": "olá"})
    client = ScriptedClient([error])

    message = call_llm(
        client, "fake", AGENTS["triagem"], session,
        max_retries=0,
    )

    assert message is None
    assert len(client.requests) == 1
    assert session.last_failure_category == "http_400"


def test_second_tool_use_failed_stops_after_the_single_specific_retry():
    session = Session()
    session.history.append({"role": "user", "content": "olá"})
    client = ScriptedClient([ToolUseFailed(), ToolUseFailed()])

    message = call_llm(
        client, "fake", AGENTS["triagem"], session,
        max_retries=1,
    )

    assert message is None
    assert len(client.requests) == 2
    assert session.last_failure_category == "http_400"


def test_selective_recovery_no_longer_depends_on_legacy_flag():
    session = Session()
    session.history.append({"role": "user", "content": "olá"})
    client = ScriptedClient([ToolUseFailed(), reply("Recuperado.")])

    message = call_llm(client, "fake", AGENTS["triagem"], session, max_retries=1)

    assert message.content == "Recuperado."
    assert len(client.requests) == 2
    assert session.last_failure_category is None


def test_enabled_tool_use_retry_does_not_change_429_policy(monkeypatch):
    monkeypatch.setattr("banco_agil.llm.time.sleep", lambda _: None)
    session = Session()
    session.history.append({"role": "user", "content": "olá"})
    client = ScriptedClient([RateLimited("0.1"), RateLimited("0.1")])

    message = call_llm(
        client, "fake", AGENTS["triagem"], session,
        max_retries=0,
    )

    assert message is None
    assert len(client.requests) == 2
    assert session.last_failure_category == "http_429"


def test_tool_use_failed_then_429_uses_each_bounded_recovery_once(monkeypatch):
    session = Session()
    session.history.append({"role": "user", "content": "olá"})
    client = ScriptedClient([ToolUseFailed(), RateLimited("0.2"), reply("Olá!")])
    waits = []
    monkeypatch.setattr("banco_agil.llm.time.sleep", waits.append)

    message = call_llm(
        client, "fake", AGENTS["triagem"], session,
        max_retries=1, max_rate_limit_wait_seconds=60,
    )

    assert message.content == "Olá!"
    assert len(client.requests) == 3
    assert waits == [20.0]


def test_financial_tool_after_tool_use_retry_has_one_persisted_effect(repo_ana_620):
    repo = repo_ana_620
    repo.update_score("11144477735", 1000)  # Somente a fixture temporária.
    client = ScriptedClient([
        ToolUseFailed(),
        reply(calls=[("solicitar_aumento_limite", {"novo_limite": 9000})]),
        reply("Seu pedido foi aprovado."),
    ])
    orch = Orchestrator(
        repo, ExchangeService(), client, "fake",
        max_llm_retries=0,
    )
    session = Session(active_agent="credito", authenticated=True, cpf="11144477735", nome="Ana Souza")

    response = orch.handle(session, "Quero aumentar meu limite para R$ 9.000.")

    assert "aprovado" in response.lower()
    assert len(client.requests) == 3
    assert [row["status_pedido"] for row in repo.list_requests()] == [STATUS_APROVADO]
    assert repo.get_client("11144477735")["limite_credito"] == 9000
    assert_history_valid(session.history)
    assert not client.script


def test_financial_tool_after_429_retry_has_one_persisted_effect(repo_ana_620, monkeypatch):
    repo = repo_ana_620
    repo.update_score("11144477735", 1000)  # Somente a fixture temporária.
    client = ScriptedClient([
        RateLimited("0"),
        reply(calls=[("solicitar_aumento_limite", {"novo_limite": 9000})]),
        reply("Seu pedido foi aprovado."),
    ])
    waits = []
    monkeypatch.setattr("banco_agil.llm.time.sleep", waits.append)
    orch = Orchestrator(
        repo, ExchangeService(), client, "fake",
        max_llm_retries=1, max_rate_limit_wait_seconds=60,
    )
    session = Session(active_agent="credito", authenticated=True, cpf="11144477735", nome="Ana Souza")

    response = orch.handle(session, "Quero aumentar meu limite para R$ 9.000.")

    assert "aprovado" in response.lower()
    assert waits == [20.0]
    assert len(client.requests) == 3
    assert [row["status_pedido"] for row in repo.list_requests()] == [STATUS_APROVADO]
    assert repo.get_client("11144477735")["limite_credito"] == 9000
    assert_history_valid(session.history)
    assert not client.script


@pytest.mark.parametrize(("value", "expected"), [("10", 20.0), ("150", 120.0), ("invalido", 60.0)])
def test_bootstrap_bounds_rate_limit_wait_setting(tmp_path, monkeypatch, value, expected):
    captured = {}

    class FakeOrchestrator:
        def __init__(self, _repo, *_args, **kwargs):
            captured.update(kwargs)
            self.credit = NS(reconcile=lambda: None)

    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("LLM_MAX_RATE_LIMIT_WAIT_SECONDS", value)
    monkeypatch.setattr(bootstrap, "load_dotenv", lambda *_args: False)
    monkeypatch.setattr(bootstrap, "setup_logging", lambda *_args: None)
    monkeypatch.setattr(bootstrap, "OpenAI", lambda **_kwargs: object())
    monkeypatch.setattr(bootstrap, "Orchestrator", FakeOrchestrator)

    bootstrap.build_orchestrator()

    assert captured["max_rate_limit_wait_seconds"] == expected


def _assert_request_protocol(request):
    pending_ids = []
    for message in request[1:]:  # a instrução de sistema não participa do protocolo de tools
        if message["role"] == "assistant":
            assert not pending_ids
            pending_ids = [call["id"] for call in message.get("tool_calls") or []]
        elif message["role"] == "tool":
            assert pending_ids and message["tool_call_id"] == pending_ids.pop(0)
        elif message["role"] == "user":
            assert not pending_ids
    assert not pending_ids


def test_two_turn_window_keeps_separate_interview_correction_and_reanalysis(repo):
    """O estado estruturado deve sobreviver a uma entrevista longa sem o histórico antigo."""
    bruno = "52998224725"
    corrected_income = {"renda_mensal": 6000}
    scripted = [
        reply("Como você descreveria sua renda mensal?"),
        reply(calls=[("registrar_respostas_entrevista", {"renda_mensal": 5000})]),
        reply("Entendi renda de R$ 5.000,00. Confirma?"),
        reply(calls=[("registrar_respostas_entrevista", corrected_income)]),
        reply("Certo, renda de R$ 6.000,00. Confirma?"),
        reply("Qual é o seu tipo de emprego?"),
        reply(calls=[("registrar_respostas_entrevista", {"tipo_emprego": "formal"})]),
        reply("Emprego formal. Confirma?"),
        reply("Quais são as despesas fixas mensais?"),
        reply(calls=[("registrar_respostas_entrevista", {"despesas_fixas": 2000})]),
        reply("Despesas de R$ 2.000,00. Confirma?"),
        reply("Quantos dependentes você possui?"),
        reply(calls=[("registrar_respostas_entrevista", {"num_dependentes": 1})]),
        reply("Um dependente. Confirma?"),
        reply("Você possui dívidas ativas?"),
        reply(calls=[("registrar_respostas_entrevista", {"tem_dividas": False})]),
        reply("Sem dívidas ativas. Confirma?"),
        reply(calls=[("registrar_entrevista_e_calcular_score", SCORE_ANSWERS)]),
        reply("Seu pedido de R$ 7.000,00 foi aprovado."),
    ]
    client = FakeLLM(scripted)
    orch = Orchestrator(repo, ExchangeService(), client, "fake")
    session = Session(active_agent="credito", authenticated=True, cpf=bruno, nome="Bruno Lima")

    rejected = orch.tool_executor.execute(
        AGENTS["credito"], session, "solicitar_aumento_limite", json.dumps({"novo_limite": 7000}),
    )
    assert rejected["status_pedido"] == STATUS_REPROVADO

    assert "renda" in orch.handle(session, "sim").lower()
    assert "confirma" in orch.handle(session, "Minha renda é cinco mil reais.").lower()
    assert session.flow.pending_interview_answers == {"renda_mensal": 5000}
    assert "confirma" in orch.handle(session, "Não, minha renda é seis mil reais.").lower()
    assert session.flow.pending_interview_answers == corrected_income

    assert "emprego" in orch.handle(session, "sim").lower()
    assert "confirma" in orch.handle(session, "Sou CLT.").lower()
    assert "despesas" in orch.handle(session, "sim").lower()
    assert "confirma" in orch.handle(session, "Minhas despesas são dois mil reais.").lower()
    assert "dependentes" in orch.handle(session, "sim").lower()
    assert "confirma" in orch.handle(session, "Tenho um filho.").lower()
    assert "dívidas" in orch.handle(session, "sim").lower()
    assert "confirma" in orch.handle(session, "Não tenho dívidas ativas.").lower()
    final = orch.handle(session, "sim")

    assert "aprovado" in final.lower()
    assert repo.get_client(bruno)["score"] == 570
    assert repo.get_client(bruno)["limite_credito"] == 7000
    assert session.desired_limit == 7000
    assert session.active_agent == "credito"
    assert [item["status_pedido"] for item in repo.list_requests()] == [STATUS_REPROVADO, STATUS_APROVADO]
    assert all(sum(message["role"] == "user" for message in request) <= 2 for request in client.requests)
    for request in client.requests:
        _assert_request_protocol(request)
    assert_history_valid(session.history)
    client.assert_exhausted()


@pytest.mark.parametrize("agent", ["triagem", "credito", "entrevista", "cambio"])
def test_global_close_remains_deterministic_after_multiple_turns(repo, agent):
    client = FakeLLM([])
    orch = Orchestrator(repo, ExchangeService(), client, "fake")
    session = orch.new_session()
    session.active_agent = agent
    session.authenticated = agent != "triagem"
    session.cpf = "11144477735" if session.authenticated else None
    session.nome = "Ana Souza" if session.authenticated else None
    if agent == "entrevista":
        FlowController.start_interview(session)
    for index in range(3):
        session.history.extend([
            {"role": "user", "content": f"turno anterior {index}"},
            {"role": "assistant", "content": "Continuidade registrada."},
        ])

    response = orch.handle(session, "encerrar o atendimento")

    assert "encerrado" in response.lower()
    assert session.ended and not client.requests
    assert_history_valid(session.history)
    client.assert_exhausted()


def test_input_estimate_includes_prompt_context_and_tools():
    session = _interview_session()
    session.history.extend([
        {"role": "user", "content": "contexto anterior"},
        {"role": "assistant", "content": "resposta anterior"},
        {"role": "user", "content": "mensagem atual"},
    ])
    payload = _build_payload(AGENTS["entrevista"], session)

    without_tools = _estimate_request_tokens(payload, [])
    with_tools = _estimate_request_tokens(payload, _tool_schemas(AGENTS["entrevista"], session))
    expanded_context = _estimate_request_tokens(
        [*payload, {"role": "user", "content": "detalhe adicional " * 100}],
        _tool_schemas(AGENTS["entrevista"], session),
    )

    assert with_tools > without_tools
    assert expanded_context > with_tools


def test_oversized_message_is_preserved_but_not_sent_or_replayed(repo):
    client = ScriptedClient([reply("Conte apenas a sua renda mensal, por favor.")])
    orch = Orchestrator(repo, ExchangeService(), client, "fake")
    session = _interview_session()
    clients_before = repo.clients_path.read_bytes()
    requests_before = repo.requests_path.read_bytes()
    long_message = ("Tenho muitos detalhes sobre minha vida financeira. " * 1000) + "Minha renda é R$ 6.000,00."

    first = orch.handle(session, long_message)
    second = orch.handle(session, long_message)

    assert "mensagem muito extensa" in first.lower()
    assert "mensagem muito extensa" in second.lower()
    assert not client.requests
    oversized_messages = [
        item for item in session.history
        if item.get("role") == "user" and item.get("content") == long_message
    ]
    assert len(oversized_messages) == 2
    assert all(item.get("_skip_llm_context") for item in oversized_messages)
    assert repo.clients_path.read_bytes() == clients_before
    assert repo.requests_path.read_bytes() == requests_before

    response = orch.handle(session, "Minha renda mensal é R$ 6.000,00.")

    assert "renda" in response.lower()
    assert len(client.requests) == 1
    assert long_message not in str(client.requests[0]["messages"])
    assert session.last_failure_category is None
    assert repo.clients_path.read_bytes() == clients_before
    assert repo.requests_path.read_bytes() == requests_before
