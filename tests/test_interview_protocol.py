"""Falhas de geração da entrevista são recuperadas sem repetir operações."""
from __future__ import annotations

import copy
import json

import pytest

from banco_agil.agents import AGENTS
from banco_agil.exchange import ExchangeService
from banco_agil.flow import Awaiting, FlowController
from banco_agil.llm import _tool_schemas, call_llm
from banco_agil.orchestrator import Orchestrator
from banco_agil.session import Session
from fakes import FakeLLM, assert_history_valid, reply

from http_fakes import provider
from support import snapshot_csv as snapshot


ANSWERS = dict(renda_mensal=6000, tipo_emprego="formal", despesas_fixas=2000,
               num_dependentes=1, tem_dividas=False)
MALFORMED_NAME = "registrar_respostas_entrevista<|channel|>commentary"


def session_for_interview():
    session = Session(authenticated=True, cpf="52998224725", nome="Bruno Lima", active_agent="entrevista")
    FlowController.start_interview(session)
    return session


def completion(*calls, text=None):
    return {"id": "local-completion", "object": "chat.completion", "created": 0,
            "model": "openai/gpt-oss-20b", "choices": [{"index": 0, "finish_reason": "tool_calls" if calls else "stop",
            "message": {"role": "assistant", "content": text, "tool_calls": [
                {"id": f"c{i}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
                for i, (name, args) in enumerate(calls)] or None}}]}


@pytest.mark.parametrize("text", ["sim", "sim, quero entrevista", "sim, quero a entrevista", "quero entrevista"])
def test_accepting_offer_keeps_ai_question_without_premature_operations(repo, text):
    client = FakeLLM([reply("Sobre qual aspecto da sua renda gostaria de começar?")])
    captured = []
    original = client.chat.completions.create
    def create(**kwargs):
        captured.append(kwargs)
        return original(**kwargs)
    client.chat.completions.create = create
    orch = Orchestrator(repo, ExchangeService(), client, "openai/gpt-oss-20b")
    session = Session(authenticated=True, cpf="52998224725", active_agent="credito")
    session.flow.awaiting = Awaiting.INTERVIEW_OFFER
    before = snapshot(repo)

    assert "?" in orch.handle(session, text)
    assert session.active_agent == "entrevista" and session.flow.interview_started
    assert session.flow.interview_answers == session.flow.pending_interview_answers == {}
    assert len(client.requests) == 1
    names = {tool['function']['name'] for tool in captured[0]['tools']}
    assert not {'registrar_respostas_entrevista', 'registrar_entrevista_e_calcular_score'} & names
    assert {'encerrar_atendimento', 'transferir_para_agente'}.issubset(names)
    assert snapshot(repo) == before


@pytest.mark.parametrize("args", [
    {}, {"renda_mensal": None}, {"renda_mensal": "não informada"}, {"renda_mensal": "6000"},
    {"renda_mensal": True}, {"renda_mensal": -1}, {"renda_mensal": float("nan")},
    {"renda_mensal": float("inf")}, {"renda_mensal": []}, {"tipo_emprego": "inventado"},
    {"num_dependentes": True}, {"num_dependentes": -1}, {"tem_dividas": "não"},
    {"renda_mensal": None, "tipo_emprego": "formal"}, {"renda_mensal": 6000, "extra": 1},
])
def test_invalid_partial_answers_do_not_reach_handler_or_mutate_state(repo, monkeypatch, args):
    orch = Orchestrator(repo, ExchangeService(), FakeLLM([]), "fake")
    session = session_for_interview()
    before = snapshot(repo)
    monkeypatch.setattr(orch.tool_executor, "execute", lambda *_a: pytest.fail("handler não deve executar"))

    result = orch.tool_executor.execute_from_llm(
        AGENTS["entrevista"], session, "registrar_respostas_entrevista", json.dumps(args),
    )

    assert result["codigo"] == "argumentos_invalidos"
    assert session.flow.interview_answers == session.flow.pending_interview_answers == {}
    assert session.authenticated and session.auth_failures == 0 and not session.ended
    assert snapshot(repo) == before


@pytest.mark.parametrize("args", [{"renda_mensal": 0}, {"tipo_emprego": "formal"},
                                  {"despesas_fixas": 0, "num_dependentes": 0, "tem_dividas": False}])
def test_legitimate_partial_answers_remain_pending_until_customer_confirmation(repo, args):
    orch = Orchestrator(repo, ExchangeService(), FakeLLM([]), "fake")
    session = session_for_interview()
    before = snapshot(repo)
    result = orch.tool_executor.execute_from_llm(
        AGENTS["entrevista"], session, "registrar_respostas_entrevista", json.dumps(args),
    )
    assert result["confirmacao_necessaria"] is True
    assert session.flow.pending_interview_answers == args
    assert session.flow.interview_answers == {}
    assert snapshot(repo) == before


@pytest.mark.parametrize("field", list(ANSWERS))
@pytest.mark.parametrize("missing", [True, False])
def test_score_requires_every_field_and_rejects_null_even_with_confirmed_data(repo, field, missing):
    orch = Orchestrator(repo, ExchangeService(), FakeLLM([]), "fake")
    session = session_for_interview()
    session.flow.interview_answers = dict(ANSWERS)
    args = dict(ANSWERS)
    args.pop(field) if missing else args.update({field: None})
    before = snapshot(repo)
    result = orch.tool_executor.execute_from_llm(
        AGENTS["entrevista"], session, "registrar_entrevista_e_calcular_score", json.dumps(args),
    )
    assert result["codigo"] == "argumentos_invalidos"
    assert snapshot(repo) == before
    assert session.flow.interview_answers == ANSWERS


@pytest.mark.parametrize("name,args", [
    (MALFORMED_NAME, {"renda_mensal": 6000}),
    ("registrar_respostas_entrevista", {"renda_mensal": None}),
    ("registrar_respostas_entrevista", {}),
])
def test_entire_invalid_batch_is_rejected_before_valid_score_or_history_replay(repo, provider, name, args):
    client, requests, responses = provider
    responses.append((200, completion(("registrar_entrevista_e_calcular_score", ANSWERS), (name, args))))
    orch = Orchestrator(repo, ExchangeService(), client, "openai/gpt-oss-20b")
    session = session_for_interview()
    session.flow.interview_answers = dict(ANSWERS)
    before = snapshot(repo)

    text = orch.handle(session, "pode concluir")

    assert text == "Seus dados confirmados foram preservados; não repeti o cálculo do score."
    assert session.last_failure_category == "invalid_tool_response"
    assert len(requests) == 1 and requests[0]["model"] == "openai/gpt-oss-20b"
    assert not any(m.get("tool_calls") for m in session.history)
    assert snapshot(repo) == before
    assert MALFORMED_NAME not in json.dumps(session.history)
    assert_history_valid(session.history)


@pytest.mark.parametrize("bad_case", ["null", "malformed_name"])
def test_distinct_http_400_errors_preserve_interview_after_single_recovery(repo, provider, bad_case, caplog):
    client, requests, responses = provider
    message = ("renda_mensal: null does not match number" if bad_case == "null" else
               f"tool {MALFORMED_NAME} not in request.tools")
    responses.extend([(400, {"error": {"type": "invalid_request_error", "code": "tool_use_failed", "message": message}})] * 2)
    orch = Orchestrator(repo, ExchangeService(), client, "openai/gpt-oss-20b")
    session = session_for_interview()
    session.flow.interview_answers = {"tipo_emprego": "formal"}
    before = snapshot(repo)

    text = orch.handle(session, "ainda não informei minha renda")

    assert text == "Qual é a sua renda mensal?"
    assert session.last_failure_category == "http_400"
    assert len(requests) == 2
    assert session.flow.interview_answers == {"tipo_emprego": "formal"}
    assert session.flow.pending_interview_answers == {}
    assert snapshot(repo) == before
    assert message not in caplog.text
    names = [t["function"]["name"] for t in requests[0]["tools"]]
    assert "registrar_respostas_entrevista" in names
    assert MALFORMED_NAME not in names
    assert MALFORMED_NAME not in json.dumps(requests[0])


def test_real_sdk_sends_exact_registered_name_and_partial_number_schema(provider):
    client, requests, responses = provider
    responses.append((200, completion(("registrar_respostas_entrevista", {"renda_mensal": 6000}))))
    session = session_for_interview()
    session.history = [{"role": "user", "content": "minha renda é 6000"}]
    result = call_llm(client, "openai/gpt-oss-20b", AGENTS["entrevista"], session)
    assert result.tool_calls[0].function.name == "registrar_respostas_entrevista"
    tool = next(t["function"] for t in requests[0]["tools"] if t["function"]["name"] == "registrar_respostas_entrevista")
    assert tool["parameters"]["properties"]["renda_mensal"]["type"] == "number"
    assert tool["parameters"]["minProperties"] == 1
    assert tool["parameters"]["additionalProperties"] is False
    assert tool["parameters"]["required"] == []


def test_valid_interview_still_confirms_and_calculates_only_collected_data(repo):
    client = FakeLLM([
        reply(calls=[("registrar_respostas_entrevista", ANSWERS)]), reply("Confirma esses cinco dados?"),
        reply(calls=[("registrar_entrevista_e_calcular_score", ANSWERS)]), reply("Seu score foi atualizado."),
    ])
    orch = Orchestrator(repo, ExchangeService(), client, "fake")
    session = session_for_interview()
    before = snapshot(repo)

    assert "Confirma" in orch.handle(session, "Ganho 6000, sou CLT, gasto 2000, um dependente, sem dívidas.")
    assert snapshot(repo) == before
    assert session.flow.pending_interview_answers == ANSWERS
    assert "atualizado" in orch.handle(session, "sim")
    assert repo.get_client(session.cpf)["score"] == 570
    assert repo.get_client(session.cpf)["limite_credito"] == 2000
    assert repo.list_requests() == []
    client.assert_exhausted()
    assert_history_valid(session.history)


def test_score_schema_only_appears_after_all_confirmations():
    session = session_for_interview()
    for missing_field in ANSWERS:
        session.flow.interview_answers = {k: v for k, v in ANSWERS.items() if k != missing_field}
        assert "registrar_entrevista_e_calcular_score" not in {
            t["function"]["name"] for t in _tool_schemas(AGENTS["entrevista"], session)
        }
    session.flow.interview_answers = dict(ANSWERS)
    assert "registrar_entrevista_e_calcular_score" in {
        t["function"]["name"] for t in _tool_schemas(AGENTS["entrevista"], session)
    }


def test_invalid_response_preserves_pending_confirmation_and_session_isolation(repo):
    orch = Orchestrator(repo, ExchangeService(), FakeLLM([reply(calls=[(MALFORMED_NAME, {})])]), "fake")
    session, other = session_for_interview(), session_for_interview()
    session.flow.pending_interview_answers = {"renda_mensal": 6000}
    before = copy.deepcopy(other.flow)
    assert "me envie **confirmo** para continuar" in orch.handle(session, "o resumo anterior")
    assert session.flow.pending_interview_answers == {"renda_mensal": 6000}
    assert session.flow.interview_answers == {}
    assert other.flow == before


@pytest.mark.parametrize("status", [401, 403, 429, 500])
def test_provider_failure_preserves_data_with_bounded_429_recovery(repo, provider, status, monkeypatch):
    monkeypatch.setattr("banco_agil.llm.time.sleep", lambda _: None)
    client, requests, responses = provider
    responses.extend([(status, {"error": {"message": "local provider failure"}})] * (2 if status == 429 else 1))
    orch = Orchestrator(repo, ExchangeService(), client, "openai/gpt-oss-20b")
    session = session_for_interview()
    before = snapshot(repo)
    orch.handle(session, "minha renda é 6000")
    assert session.last_failure_category == f"http_{status}"
    assert len(requests) == (2 if status == 429 else 1)
    assert session.flow.interview_answers == session.flow.pending_interview_answers == {}
    assert snapshot(repo) == before


def test_tool_arguments_cannot_replace_confirmed_values(repo):
    orch = Orchestrator(repo, ExchangeService(), FakeLLM([]), "fake")
    session = session_for_interview()
    session.flow.interview_answers = dict(ANSWERS)
    before = snapshot(repo)
    result = orch.tool_executor.execute_from_llm(
        AGENTS["entrevista"], session, "registrar_entrevista_e_calcular_score",
        json.dumps({**ANSWERS, "renda_mensal": 9000}),
    )
    assert result["codigo"] == "dados_entrevista_nao_conferem"
    assert snapshot(repo) == before
    assert session.flow.interview_answers == ANSWERS
