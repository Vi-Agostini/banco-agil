"""Bruno (score 350): pedido de R$ 7.000 reprovado -> entrevista -> score 570 -> reanálise aprovada."""
import pytest
import json
import logging

import banco_agil.llm as llm_module
from banco_agil.exchange import ExchangeService
from banco_agil.flow import Awaiting
from banco_agil.messages import CLOSED, LOOP_GUARD_REPLY, REPEAT_GUARD_REPLY
from banco_agil.orchestrator import Orchestrator
from banco_agil.statuses import STATUS_APROVADO, STATUS_REPROVADO
from banco_agil.tools import MSG_AUTH_FAILED, MSG_FAREWELL, MSG_INVALID_INPUT_LIMIT
from fakes import FakeLLM, assert_history_valid, reply

from orchestrator_fakes import make, logged_in


def test_fake_llm_assert_exhausted_detects_unused_replies():
    FakeLLM([]).assert_exhausted()
    with pytest.raises(AssertionError, match="unused response"):
        FakeLLM([reply("unused")]).assert_exhausted()


def test_full_flow_with_implicit_handoff(repo_ana_620):
    orch = make(repo_ana_620, [
        reply("Olá, Ana Souza. Como posso ajudar?"),
        reply(calls=[("consultar_limite", {})]),
        reply("Seu limite atual é R$ 5.000,00."),
    ])
    s = orch.new_session()
    assert "Olá, Ana" in orch.handle(s, "11144477735 12/05/1990")
    assert s.authenticated and s.active_agent == "triagem"
    assert "5.000" in orch.handle(s, "quero ver meu limite")  # uma única resposta, sem aviso de transição
    assert s.active_agent == "credito"
    assert s.tool_facts["transferir_para_agente"]["result"]["ok"] is True
    orch.client.assert_exhausted()
    assert_history_valid(s.history)


def test_tool_out_of_scope_is_blocked(repo):
    orch = make(repo, [reply(calls=[("consultar_limite", {})])])
    s = orch.new_session()
    response = orch.handle(s, "meu limite?")
    assert "CPF" in response
    assert s.last_failure_category == "invalid_tool_response"
    assert len(orch.client.requests) == 1
    assert not s.authenticated
    orch.client.assert_exhausted()


# Encerramento local.
def test_end_conversation_is_deterministic(repo):
    orch = make(repo, [])
    s = orch.new_session()
    assert orch.handle(s, "quero encerrar") == MSG_FAREWELL
    assert s.ended and not orch.client.requests
    assert any(
        call["function"]["name"] == "encerrar_atendimento"
        for message in s.history
        for call in message.get("tool_calls", [])
    )
    assert orch.handle(s, "oi") == CLOSED
    assert_history_valid(s.history)
    orch.client.assert_exhausted()


@pytest.mark.parametrize(("agent", "state"), [
    ("triagem", "prelogin"),
    ("triagem", "birth_date"),
    ("credito", "interview_offer"),
    ("credito", "limit_amount"),
    ("entrevista", "interview"),
    ("cambio", "exchange"),
    ("credito", "credit"),
])
def test_global_close_interceptor_ends_every_state_via_tool(repo, agent, state):
    orch = make(repo, [])
    s = orch.new_session()
    s.active_agent = agent
    s.authenticated = state not in {"prelogin", "birth_date"}
    if state == "birth_date":
        s.flow.pending_cpf = "11144477735"
        s.flow.awaiting = Awaiting.BIRTH_DATE
    elif state == "interview_offer":
        s.flow.awaiting = Awaiting.INTERVIEW_OFFER
    elif state == "limit_amount":
        s.flow.awaiting = Awaiting.LIMIT_AMOUNT
        s.flow.input_kind = "limit"

    assert orch.handle(s, "quero encerrar o atendimento") == MSG_FAREWELL
    assert s.ended and not orch.client.requests
    assert any(
        call["function"]["name"] == "encerrar_atendimento"
        for message in s.history
        for call in message.get("tool_calls", [])
    )
    assert orch.handle(s, "oi") == CLOSED
    assert_history_valid(s.history)
    orch.client.assert_exhausted()


def test_question_during_pending_limit_goes_to_llm_and_preserves_state(repo_ana_620):
    orch = make(repo_ana_620, [reply(calls=[("consultar_limite", {})]), reply("Seu limite atual é R$ 5.000,00.")])
    s = logged_in(orch, agent="credito")
    s.flow.awaiting = Awaiting.LIMIT_AMOUNT
    s.flow.input_kind = "limit"
    s.flow.input_prompt = "Informe o valor desejado para o novo limite."

    assert "5.000" in orch.handle(s, "qual é o meu limite atual?")
    assert len(orch.client.requests) == 2
    assert s.flow.awaiting is Awaiting.LIMIT_AMOUNT and s.flow.input_kind == "limit"
    assert any("Contexto: aguardando o valor do novo limite desejado" in m["content"] for m in orch.client.requests[0] if m["role"] == "system")
    orch.client.assert_exhausted()


def test_unexpected_reply_during_pending_interview_offer_goes_to_llm(repo_ana_620):
    orch = make(repo_ana_620, [reply(calls=[("consultar_limite", {})]), reply("Seu limite atual é R$ 5.000,00.")])
    s = logged_in(orch, agent="credito")
    s.flow.awaiting = Awaiting.INTERVIEW_OFFER

    assert "5.000" in orch.handle(s, "qual é o meu limite atual?")
    assert len(orch.client.requests) == 2
    assert s.flow.awaiting is Awaiting.INTERVIEW_OFFER and s.active_agent == "credito"
    assert any("Contexto: o cliente foi perguntado se aceita a entrevista financeira" in m["content"] for m in orch.client.requests[0] if m["role"] == "system")
    orch.client.assert_exhausted()


def test_llm_path_approved_limit_request_clears_awaiting_amount(repo_ana_620):
    orch = make(repo_ana_620, [
        reply(calls=[("solicitar_aumento_limite", {"novo_limite": 8000})]),
        reply("Aprovado!"),
        reply("Certo."),
    ])
    s = logged_in(orch, agent="credito")
    s.flow.awaiting = Awaiting.LIMIT_AMOUNT
    s.flow.input_kind = "limit"

    orch.handle(s, "quero 8000 reais")

    assert [r["status_pedido"] for r in repo_ana_620.list_requests()] == [STATUS_APROVADO]
    assert s.flow.awaiting is Awaiting.NONE
    orch.handle(s, "2")
    assert len(repo_ana_620.list_requests()) == 1
    orch.client.assert_exhausted()


def test_llm_path_rejected_limit_request_leaves_interview_offer_pending(repo):
    bruno = "52998224725"
    orch = make(repo, [
        reply(calls=[("solicitar_aumento_limite", {"novo_limite": 7000})]),
        reply('Como você descreveria sua renda mensal hoje?'),
    ])
    s = logged_in(orch, cpf=bruno, nome="Bruno Lima", agent="credito")
    s.flow.awaiting = Awaiting.LIMIT_AMOUNT
    s.flow.input_kind = "limit"

    orch.handle(s, "quero 7000 reais")

    assert [r["status_pedido"] for r in repo.list_requests()] == [STATUS_REPROVADO]
    assert s.flow.awaiting is Awaiting.INTERVIEW_OFFER
    assert "renda" in orch.handle(s, "sim").lower()
    assert s.active_agent == "entrevista"
    orch.client.assert_exhausted()


def test_transfer_to_another_agent_clears_previous_agent_flow_state(repo):
    orch = make(repo, [
        reply(calls=[("transferir_para_agente", {"destino": "cambio"})]),
        reply("A cotação está atualizada."),
    ])
    s = logged_in(orch, agent="credito")
    s.flow.awaiting = Awaiting.LIMIT_AMOUNT
    s.flow.input_kind = "limit"
    s.flow.input_prompt = "Informe o novo limite."
    s.flow.attempts = 2

    orch.handle(s, "consulte o dólar")

    assert s.active_agent == "cambio"
    assert s.flow.awaiting is Awaiting.NONE
    assert s.flow.input_kind is None and s.flow.input_prompt is None
    assert s.flow.attempts == 0
    orch.client.assert_exhausted()


def test_successful_quote_during_pending_limit_preserves_limit_state(repo):
    orch = make(repo, [
        reply(calls=[("consultar_cotacao", {"moeda": "USD"})]),
        reply("A cotação está atualizada."),
    ])
    orch.fx.get_rate = lambda currency: {"moeda": currency, "cotacao": 5.0}
    s = logged_in(orch, agent="cambio")
    s.flow.awaiting = Awaiting.LIMIT_AMOUNT
    s.flow.input_kind = "limit"

    orch.handle(s, "qual a cotação do dólar?")

    assert s.flow.awaiting is Awaiting.LIMIT_AMOUNT and s.flow.input_kind == "limit"
    orch.client.assert_exhausted()


def test_three_auth_failures_end_session(repo):
    orch = make(repo, [])
    s = orch.new_session()
    wrong = "11144477735 01/01/2000"
    orch.handle(s, wrong); orch.handle(s, wrong)
    assert not s.ended
    assert orch.handle(s, wrong) == MSG_AUTH_FAILED
    assert s.ended and not s.authenticated and s.auth_failures == 3
    assert orch.handle(s, "mais uma") == CLOSED
    assert not orch.client.requests
    orch.client.assert_exhausted()


def test_invalid_login_restarts_generically_and_ends_after_three_attempts(repo):
    orch = make(repo, [])
    s = orch.new_session()

    first = orch.handle(s, "11144477735 (31/02/1990)")
    second = orch.handle(s, "11144477735 (31/02/1990)")
    assert "mais duas tentativas" in first
    assert "mais uma tentativa" in second
    assert orch.handle(s, "11144477735 (31/02/1990)") == MSG_AUTH_FAILED
    assert s.auth_failures == 3 and s.ended
    assert orch.client.requests == []
    orch.client.assert_exhausted()


def test_no_tool_runs_after_end_in_same_batch(repo_ana_620):
    orch = make(repo_ana_620, [reply(calls=[("encerrar_atendimento", {}), ("solicitar_aumento_limite", {"novo_limite": 8000})])])
    s = logged_in(orch, agent="credito")
    orch.handle(s, "qual é o meu limite atual?")
    assert not s.ended and s.last_failure_category == "invalid_tool_response"
    tools_out = [m["content"] for m in s.history if m["role"] == "tool"]
    assert tools_out == []
    assert repo_ana_620.get_client("11144477735")["limite_credito"] == 5000
    assert repo_ana_620.list_requests() == []
    orch.client.assert_exhausted()
    assert_history_valid(s.history)


def test_end_during_interview(repo):
    orch = make(repo, [reply(calls=[("encerrar_atendimento", {})])])
    s = logged_in(orch, agent="entrevista")
    assert orch.handle(s, "desisti, pode encerrar") == MSG_FAREWELL and s.ended
    orch.client.assert_exhausted()


def test_end_available_in_every_agent(repo):
    for agent in ("triagem", "credito", "entrevista", "cambio"):
        orch = make(repo, [])
        s = logged_in(orch, agent=agent)
        assert orch.handle(s, "tchau") == MSG_FAREWELL and s.ended
        orch.client.assert_exhausted()


# Autorização pelo orquestrador.
def test_unauthenticated_cannot_reach_credit_tools(repo):
    orch = make(repo, [reply(calls=[("solicitar_aumento_limite", {"novo_limite": 9000})])])
    s = orch.new_session()
    s.active_agent = "credito"  # mesmo com o agente "errado" ativo, a ferramenta nega
    orch.handle(s, "quero 9000")
    assert not s.authenticated and len(orch.client.requests) == 1
    assert not any(m["role"] == "tool" for m in s.history)
    assert s.last_failure_category == "invalid_tool_response"
    assert repo.list_requests() == []
    assert repo.list_requests() == []
    orch.client.assert_exhausted()


def test_model_cannot_pass_other_customer_cpf(repo):
    orch = make(repo, [
        reply(calls=[("consultar_limite", {"cpf": "52998224725"})]),
        reply(calls=[("solicitar_aumento_limite", {"novo_limite": 3000, "cpf": "52998224725"})]),
    ])
    s = logged_in(orch, agent="credito")
    orch.handle(s, "limite do Bruno"); orch.handle(s, "aumente o do Bruno")
    outs = [m["content"] for m in s.history if m["role"] == "tool"]
    assert all("Argumentos inválidos" in o for o in outs)
    assert repo.get_client("52998224725")["limite_credito"] == 2000 and repo.list_requests() == []
    orch.client.assert_exhausted()


# Tratamento de falhas.
def test_llm_failure_is_recorded_and_recoverable(repo):
    class Boom:
        chat = type("C", (), {"completions": type("D", (), {"create": staticmethod(lambda **k: (_ for _ in ()).throw(RuntimeError("429")))})})
    orch = Orchestrator(repo, ExchangeService(), Boom(), "fake")
    s = logged_in(orch, agent="credito")
    assert "indisponibilidade" in orch.handle(s, "oi").lower()
    assert "RuntimeError" in s.last_llm_response  # tipo do erro fica em last_llm_response (debug)
    assert s.last_failure_reason is not None  # motivo sanitizado exibido ao cliente/debug
    assert [m["role"] for m in s.history] == ["assistant", "user", "assistant"]  # exibido == registrado
    orch.client = FakeLLM([reply("Como posso ajudar com seu limite?")])
    assert "limite" in orch.handle(s, "oi de novo")
    assert s.last_failure_reason is None
    orch.client.assert_exhausted()
    assert_history_valid(s.history)


def test_authenticated_triage_reports_provider_outage_without_financial_mutation(repo, monkeypatch):
    monkeypatch.setattr(llm_module.time, "sleep", lambda _: None)
    class RateLimited(RuntimeError):
        status_code = 429

    class Boom:
        chat = type("C", (), {"completions": type("D", (), {
            "create": staticmethod(lambda **k: (_ for _ in ()).throw(RateLimited("provider detail")))
        })})

    orch = Orchestrator(repo, ExchangeService(), Boom(), "fake")
    session = logged_in(orch, agent="triagem")
    client_data = repo.clients_path.read_bytes()
    requests_data = repo.requests_path.read_bytes()

    response = orch.handle(session, "quero aumentar meu limite para 8000")

    assert "indisponibilidade" in response.lower()
    assert "diga novamente" not in response.lower()
    assert session.authenticated and session.active_agent == "credito" and not session.ended
    assert session.tool_facts["transferir_para_agente"]["result"]["ok"] is True
    assert repo.clients_path.read_bytes() == client_data
    assert repo.requests_path.read_bytes() == requests_data


def test_llm_retries_once_when_provider_requests_a_short_wait(repo, caplog, monkeypatch):
    class RateLimited(RuntimeError):
        status_code = 429
        headers = {"Retry-After": "0.1"}

    class Flaky:
        def __init__(self):
            self.attempts = 0
            self.chat = type("C", (), {"completions": type("D", (), {"create": self.create})})

        def create(self, **_kwargs):
            self.attempts += 1
            if self.attempts < 2:
                raise RateLimited("private provider detail")
            return reply("Posso ajudar com o limite.")

    client = Flaky()
    waits = []
    monkeypatch.setattr(llm_module.time, "sleep", waits.append)
    orch = Orchestrator(repo, ExchangeService(), client, "fake", max_llm_retries=2, test_id="CR-test")
    session = logged_in(orch)
    caplog.set_level(logging.INFO, logger="banco_agil")

    orch.handle(session, "texto-secreto 11144477735 token fake-sensitive")

    events = [json.loads(record.message) for record in caplog.records if record.message.startswith("{")]
    llm_events = [event for event in events if event["event"] == "llm_call"]
    assert client.attempts == 2
    assert waits == [20.0]
    assert [event["attempt"] for event in llm_events] == [1, 2]
    assert [event["status_http"] for event in llm_events] == [429, None]
    assert all(event["correlation_id"] == session.correlation_id for event in llm_events)
    assert all(event["test_id"] == "CR-test" for event in llm_events)
    assert "texto-secreto" not in caplog.text and "private provider detail" not in caplog.text
    assert "11144477735" not in caplog.text and "fake-sensitive" not in caplog.text


def test_credit_orchestrator_keeps_invalid_score_controlled_and_does_not_write(repo):
    orch = make(repo, [reply(calls=[("solicitar_aumento_limite", {"novo_limite": 8000})])])
    session = logged_in(orch, agent="credito")
    repo._update_client("11144477735", score=5000)
    clients_before = repo.clients_path.read_bytes()
    requests_before = repo.requests_path.read_bytes()

    response = orch.handle(session, "execute a avaliação solicitada")

    assert response and "aprovado" not in response.lower() and "reprovado" not in response.lower()
    assert "não" in response.lower() or "problema" in response.lower() or "inválid" in response.lower()
    assert repo.clients_path.read_bytes() == clients_before
    assert repo.requests_path.read_bytes() == requests_before
    orch.client.assert_exhausted()


def test_interview_llm_failure_uses_resume_message_and_keeps_diagnostic(repo):
    class BadRequest(RuntimeError):
        status_code = 400

    class Boom:
        chat = type("C", (), {"completions": type("D", (), {
            "create": staticmethod(lambda **k: (_ for _ in ()).throw(BadRequest("private provider detail")))
        })})

    orch = Orchestrator(repo, ExchangeService(), Boom(), "fake")
    s = logged_in(orch, agent="entrevista")
    s.flow.interview_answers = {
        "renda_mensal": 6000, "tipo_emprego": "formal", "despesas_fixas": 2000,
        "num_dependentes": 1,
    }

    response = orch.handle(s, "não")

    assert "indisponibilidade" in response.lower()
    assert "HTTP 400" in s.last_failure_reason
    assert "private provider detail" not in s.last_failure_reason


def test_triage_fallback_does_not_count_credit_request_as_auth_failure(repo):
    class Boom:
        chat = type("C", (), {"completions": type("D", (), {
            "create": staticmethod(lambda **k: (_ for _ in ()).throw(RuntimeError("API offline")))
        })})

    orch = Orchestrator(repo, ExchangeService(), Boom(), "fake")
    s = orch.new_session()

    response = orch.handle(s, "Quero mais limite")

    assert "Qual é o seu CPF?" in response
    assert "RuntimeError" in s.last_llm_response
    assert s.auth_failures == 0


def test_triage_greeting_is_handled_locally_without_auth_failure(repo):
    orch = make(repo, [])
    s = orch.new_session()

    assert "qual é o seu cpf" in orch.handle(s, "oi").lower()
    assert len(orch.client.requests) == 0
    assert s.auth_failures == 0 and not s.ended
    orch.client.assert_exhausted()


def test_triage_greeting_and_requests_do_not_count_as_auth_failures(repo):
    orch = make(repo, [
        reply("Entendi que deseja aumentar o limite. Primeiro, preciso do seu CPF."),
        reply("Claro, posso ajudar. Qual é o seu CPF?"),
    ])
    s = orch.new_session()

    for message in ("oi", "quero mais limite", "preciso de ajuda"):
        orch.handle(s, message)

    assert len(orch.client.requests) == 2
    assert s.auth_failures == 0 and not s.ended
    orch.client.assert_exhausted()


def test_triage_cpf_only_is_not_an_auth_attempt(repo):
    orch = make(repo, [])
    s = orch.new_session()

    assert "data de nascimento" in orch.handle(s, "11144477735")
    assert len(orch.client.requests) == 0
    assert s.auth_failures == 0 and not s.authenticated
    assert s.flow.pending_cpf == "11144477735"

    assert "identidade foi confirmada" in orch.handle(s, "12/05/1990")
    assert s.authenticated and s.auth_failures == 0
    assert len(orch.client.requests) == 0
    assert "11144477735" not in str(s.history) + str(s.transcript)
    assert "12/05/1990" not in str(s.history) + str(s.transcript)
    orch.client.assert_exhausted()


def test_birth_date_is_redacted_when_triage_is_waiting_for_it(repo):
    orch = make(repo, [])
    s = orch.new_session()

    orch.handle(s, "11144477735")
    orch.handle(s, "12/05/1990")

    stored = str(s.history) + str(s.transcript)
    assert s.authenticated
    assert "11144477735" not in stored and "12/05/1990" not in stored
    orch.client.assert_exhausted()


def test_authenticated_message_keeps_unrelated_dates_but_redacts_birth_date(repo):
    orch = make(repo, [reply("Entendi o pedido.")])
    s = logged_in(orch, agent="credito")
    message = (
        "Meu CPF é 11144477735; vencimento em 20/11/2026; nasci em 12/05/1990; "
        "quero pagar em 25/11/2026."
    )

    orch.handle(s, message)

    stored = str(s.history) + str(s.transcript) + (s.last_llm_payload or "")
    assert "11144477735" not in stored and "12/05/1990" not in stored
    assert "20/11/2026" in stored and "25/11/2026" in stored
    orch.client.assert_exhausted()


def test_cpf_with_due_date_is_not_treated_as_birth_credential(repo):
    orch = make(repo, [reply("Posso ajudar com esse vencimento.")])
    s = orch.new_session()

    orch.handle(s, "Meu CPF é 11144477735 e o vencimento da fatura é 20/11/2026")

    stored = str(s.history) + str(s.transcript) + (s.last_llm_payload or "")
    assert not s.authenticated and s.auth_failures == 0
    assert "11144477735" not in stored
    assert "20/11/2026" in stored
    orch.client.assert_exhausted()


def test_authentication_and_limit_request_in_one_message_uses_sanitized_history(repo_ana_620):
    orch = make(repo_ana_620, [
        reply(calls=[("consultar_limite", {})]),
        reply("Seu limite atual é R$ 5.000,00."),
    ])
    s = orch.new_session()

    response = orch.handle(
        s,
        "CPF 11144477735, nasci em 12/05/1990 e quero consultar meu limite",
    )

    payload = s.last_llm_payload or ""
    assert s.authenticated and s.active_agent == "credito"
    assert "5.000" in response
    assert "11144477735" not in payload and "12/05/1990" not in payload
    assert "consultar meu limite" in payload.lower()
    orch.client.assert_exhausted()


def test_invalid_birth_date_with_cpf_counts_failure_without_sending_credentials(repo):
    orch = make(repo, [])
    s = orch.new_session()

    response = orch.handle(s, "CPF 11144477735, nasci em 31/02/1990")

    assert "mais duas tentativas" in response.lower()
    assert s.auth_failures == 1 and not s.authenticated
    assert orch.client.requests == []
    assert "11144477735" not in str(s.history) + str(s.transcript)
    assert "31/02/1990" not in str(s.history) + str(s.transcript)
    orch.client.assert_exhausted()


def test_triage_llm_successful_authentication_continues_llm_loop(repo):
    orch = make(repo, [reply("Olá, Ana Souza. Como posso ajudar?")])
    s = orch.new_session()

    assert "Olá, Ana Souza" in orch.handle(s, "quero mais limite: 11144477735, nasci em 12/05/1990")
    assert s.authenticated and s.auth_failures == 0
    assert len(orch.client.requests) == 1
    assert "11144477735" not in (s.last_llm_payload or "")
    assert "12/05/1990" not in (s.last_llm_payload or "")
    assert "quero mais limite" in str(orch.client.requests).lower()
    orch.client.assert_exhausted()


def test_complete_credentials_authenticate_locally_when_provider_is_down(repo):
    class Boom:
        chat = type("C", (), {"completions": type("D", (), {
            "create": staticmethod(lambda **k: (_ for _ in ()).throw(RuntimeError("API offline")))
        })})

    orch = Orchestrator(repo, ExchangeService(), Boom(), "fake")
    s = orch.new_session()

    response = orch.handle(s, "11144477735 12/05/1990")

    assert "indisponibilidade técnica" in response
    assert s.authenticated and s.auth_failures == 0
    assert "11144477735" not in (s.last_llm_payload or "")
    assert "12/05/1990" not in (s.last_llm_payload or "")


def test_triage_fallback_stores_only_cpf_without_counting_attempt(repo):
    class Boom:
        chat = type("C", (), {"completions": type("D", (), {
            "create": staticmethod(lambda **k: (_ for _ in ()).throw(RuntimeError("API offline")))
        })})

    orch = Orchestrator(repo, ExchangeService(), Boom(), "fake")
    s = orch.new_session()

    response = orch.handle(s, "11144477735")

    assert "data de nascimento" in response
    assert s.flow.pending_cpf == "11144477735"
    assert s.auth_failures == 0 and not s.authenticated

    response = orch.handle(s, "12/05/1990")

    assert "Olá, Ana Souza" in response
    assert s.authenticated and s.flow.pending_cpf is None
    assert s.auth_failures == 0


@pytest.mark.parametrize("agent", ["credito", "entrevista", "cambio"])
def test_authenticated_api_failure_uses_unavailable_fallback(repo, agent):
    class Boom:
        chat = type("C", (), {"completions": type("D", (), {
            "create": staticmethod(lambda **k: (_ for _ in ()).throw(RuntimeError("API offline")))
        })})

    orch = Orchestrator(repo, ExchangeService(), Boom(), "fake")
    s = logged_in(orch, agent=agent)

    assert "indisponibilidade" in orch.handle(s, "ajuda").lower()


def test_more_limit_without_amount_asks_for_amount_instead_of_refusing(repo):
    orch = make(repo, [reply("Qual seria o novo limite desejado?")])
    s = logged_in(orch, agent="credito")

    response = orch.handle(s, "quero mais limite")

    assert "?" in response and s.flow.awaiting is Awaiting.LIMIT_AMOUNT
    assert len(orch.client.requests) == 1
    orch.client.assert_exhausted()


def test_invalid_limit_answer_does_not_repeat_the_same_question(repo):
    orch = make(repo, [reply(calls=[("solicitar_aumento_limite", {"novo_limite": 0})])])
    s = logged_in(orch, agent="credito")

    response = orch.handle(s, "0")

    assert "precisa ser maior que zero" in response
    assert "Nenhum pedido foi registrado" not in response
    assert "Tentativa" not in response
    assert "Qual valor de limite você gostaria de solicitar?" not in response
    assert len(orch.client.requests) == 1
    assert repo.list_requests() == []
    orch.client.assert_exhausted()


def test_large_limit_answer_is_processed_without_asking_again(repo):
    orch = make(repo, [reply("Qual seria o novo limite desejado?")])
    s = logged_in(orch, agent="credito")

    assert "novo limite desejado" in orch.handle(s, "quero aumentar meu limite")
    response = orch.handle(s, "999999999999")

    assert "foi reprovado pelo score atual" in response
    assert "Qual valor" not in response
    assert len(orch.client.requests) == 1
    assert repo.list_requests()[-1]["novo_limite_solicitado"] == "999999999999.00"
    orch.client.assert_exhausted()


def test_unparseable_limit_answer_gets_direct_error_not_repeated_question(repo):
    orch = make(repo, [reply("Qual seria o novo limite desejado?")])
    s = logged_in(orch, agent="credito")

    orch.handle(s, "quero aumentar meu limite")
    response = orch.handle(s, "999999999999999999")

    assert "O valor informado é inválido" in response
    assert "Não consegui interpretar esse número" in response
    assert "Qual valor" not in response
    assert len(orch.client.requests) == 1
    assert repo.list_requests() == []
    orch.client.assert_exhausted()


def test_four_invalid_limit_inputs_return_to_triage_after_three_warnings(repo):
    orch = make(repo, [reply("Qual seria o novo limite desejado?")])
    s = logged_in(orch, agent="credito")
    orch.handle(s, "quero aumentar meu limite")

    for attempt in range(1, 4):
        response = orch.handle(s, "999999999999999999999999999999")
        assert "O valor informado é inválido" in response
        assert "Tentativa" not in response
        assert "Nenhum pedido foi registrado" not in response
        assert s.flow.awaiting is Awaiting.LIMIT_AMOUNT
        assert not s.ended

    response = orch.handle(s, "999999999999999999999999999999")

    assert response == "O valor informado não é válido. Se quiser, posso ajudar com outra dúvida ou encerrar a conversa."
    assert s.active_agent == "triagem"
    assert s.flow.awaiting is Awaiting.NONE and not s.ended
    assert len(orch.client.requests) == 1
    orch.client.assert_exhausted()


def test_invalid_interview_reply_is_delegated_to_llm_without_state_mutation(repo_ana_620):
    orch = make(repo_ana_620, [reply("Não consegui interpretar esse valor. Pode explicar sua renda de outra forma?")])
    s = logged_in(orch, agent="entrevista")
    s.flow.interview_started = True
    response = orch.handle(s, "-1")

    assert "renda" in response
    assert s.active_agent == "entrevista" and not s.ended
    assert repo_ana_620.get_client("11144477735")["score"] == 620
    assert s.flow.interview_answers == {}
    assert s.flow.pending_interview_answers == {}
    orch.client.assert_exhausted()


def test_currency_clarifications_do_not_count_as_invalid_customer_attempts(repo):
    invalid = ("consultar_cotacao", {"moeda": "??"})
    orch = make(repo, [step for _ in range(4) for step in (
        reply(calls=[invalid]), reply("Qual moeda você deseja consultar? Pode informar o nome ou o país."))])
    s = logged_in(orch, agent="cambio")
    s.flow.input_kind = "currency"
    s.flow.input_prompt = "Informe a moeda que deseja consultar."

    for attempt in range(4):
        response = orch.handle(s, "??")
        assert "Qual moeda" in response and "Tentativa" not in response
        assert s.flow.attempts == 0 and s.flow.unclear_count == 0
    assert s.active_agent == "cambio" and not s.ended
    orch.client.assert_exhausted()


def test_interview_natural_reply_is_delegated_to_llm(repo):
    orch = make(repo, [reply("Entendi que você é registrado. Agora me conte sua renda mensal.")])
    s = logged_in(orch, agent="entrevista")
    s.flow.interview_started = True

    assert "renda" in orch.handle(s, "sou registrado").lower()
    assert s.flow.interview_answers == {}
    assert s.flow.pending_interview_answers == {}
    assert s.flow.attempts == 0
    assert s.flow.unclear_count == 0 and not s.ended
    orch.client.assert_exhausted()


def test_last_llm_response_is_available_for_debug(repo):
    orch = make(repo, [reply("Olá! Como posso ajudar?")])
    s = logged_in(orch)
    assert orch.handle(s, "oi") == "Olá!\n\nComo posso ajudar?"
    assert "Olá! Como posso ajudar?" in s.last_llm_response
    assert '"agente": "triagem"' in s.last_llm_response
    orch.client.assert_exhausted()


def test_model_instability_text_is_identified_in_debug(repo):
    orch = make(repo, [reply("Desculpe, estou com uma instabilidade momentânea.")])
    s = logged_in(orch)

    orch.handle(s, "oi")

    assert "texto do LLM" in s.last_failure_reason
    orch.client.assert_exhausted()


def test_empty_llm_reply_is_handled(repo):
    orch = make(repo, [reply("   ")])
    s = logged_in(orch)
    assert orch.handle(s, "oi") == "Desculpe, não consegui entender.\n\nPode reformular a sua mensagem?"
    assert_history_valid(s.history)
    assert s.history[-1] == {"role": "assistant", "content": "Desculpe, não consegui entender.\n\nPode reformular a sua mensagem?"}
    orch.client.assert_exhausted()


def test_malformed_response_object(repo):
    class Weird:
        chat = type("C", (), {"completions": type("D", (), {"create": staticmethod(lambda **k: object())})})
    orch = Orchestrator(repo, ExchangeService(), Weird(), "fake")
    s = logged_in(orch, agent="credito")
    assert "indisponibilidade" in orch.handle(s, "oi").lower()


def test_repeated_consultation_stops_early_with_confirmed_result_and_valid_history(repo_ana_620):
    repo = repo_ana_620
    orch = make(repo, [reply(calls=[("consultar_limite", {})]) for _ in range(2)], max_steps=3)
    s = logged_in(orch, agent="credito")
    assert orch.handle(s, "limite?") == "Seu limite de crédito atual é R$ 5.000,00."
    assert len(orch.client.requests) == 2
    assert s.last_failure_category == "progress_stalled"
    assert_history_valid(s.history)
    assert "repetida" in [m["content"] for m in s.history if m["role"] == "tool"][-1]
    orch.client.assert_exhausted()


def test_interview_invalid_reply_uses_llm_without_financial_write(repo_ana_620):
    orch = make(repo_ana_620, [reply("Esse dado ficou ambíguo. Pode esclarecer?")])
    s = logged_in(orch, agent="entrevista")
    s.flow.interview_started = True

    response = orch.handle(s, "-1")

    assert "ambíguo" in response
    assert len(orch.client.requests) == 1
    assert repo_ana_620.get_client("11144477735")["score"] == 620
    orch.client.assert_exhausted()
    assert_history_valid(s.history)


def test_three_consecutive_random_messages_end_session(repo):
    invalid = reply(calls=[("registrar_entrada_invalida", {})])
    orch = make(repo, [invalid, reply("Reformule, por favor."),
                       invalid, reply("Pode explicar de outra forma?"), invalid])
    s = logged_in(orch)

    orch.handle(s, "asdfgh")
    orch.handle(s, "qwer123")
    response = orch.handle(s, "zxcmnb")

    assert response == MSG_INVALID_INPUT_LIMIT
    assert s.flow.unclear_count == 3 and s.ended
    assert len(orch.client.requests) == 5
    orch.client.assert_exhausted()


def test_three_unclear_replies_end_session_across_agents(repo):
    invalid = reply(calls=[("registrar_entrada_invalida", {})])
    orch = make(repo, [invalid, reply("Reformule, por favor."),
                       invalid, reply("Pode explicar de outra forma?"), invalid])
    s = logged_in(orch)

    orch.handle(s, "asdfgh")
    s.active_agent = "credito"
    orch.handle(s, "qwer123")
    s.active_agent = "entrevista"
    s.flow.interview_answers = {
        "renda_mensal": 6000, "tipo_emprego": "formal", "despesas_fixas": 2000,
        "num_dependentes": 1, "tem_dividas": False,
    }
    response = orch.handle(s, "zxcmnb")

    assert response == MSG_INVALID_INPUT_LIMIT
    assert s.flow.unclear_count == 3 and s.ended
    orch.client.assert_exhausted()


def test_three_complete_invalid_auth_calls_end_triage(repo):
    orch = make(repo, [])
    s = orch.new_session()
    # Tentativas completas vêm do cliente, nunca de argumentos inventados pelo modelo.
    for _ in range(2):
        orch.handle(s, "11144477735 01/01/2000")
    response = orch.handle(s, "11144477735 01/01/2000")
    assert response == MSG_AUTH_FAILED
    assert s.ended and s.auth_failures == 3
    assert not orch.client.requests
    orch.client.assert_exhausted()


def test_valid_reply_resets_random_input_counter(repo):
    orch = make(repo, [reply(calls=[("registrar_entrada_invalida", {})]),
                       reply("Não entendi. Reformule, por favor."),
                       reply("Olá! Qual é o seu CPF?")])
    s = orch.new_session()

    orch.handle(s, "asdfgh")
    orch.handle(s, "quero acessar minha conta")

    assert s.flow.unclear_count == 0 and not s.ended
    orch.client.assert_exhausted()


def test_assistant_unclear_wording_does_not_increment_counter(repo):
    orch = make(repo, [reply("Não entendi. Pode reformular?"), reply("Uma dúvida compreendida.")])
    s = logged_in(orch)

    orch.handle(s, "mensagem aleatória")
    assert s.flow.unclear_count == 0
    orch.handle(s, "pergunta compreensível")
    assert s.flow.unclear_count == 0 and not s.ended
    orch.client.assert_exhausted()


def test_three_identical_greetings_use_repeat_guard_reply(repo):
    greeting = reply("Olá! Como posso ajudar?")
    orch = make(repo, [greeting, greeting, greeting])
    s = logged_in(orch)

    assert orch.handle(s, "oi") == "Olá!\n\nComo posso ajudar?"
    assert orch.handle(s, "olá") == "Olá!\n\nComo posso ajudar?"
    third = orch.handle(s, "bom dia")
    assert third == REPEAT_GUARD_REPLY and third != LOOP_GUARD_REPLY
    assert s.ended
    orch.client.assert_exhausted()


def test_different_reply_breaks_repeat_guard_sequence(repo):
    repeated = reply("Posso ajudar com seu limite?")
    orch = make(repo, [repeated, repeated, reply("Claro, diga como posso ajudar.")])
    s = logged_in(orch, agent="credito")

    orch.handle(s, "oi")
    orch.handle(s, "olá")
    assert orch.handle(s, "bom dia") == "Claro, diga como posso ajudar."
    assert not s.ended
    orch.client.assert_exhausted()


def test_state_change_breaks_repeat_guard_sequence(repo):
    repeated = reply("Olá! Como posso ajudar?")
    orch = make(repo, [repeated, repeated, repeated])
    s = logged_in(orch)

    orch.handle(s, "oi")
    orch.handle(s, "olá")
    s.auth_failures += 1
    assert orch.handle(s, "bom dia") == "Olá!\n\nComo posso ajudar?"
    assert not s.ended
    orch.client.assert_exhausted()


def test_repeated_api_failures_never_trigger_repeat_guard(repo):
    class Boom:
        chat = type("C", (), {"completions": type("D", (), {
            "create": staticmethod(lambda **k: (_ for _ in ()).throw(RuntimeError("API offline")))
        })})

    orch = Orchestrator(repo, ExchangeService(), Boom(), "fake")
    s = logged_in(orch, agent="credito")

    responses = [orch.handle(s, "oi") for _ in range(5)]

    assert all("indisponibilidade" in response.lower() for response in responses)
    assert not s.ended


def test_invalid_tool_name_and_bad_json(repo):
    orch = make(repo, [
        reply(calls=[("ferramenta_inexistente", {})]),
        reply(calls=[("consultar_limite", "{nao e json")]),
    ])
    s = logged_in(orch, agent="credito")
    orch.handle(s, "x"); orch.handle(s, "y")
    outs = [m["content"] for m in s.history if m["role"] == "tool"]
    assert outs == []
    assert repo.list_requests() == [] and s.last_failure_category == "invalid_tool_response"
    orch.client.assert_exhausted()


def test_transfer_loop_is_capped(repo):
    # Caminho válido com assinaturas distintas: testa o limite de transferências,
    # sem ser interrompido antes pela proteção de chamadas idênticas.
    script = [reply(calls=[("transferir_para_agente", {"destino": target})])
              for target in ("cambio", "triagem", "credito", "entrevista")]
    orch = make(repo, script, max_steps=8)
    s = logged_in(orch, agent="credito")
    assert orch.handle(s, "oi")
    assert sum("Limite de transferências" in m.get("content", "") for m in s.history if m["role"] == "tool") == 1
    outs = [m["content"] for m in s.history if m["role"] == "tool"]
    assert any("Limite de transferências" in o for o in outs)
    orch.client.assert_exhausted()


def test_credentials_removed_from_llm_history_and_masked_on_screen(repo):
    orch = make(repo, [reply("Olá, Ana. Como posso ajudar?")])
    s = orch.new_session()
    orch.handle(s, "meu cpf é 111.444.777-35 e nasci em 12/05/1990")
    dump = str(s.history) + str(s.transcript)
    for secret in ("111.444.777-35", "11144477735", "12/05/1990"):
        assert secret not in dump
    assert any("***.***.***-35" in c for r, c in s.transcript if r == "user")
    assert s.authenticated and len(orch.client.requests) == 1
    assert "111.444.777-35" not in str(orch.client.requests)
    assert "11144477735" not in (s.last_llm_payload or "")
    assert "12/05/1990" not in (s.last_llm_payload or "")
    orch.client.assert_exhausted()


@pytest.mark.parametrize("cpf", ["111.444.777-35", "11144477735"])
def test_complete_credentials_are_redacted_before_llm_but_intent_survives(repo, cpf):
    orch = make(repo, [reply("Olá, Ana. Vou ajudar com seu limite.")])
    s = orch.new_session()

    orch.handle(s, f"Quero consultar meu limite; meu CPF é {cpf} e nasci em 12/05/1990")

    sent = str(orch.client.requests)
    stored = str(s.history) + str(s.transcript) + (s.last_llm_payload or "")
    assert s.authenticated
    assert "11144477735" not in sent + stored
    assert "111.444.777-35" not in sent + stored
    assert "12/05/1990" not in sent + stored
    assert "consultar meu limite" in sent.lower()
    assert "***.***.***-35" in stored
    orch.client.assert_exhausted()


@pytest.mark.parametrize("agent", ["credito", "entrevista", "cambio"])
@pytest.mark.parametrize("cpf", ["111.444.777-35", "11144477735"])
def test_redacted_personal_data_does_not_break_agent_tasks(repo, agent, cpf, caplog):
    import logging
    caplog.set_level(logging.INFO, logger="banco_agil")
    calls = {
        "credito": ("consultar_limite", {}),
        "entrevista": ("registrar_entrevista_e_calcular_score", {
            "renda_mensal": 6000, "tipo_emprego": "formal", "despesas_fixas": 2000,
            "num_dependentes": 1, "tem_dividas": False,
        }),
        "cambio": ("consultar_cotacao", {"moeda": "USD"}),
    }
    orch = make(repo, [reply(calls=[calls[agent]]), reply("Operação concluída com os dados informados.")])
    fx_calls = []
    if agent == "cambio":
        orch.fx.get_rate = lambda code: (fx_calls.append(code) or {
            "moeda": code, "valor_em_brl": 5.5, "fonte": "fake",
        })
    s = logged_in(orch, agent=agent)
    if agent == "entrevista":
        s.flow.interview_answers = {
            "renda_mensal": 6000, "tipo_emprego": "formal", "despesas_fixas": 2000,
            "num_dependentes": 1, "tem_dividas": False,
        }
    intent = {
        "credito": "Quero consultar meu limite atual",
        "entrevista": "Quero registrar entrevista, renda 6000, emprego formal e despesas 2000",
        "cambio": "Quero consultar a cotação do dólar USD",
    }[agent]

    orch.handle(s, f"{intent}; CPF {cpf}; nasci em 12/05/1990")

    sent = str(orch.client.requests)
    transcript = str(s.transcript)
    assert "11144477735" not in sent + transcript
    assert "111.444.777-35" not in sent + transcript
    assert "12/05/1990" not in sent + transcript
    assert "11144477735" not in caplog.text and "12/05/1990" not in caplog.text
    assert intent.lower() in sent.lower()
    if agent == "entrevista":
        assert "6000" in sent
        assert s.active_agent == "credito"
        assert repo.get_client(s.cpf)["score"] == 570
    elif agent == "cambio":
        assert fx_calls == ["USD"]
    else:
        assert any("limite_credito_atual" in msg.get("content", "") for msg in s.history if msg["role"] == "tool")
    orch.client.assert_exhausted()


def test_error_log_has_no_sensitive_data(repo, caplog):
    import logging
    caplog.set_level(logging.INFO, logger="banco_agil")
    orch = make(repo, [reply("Olá, Ana. Como posso ajudar?")])
    s = orch.new_session()
    orch.handle(s, "quero ajuda; CPF 11144477735 e nasci em 12/05/1990")
    assert "11144477735" not in caplog.text and "12/05/1990" not in caplog.text
    assert "11144477735" not in (s.last_llm_payload or "")
    assert "12/05/1990" not in (s.last_llm_payload or "")
    assert "11144477735" not in (s.last_llm_response or "")
    assert "12/05/1990" not in (s.last_llm_response or "")
    orch.client.assert_exhausted()


def test_integrated_flow_rejection_interview_reapproval(repo):
    """Bruno (score 350): pedido de R$ 7.000 reprovado -> entrevista -> score 570 -> reanálise aprovada."""
    bruno = "52998224725"
    orch = make(repo, [
        reply("Olá, Bruno. Como posso ajudar?"),
        reply(calls=[("solicitar_aumento_limite", {"novo_limite": 7000})]),
        reply("Como está sua renda mensal hoje?"),
        reply(calls=[("registrar_respostas_entrevista", {
            "renda_mensal": 6000, "tipo_emprego": "formal", "despesas_fixas": 2000,
            "num_dependentes": 1, "tem_dividas": False})]),
        reply("Entendi os cinco dados. Confirma esse resumo?"),
        reply(calls=[("registrar_entrevista_e_calcular_score", {
            "renda_mensal": 6000, "tipo_emprego": "formal", "despesas_fixas": 2000,
            "num_dependentes": 1, "tem_dividas": False})]),                           # volta ao crédito sozinho
        reply("Ótima notícia: seu limite agora é R$ 7.000,00!"),
    ])
    s = orch.new_session()
    orch.handle(s, f"{bruno} 30/11/1985")
    r2 = orch.handle(s, "quero aumentar meu limite para 7000")
    assert "entrevista" in r2 and s.active_agent == "credito"
    assert repo.get_client(bruno)["limite_credito"] == 2000
    assert "renda" in orch.handle(s, "sim").lower() and s.active_agent == "entrevista"
    r4 = orch.handle(s, "6000, formal, 2000 de despesas, 1 dependente, sem dívidas")
    assert "confirma" in r4.lower() and s.active_agent == "entrevista"
    r5 = orch.handle(s, "sim")
    assert "R$ 7.000" in r5 and s.active_agent == "credito"
    assert repo.get_client(bruno)["score"] == 570 and repo.get_client(bruno)["limite_credito"] == 7000
    assert [r["status_pedido"] for r in repo.list_requests()] == [STATUS_REPROVADO, STATUS_APROVADO]
    assert_history_valid(s.history)
    visible = " ".join(c for role, c in s.transcript if role == "assistant").lower()
    assert "agente" not in visible and "transfer" not in visible  # transições invisíveis
    prompts = [req[0]["content"] for req in orch.client.requests]
    assert "recepção e autenticação" in prompts[0] and "entrevista financeira" in prompts[3]
    orch.client.assert_exhausted()


def test_declining_credit_interview_returns_to_triage_menu(repo_ana_620):
    rejected = ("solicitar_aumento_limite", {"novo_limite": 9000})
    orch = make(repo_ana_620, [reply(calls=[rejected])])
    s = logged_in(orch, agent="credito")

    rejected_reply = orch.handle(s, "quero aumentar para 9000")
    assert "entrevista financeira" in rejected_reply
    assert s.flow.awaiting is Awaiting.INTERVIEW_OFFER

    decline_reply = orch.handle(s, "não")

    assert "consulta de limite, câmbio" in decline_reply
    assert s.active_agent == "triagem"
    assert s.flow.awaiting is Awaiting.NONE and not s.ended
    orch.client.assert_exhausted()


def test_ambiguous_decline_during_interview_offer_does_not_end_session(repo_ana_620):
    rejected = ("solicitar_aumento_limite", {"novo_limite": 9000})
    orch = make(repo_ana_620, [reply(calls=[rejected])])
    s = logged_in(orch, agent="credito")

    orch.handle(s, "quero aumentar para 9000")
    decline_reply = orch.handle(s, "não quero mais")

    assert "consulta de limite, câmbio" in decline_reply
    assert s.active_agent == "triagem"
    assert s.flow.awaiting is Awaiting.NONE and not s.ended
    orch.client.assert_exhausted()


@pytest.mark.parametrize("text", [
    "pode encerrar a entrevista", "quero finalizar a entrevista", "quero cancelar só a entrevista",
])
def test_interview_close_transfers_to_credit_without_ending(repo, text):
    """Cancelar apenas a entrevista devolve o atendimento ao Crédito
    sem fechar a sessão nem depender de uma chamada ao modelo."""
    orch = make(repo, [])  # nenhuma resposta LLM esperada
    s = logged_in(orch, agent="entrevista")

    reply_text = orch.handle(s, text)

    # Sem chamada ao LLM: o interceptor resolveu de forma determinística.
    assert len(orch.client.requests) == 0
    assert not s.ended
    assert s.active_agent == "credito"
    # Mensagem de entrevista desistida (interpelada de volta ao crédito).
    assert "consulta de limite, câmbio" in reply_text


def test_ambiguous_close_during_interview_asks_confirmation(repo):
    """Um pedido ambíguo de encerramento durante a entrevista exige esclarecimento."""
    orch = make(repo, [])
    s = logged_in(orch, agent="entrevista")

    reply_text = orch.handle(s, "encerrar")

    assert not s.ended
    assert s.active_agent == "entrevista"  # continua na entrevista até confirmar
    assert len(orch.client.requests) == 0  # determinístico, sem LLM
    assert "encerrar apenas a entrevista" in reply_text or "encerrar todo o atendimento" in reply_text


@pytest.mark.parametrize("text", [
    "encerrar o atendimento",
    "quero encerrar todo o atendimento",
    "pode finalizar todo o atendimento",
    "encerrar a conversa",
    "quero encerrar tudo",
])
def test_explicit_global_close_during_interview_still_ends_without_financial_effects(repo, text):
    """Pedido global encerra antes do pipeline e limpa propostas ainda pendentes."""
    orch = make(repo, [])
    s = logged_in(orch, agent="entrevista")
    s.flow.interview_started = True
    s.flow.interview_answers = {"tipo_emprego": "formal"}
    s.flow.pending_interview_answers = {"renda_mensal": 6000}
    before = {
        "clientes": repo.clients_path.read_bytes(),
        "score": repo.score_path.read_bytes(),
        "requests": repo.requests_path.read_bytes(),
    }

    reply_text = orch.handle(s, text)

    assert s.ended
    assert MSG_FAREWELL in reply_text
    assert len(orch.client.requests) == 0  # determinístico, sem LLM
    assert s.flow.interview_answers == {}
    assert s.flow.pending_interview_answers == {}
    assert not s.flow.interview_started
    assert [
        call["function"]["name"]
        for message in s.history
        for call in message.get("tool_calls", [])
    ] == ["encerrar_atendimento"]
    assert before == {
        "clientes": repo.clients_path.read_bytes(),
        "score": repo.score_path.read_bytes(),
        "requests": repo.requests_path.read_bytes(),
    }


def test_global_close_with_confirmed_interview_blocks_calculation_and_reanalysis(repo):
    """Mesmo com cinco dados confirmados, fechar a sessão não pode calcular score."""
    orch = make(repo, [])
    s = logged_in(orch, agent="entrevista")
    s.flow.interview_started = True
    s.flow.interview_answers = {
        "renda_mensal": 6000,
        "tipo_emprego": "formal",
        "despesas_fixas": 2000,
        "num_dependentes": 1,
        "tem_dividas": False,
    }
    before = {
        "clientes": repo.clients_path.read_bytes(),
        "score": repo.score_path.read_bytes(),
        "requests": repo.requests_path.read_bytes(),
    }

    orch.handle(s, "quero encerrar todo o atendimento")

    assert s.ended
    assert len(orch.client.requests) == 0
    assert s.automatic_reanalysis_result is None
    assert before == {
        "clientes": repo.clients_path.read_bytes(),
        "score": repo.score_path.read_bytes(),
        "requests": repo.requests_path.read_bytes(),
    }


def test_operation_cancellation_does_not_end_global_session(repo):
    """Cancelar uma operação permanece assunto do atendimento, não encerramento global."""
    orch = make(repo, [reply("Posso ajudar a esclarecer essa operação.")])
    s = logged_in(orch, agent="entrevista")

    orch.handle(s, "quero cancelar a operação")

    assert not s.ended
    assert len(orch.client.requests) == 1


# Confirmação local de encerramento da entrevista.
def test_confirmation_sim_ends_only_interview_and_returns_to_credit(repo):
    """'sim' confirma a primeira opção da pergunta: encerra só a entrevista."""
    orch = make(repo, [])
    s = logged_in(orch, agent="entrevista")
    orch.handle(s, "encerrar")
    assert s.flow.awaiting is Awaiting.CLOSE_CONFIRM

    reply_text = orch.handle(s, "sim")

    assert not s.ended and s.active_agent == "credito"
    assert s.flow.awaiting is Awaiting.NONE  # estado pendente limpo
    assert "consulta de limite, câmbio" in reply_text  # devolvido ao crédito
    assert len(orch.client.requests) == 0


def test_confirmation_no_keeps_interview_open(repo):
    """'não' recusa o encerramento; a entrevista segue e nada é fechado."""
    orch = make(repo, [])
    s = logged_in(orch, agent="entrevista")
    orch.handle(s, "encerrar")

    reply_text = orch.handle(s, "não")

    assert not s.ended and s.active_agent == "entrevista"
    assert s.flow.awaiting is Awaiting.NONE
    assert "não vou encerrar nada" in reply_text
    assert len(orch.client.requests) == 0


def test_confirmation_explicit_global_scope_ends_session(repo):
    "'encerrar tudo' nomeia o escopo global e encerra o atendimento completo."
    orch = make(repo, [])
    s = logged_in(orch, agent="entrevista")
    orch.handle(s, "encerrar")

    reply_text = orch.handle(s, "encerrar tudo")

    assert s.ended and MSG_FAREWELL in reply_text
    assert s.flow.awaiting is Awaiting.NONE
    assert len(orch.client.requests) == 0


def test_confirmation_clarification_repeats_question_without_ending(repo):
    """Pedido de esclarecimento repete a pergunta; após 3, cancela e mantém a entrevista."""
    orch = make(repo, [])
    s = logged_in(orch, agent="entrevista")
    orch.handle(s, "encerrar")

    first = orch.handle(s, "como assim?")
    assert not s.ended and s.active_agent == "entrevista"
    assert s.flow.awaiting is Awaiting.CLOSE_CONFIRM  # continua pendente
    assert "'sim'" in first  # repete as opções da confirmação

    orch.handle(s, "não entendi")
    final = orch.handle(s, "pode repetir?")  # 3ª tentativa de esclarecimento

    assert not s.ended and s.active_agent == "entrevista"
    assert s.flow.awaiting is Awaiting.NONE  # cancelada após 3 repetições
    assert "não vou encerrar nada" in final
    assert len(orch.client.requests) == 0


def test_confirmation_unrelated_answer_abandons_state_without_closing(repo):
    """Resposta fora do contexto abandona a confirmação e segue o fluxo (LLM), sem fechar."""
    orch = make(repo, [reply("Claro! Posso ajudar com seu limite ou cotações.")])
    s = logged_in(orch, agent="entrevista")
    s.flow.interview_answers = {
        "renda_mensal": 6000, "tipo_emprego": "formal", "despesas_fixas": 2000,
        "num_dependentes": 1, "tem_dividas": False,
    }
    orch.handle(s, "encerrar")

    reply_text = orch.handle(s, "quero aumentar meu limite")

    assert not s.ended
    assert s.flow.awaiting is Awaiting.NONE  # estado pendente descartado
    assert reply_text.startswith("Claro!")
    orch.client.assert_exhausted()
