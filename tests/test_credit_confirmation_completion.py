"""A confirmação de crédito leva a uma decisão real ou preserva a pendência.

Decisões, falhas e relógio são simulados. As operações usam apenas o repositório
temporário; os testes não avaliam a compreensão do modelo remoto.
"""
import copy
import json

import pytest

from banco_agil.agents import AGENTS
from banco_agil.flow import Awaiting, FlowController
from banco_agil.llm import _tool_schemas
from banco_agil.response_guard import grounded_reply
from fakes import failure
from orchestrator_fakes import logged_in
from structured_fakes import action, check_protocol, completion, make, message
from support import snapshot_csv as csv


PROMISE = (
    "Sua solicitação de aumento para R$ 8.000,00 foi confirmada e está sendo processada. "
    "Em breve você receberá a resposta sobre a aprovação."
)
APPROVED = "Seu pedido foi aprovado e o novo limite de R$ 8.000,00 já está ativo."


class RateLimited(Exception):
    status_code = 429
    headers = {"Retry-After": "3"}
    body = {"error": {"code": "rate_limit_exceeded", "type": "tokens", "param": ""}}


def staged(orch, *, amount=8000, confirmed=False):
    session = logged_in(orch, agent="credito")
    session.credit_proposal = {
        "valor": amount, "novo_limite": amount, "modalidade": "limite_total",
        "confirmada": confirmed,
    }
    return session


def assert_no_async_promise(text):
    normalized = text.casefold()
    for forbidden in ("em breve", "está sendo processada", "está sendo processado",
                      "receberá a resposta", "aguarde a aprovação"):
        assert forbidden not in normalized


def assert_preserved_pending(orch, session, before):
    assert csv(orch.repo) == before and not orch.repo.list_requests()
    assert session.credit_proposal["confirmada"]
    assert not session.credit_proposal.get("executada")
    assert session.credit_proposal["novo_limite"] == 8000
    assert session.authenticated and session.active_agent == "credito" and not session.ended


@pytest.mark.parametrize("mode", ["native", "structured"])
@pytest.mark.parametrize("amount,status", [(8000, "aprovado"), (20000, "reprovado")])
@pytest.mark.parametrize("already_confirmed", [False, True])
def test_acknowledgement_is_repaired_by_ai_then_real_score_decision_is_reported(repo_ana_620, mode, amount, status, already_confirmed):
    promise = PROMISE.replace("8.000,00", "20.000,00") if amount == 20000 else PROMISE
    script = [message(promise), action("solicitar_aumento_limite", novo_limite=amount)]
    if status == "aprovado":
        script.append(message(APPROVED))
    orch = make(repo_ana_620, script, mode)
    session = staged(orch, amount=amount, confirmed=already_confirmed)
    before = csv(orch.repo)

    answer = orch.handle(session, "confirmo")

    requests = orch.repo.list_requests()
    assert len(requests) == 1 and requests[0]["status_pedido"] == status
    assert float(requests[0]["novo_limite_solicitado"]) == amount
    assert session.credit_proposal["executada"]
    assert session.credit_confirmation_turn == session.turn
    assert orch.repo.get_client(session.cpf)["limite_credito"] == (8000 if status == "aprovado" else 5000)
    assert csv(orch.repo)["score_limite.csv"] == before["score_limite.csv"]
    assert_no_async_promise(answer)
    if status == "aprovado":
        assert "aprovado" in answer and "8.000,00" in answer
        repair_state = " ".join(m["content"] for m in orch.client.requests[1]["messages"] if m["role"] == "system")
        closing_state = " ".join(m["content"] for m in orch.client.requests[2]["messages"] if m["role"] == "system")
        assert "A decisão anterior não concluiu o pedido confirmado" in repair_state
        assert "A decisão anterior não concluiu o pedido confirmado" not in closing_state
    else:
        assert "não foi aprovado" in answer and "entrevista" in answer.casefold()
        assert session.flow.awaiting is Awaiting.INTERVIEW_OFFER
    # O pedido deve partir de uma ação posterior do modelo, nunca do tratamento
    # do texto executando por conta própria uma operação financeira.
    assert len(orch.client.requests) == len(script)
    assert not any(m.get("role") == "assistant" and m.get("content") == promise
                   for m in session.history)
    check_protocol(orch, session)


@pytest.mark.parametrize("mode", ["native", "structured"])
@pytest.mark.parametrize("acknowledgement", [PROMISE, "Obrigado pela confirmação.", ""])
def test_only_one_ai_completion_repair_then_honest_pending_without_write(repo_ana_620, mode, acknowledgement):
    # Conteúdo estruturado vazio é inválido e bloqueante; no modo nativo,
    # texto vazio pode indicar uma decisão pendente que admite recuperação.
    script = ([message(acknowledgement)] if acknowledgement == "" and mode == "structured"
              else [message(acknowledgement), message(PROMISE)])
    orch = make(repo_ana_620, script, mode)
    session = staged(orch)
    before = csv(orch.repo)

    answer = orch.handle(session, "confirmo")

    assert_preserved_pending(orch, session, before)
    assert "ainda não foi concluída" in answer.casefold()
    assert_no_async_promise(answer)
    assert len(orch.client.requests) == len(script)
    check_protocol(orch, session)


@pytest.mark.parametrize("mode", ["native", "structured"])
@pytest.mark.parametrize("bad_action", ["transfer", "recollect"])
def test_confirmation_turn_blocks_handoff_and_recollection_before_ai_repair(repo_ana_620, mode, bad_action):
    bad = (action("transferir_para_agente", destino="triagem") if bad_action == "transfer"
           else action("registrar_intencao_credito", valor=8000, modalidade="limite_total"))
    orch = make(repo_ana_620, [bad, action("solicitar_aumento_limite", novo_limite=8000), message(APPROVED)], mode)
    session = staged(orch)

    answer = orch.handle(session, "confirmo")

    assert "aprovado" in answer and len(orch.repo.list_requests()) == 1
    assert session.active_agent == "credito" and session.credit_proposal["executada"]
    assert "transferir_para_agente" not in session.tool_facts
    assert "registrar_intencao_credito" not in session.tool_facts
    for request in orch.client.requests[:2]:
        offered = json.dumps(request.get("response_format", request.get("tools")), ensure_ascii=False)
        assert '"solicitar_aumento_limite"' in offered
        assert '"transferir_para_agente"' not in offered
        assert '"registrar_intencao_credito"' not in offered
    check_protocol(orch, session)


@pytest.mark.parametrize("mode", ["native", "structured"])
@pytest.mark.parametrize("bad", ["wrong_amount", "truncated"])
def test_invalid_financial_arguments_and_truncation_remain_blocking_without_repair(repo_ana_620, mode, bad):
    invalid = (action("solicitar_aumento_limite", novo_limite=-1) if bad == "wrong_amount" else
               completion(message(PROMISE), mode=mode, finish="length"))
    orch = make(repo_ana_620, [invalid], mode)
    session = staged(orch)
    before = csv(orch.repo)

    answer = orch.handle(session, "confirmo")

    assert len(orch.client.requests) == 1
    assert_preserved_pending(orch, session, before)
    assert_no_async_promise(answer)
    check_protocol(orch, session)


def test_malformed_structured_json_remains_blocking_after_credit_confirmation(repo_ana_620):
    orch = make(repo_ana_620, [completion("not json")])
    session = staged(orch)
    before = csv(orch.repo)

    answer = orch.handle(session, "confirmo")

    assert len(orch.client.requests) == 1
    assert session.last_failure_category == "invalid_structured_response"
    assert_preserved_pending(orch, session, before)
    assert_no_async_promise(answer)
    check_protocol(orch, session)


@pytest.mark.parametrize("mode", ["native", "structured"])
@pytest.mark.parametrize("error", ["timeout", "400", "429"])
def test_provider_failure_during_repair_preserves_consent_and_never_invents_processing(repo_ana_620, mode, error, monkeypatch):
    waits = []
    monkeypatch.setattr("banco_agil.llm.time.sleep", waits.append)
    errors = ([TimeoutError()] if error == "timeout" else
              [failure("json_validate_failed"), failure("json_validate_failed")] if error == "400" else
              [RateLimited(), RateLimited()])
    orch = make(repo_ana_620, [message(PROMISE), *errors], mode)
    session = staged(orch)
    before = csv(orch.repo)

    answer = orch.handle(session, "confirmo")

    assert_preserved_pending(orch, session, before)
    assert "ainda não foi concluída" in answer.casefold()
    assert_no_async_promise(answer)
    assert waits == ([20] if error == "429" else [])
    assert len(orch.client.requests) == 1 + len(errors)
    check_protocol(orch, session)


@pytest.mark.parametrize("mode", ["native", "structured"])
@pytest.mark.parametrize("kind", ["429", "400"])
def test_completion_repair_shares_existing_provider_recovery_budget(repo_ana_620, mode, kind, monkeypatch):
    waits = []
    monkeypatch.setattr("banco_agil.llm.time.sleep", waits.append)
    first = RateLimited() if kind == "429" else failure("json_validate_failed")
    last = RateLimited() if kind == "429" else failure("json_validate_failed")
    orch = make(repo_ana_620, [first, message(PROMISE), last], mode)
    session = staged(orch)
    before = csv(orch.repo)

    answer = orch.handle(session, "confirmo")

    assert len(orch.client.requests) == 3
    assert waits == ([20] if kind == "429" else [])
    assert_preserved_pending(orch, session, before)
    assert_no_async_promise(answer)
    check_protocol(orch, session)


@pytest.mark.parametrize("mode", ["native", "structured"])
def test_repair_can_recover_429_then_execute_once_without_second_cooldown(repo_ana_620, mode, monkeypatch):
    waits = []
    monkeypatch.setattr("banco_agil.llm.time.sleep", waits.append)
    orch = make(repo_ana_620, [message(PROMISE), RateLimited(),
        action("solicitar_aumento_limite", novo_limite=8000), message(APPROVED)], mode)
    session = staged(orch)

    answer = orch.handle(session, "confirmo")

    assert waits == [20] and len(orch.repo.list_requests()) == 1
    assert session.credit_proposal["executada"] and "aprovado" in answer
    assert_no_async_promise(answer)
    check_protocol(orch, session)


@pytest.mark.parametrize("mode", ["native", "structured"])
@pytest.mark.parametrize("closing", [PROMISE, "timeout", "repeat_operation"])
def test_operation_result_takes_priority_over_promise_or_failed_closing_without_reexecution(repo_ana_620, mode, closing):
    after = (TimeoutError() if closing == "timeout" else
             action("solicitar_aumento_limite", novo_limite=8000) if closing == "repeat_operation" else
             message(closing))
    orch = make(repo_ana_620, [message(PROMISE), action("solicitar_aumento_limite", novo_limite=8000), after], mode)
    session = staged(orch)

    answer = orch.handle(session, "confirmo")

    assert len(orch.repo.list_requests()) == 1
    assert orch.repo.get_client(session.cpf)["limite_credito"] == 8000
    assert session.credit_proposal["executada"] and "aprovado" in answer
    assert "8.000,00" in answer and "ainda não foi concluída" not in answer.casefold()
    assert_no_async_promise(answer)
    assert not session.ended and session.authenticated
    check_protocol(orch, session)


@pytest.mark.parametrize("mode", ["native", "structured"])
def test_timeout_before_operation_resumes_next_turn_with_preserved_consent_once(repo_ana_620, mode):
    orch = make(repo_ana_620, [message(PROMISE), TimeoutError()], mode)
    session = staged(orch)
    before = csv(orch.repo)
    first = orch.handle(session, "confirmo")
    assert_preserved_pending(orch, session, before)
    assert_no_async_promise(first)
    previous_confirmation_turn = session.credit_confirmation_turn
    orch.client.script.extend([action("solicitar_aumento_limite", novo_limite=8000), message(APPROVED)])

    answer = orch.handle(session, "continue o pedido autorizado")

    assert session.credit_confirmation_turn == previous_confirmation_turn < session.turn
    assert len(orch.repo.list_requests()) == 1 and "aprovado" in answer
    check_protocol(orch, session)


@pytest.mark.parametrize("mode", ["native", "structured"])
def test_repeated_explicit_confirmation_after_failure_renews_only_turn_marker(repo_ana_620, mode):
    orch = make(repo_ana_620, [TimeoutError(), message(PROMISE),
        action("solicitar_aumento_limite", novo_limite=8000), message(APPROVED)], mode)
    session = staged(orch)
    orch.handle(session, "confirmo")
    assert not orch.repo.list_requests() and session.credit_proposal["confirmada"]
    previous_turn = session.credit_confirmation_turn

    answer = orch.handle(session, "confirmo")

    assert session.credit_confirmation_turn == session.turn > previous_turn
    assert len(orch.repo.list_requests()) == 1 and "aprovado" in answer
    check_protocol(orch, session)


@pytest.mark.parametrize("mode", ["native", "structured"])
def test_old_credit_consent_does_not_force_an_operation_when_customer_changes_subject(repo_ana_620, mode):
    orch = make(repo_ana_620, [action("transferir_para_agente", destino="cambio"),
        message("Qual moeda você gostaria de consultar?")], mode)
    session = staged(orch, confirmed=True)
    session.credit_confirmation_turn = session.turn
    proposal = copy.deepcopy(session.credit_proposal)
    before = csv(orch.repo)

    answer = orch.handle(session, "Agora quero falar de câmbio")

    assert session.active_agent == "cambio" and "moeda" in answer
    assert session.credit_proposal == proposal and csv(orch.repo) == before
    assert not orch.repo.list_requests() and session.credit_confirmation_turn < session.turn
    check_protocol(orch, session)


@pytest.mark.parametrize("mode", ["native", "structured"])
@pytest.mark.parametrize("text", ["isso", "confirmo, podemos continuar"])
def test_insufficient_credit_consent_never_becomes_a_completion_obligation(repo_ana_620, mode, text):
    orch = make(repo_ana_620, [message(PROMISE)], mode)
    session = staged(orch)
    before = csv(orch.repo)

    answer = orch.handle(session, text)

    assert not session.credit_proposal["confirmada"] and not orch.repo.list_requests()
    assert session.credit_confirmation_turn != session.turn
    assert csv(orch.repo) == before and len(orch.client.requests) == 1
    assert_no_async_promise(answer)
    check_protocol(orch, session)


@pytest.mark.parametrize("mode", ["native", "structured"])
def test_interview_data_confirmation_does_not_authorize_staged_credit(repo_ana_620, mode):
    orch = make(repo_ana_620, [message("Qual é seu tipo de emprego?")], mode)
    session = staged(orch)
    session.active_agent = "entrevista"
    FlowController.start_interview(session)
    session.flow.pending_interview_answers = {"renda_mensal": 20000}
    before = csv(orch.repo)

    answer = orch.handle(session, "confirmo")

    assert session.flow.interview_answers == {"renda_mensal": 20000}
    assert not session.flow.pending_interview_answers
    assert not session.credit_proposal["confirmada"] and not orch.repo.list_requests()
    assert session.credit_confirmation_turn != session.turn
    assert csv(orch.repo) == before and "emprego" in answer
    check_protocol(orch, session)


@pytest.mark.parametrize("mode", ["native", "structured"])
def test_uncertain_operation_receipt_is_preserved_without_new_completion_loop(repo_ana_620, mode):
    orch = make(repo_ana_620, [message(PROMISE)], mode)
    session = staged(orch, confirmed=True)
    session.credit_operation = {"amount": 8000, "receipt": {"persisted_request_id": 0}}
    receipt = copy.deepcopy(session.credit_operation)
    before = csv(orch.repo)

    answer = orch.handle(session, "confirmo")

    assert session.credit_operation == receipt
    assert csv(orch.repo) == before and not orch.repo.list_requests()
    assert len(orch.client.requests) == 1
    assert_no_async_promise(answer)
    assert session.authenticated and not session.ended
    check_protocol(orch, session)


@pytest.mark.parametrize("promise", [PROMISE,
    "Seu pedido está em análise. Em breve você receberá o resultado.",
    "Sua solicitação está sendo processada; aguarde a aprovação."])
def test_future_processing_claims_require_actual_financial_results(repo_ana_620, promise):
    orch = make(repo_ana_620, [])
    session = staged(orch)
    before = csv(orch.repo)

    answer = grounded_reply(session, promise)

    assert answer != promise
    assert_no_async_promise(answer)
    assert csv(orch.repo) == before and not orch.repo.list_requests()


@pytest.mark.parametrize("text", [
    "Não estamos processando seu pedido em segundo plano.",
    "Seu pedido não está sendo processado neste momento.",
    "Se você confirmar, a solicitação poderá ser analisada.",
    "Caso autorize o pedido, você receberá o resultado da análise neste atendimento.",
    "Você gostaria de confirmar o pedido antes da análise?",
])
def test_background_claim_guard_preserves_negations_conditions_and_questions(repo_ana_620, text):
    orch = make(repo_ana_620, [])
    session = staged(orch)
    before = csv(orch.repo)

    assert grounded_reply(session, text) == text

    assert csv(orch.repo) == before and not orch.repo.list_requests()
    assert not session.credit_proposal["confirmada"]


def test_background_claim_guard_allows_a_completed_request_with_actual_result(repo_ana_620):
    orch = make(repo_ana_620, [])
    session = staged(orch, confirmed=True)
    result = orch.tool_executor.execute_from_llm(AGENTS["credito"], session,
        "solicitar_aumento_limite", '{"novo_limite":8000}')
    assert result["status_pedido"] == "aprovado" and session.credit_proposal["executada"]
    before = csv(orch.repo)
    text = "Seu pedido já foi processado. O limite atual é R$ 8.000,00."

    assert grounded_reply(session, text) == text

    assert csv(orch.repo) == before and len(orch.repo.list_requests()) == 1


@pytest.mark.parametrize("mode", ["native", "structured"])
def test_confirmation_catalog_restrictions_do_not_affect_later_turns(repo_ana_620, mode):
    orch = make(repo_ana_620, [TimeoutError()], mode)
    session = staged(orch)
    orch.handle(session, "confirmo")
    current = {tool["function"]["name"] for tool in _tool_schemas(AGENTS["credito"], session)}
    assert "solicitar_aumento_limite" in current
    assert "transferir_para_agente" not in current and "registrar_intencao_credito" not in current

    session.turn += 1

    later = {tool["function"]["name"] for tool in _tool_schemas(AGENTS["credito"], session)}
    assert {"transferir_para_agente", "registrar_intencao_credito"} <= later
    assert not orch.repo.list_requests()
    check_protocol(orch, session)
