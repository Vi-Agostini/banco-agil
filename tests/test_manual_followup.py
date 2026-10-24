"""Retomada de atendimento com CSVs, relógio e provedor isolados."""
import copy
import json
import logging

import pytest

from banco_agil.agents import AGENTS
from banco_agil.flow import FlowController
from banco_agil.interview import has_interview_request
from banco_agil.llm import _tool_schemas
from fakes import RateLimited
from orchestrator_fakes import logged_in
from structured_fakes import action, check_protocol, make, message
from support import snapshot_csv as csv


def original_request(orch):
    s = logged_in(orch, agent="credito")
    s.history.append({"role": "user", "content": "Quero aumentar meu limite para 31000"})
    collect = orch.tool_executor.execute_from_llm(AGENTS["credito"], s,
        "registrar_intencao_credito", '{"valor":31000,"modalidade":"limite_total"}')
    assert collect["ok"] and s.credit_proposal["confirmada"]
    result = orch.tool_executor.execute_from_llm(AGENTS["credito"], s,
        "solicitar_aumento_limite", '{"novo_limite":31000}')
    assert result["status_pedido"] == "reprovado"
    FlowController.start_interview(s)
    s.active_agent = "entrevista"
    s.flow.interview_answers = {"renda_mensal": 123000, "num_dependentes": 9}
    s.flow.pending_interview_answers = {"tipo_emprego": "autonomo"}
    assert has_interview_request(s)
    return s


def protected_state(session):
    return copy.deepcopy({
        "answers": session.flow.interview_answers,
        "pending": session.flow.pending_interview_answers,
        "proposal": session.credit_proposal,
        "desired": session.desired_limit,
        "source_id": session.last_request_id,
        "source_status": session.last_request_status,
        "authenticated": session.authenticated,
    })


@pytest.mark.parametrize("mode", ["native", "structured"])
def test_return_to_credit_keeps_original_request_and_does_not_ask_its_amount_again(repo_ana_620, mode):
    orch = make(repo_ana_620, [action("transferir_para_agente", destino="credito"),
        message("Qual valor você deseja solicitar como novo limite?")], mode)
    s = original_request(orch)
    before, state = csv(orch.repo), protected_state(s)
    response = orch.handle(s, "E meu aumento de limite?")
    assert s.active_agent == "credito" and has_interview_request(s)
    assert "31.000,00" in response and "entrevista" in response
    assert "Qual valor" not in response
    assert protected_state(s) == state and csv(orch.repo) == before
    assert len(orch.repo.list_requests()) == 1
    check_protocol(orch, s)


@pytest.mark.parametrize("mode", ["native", "structured"])
def test_repeating_original_interview_request_does_not_replace_proposal_or_write(repo_ana_620, mode):
    orch = make(repo_ana_620, [action("registrar_intencao_credito", valor=31000, modalidade="limite_total"),
        message("Seu pedido anterior está preservado. Quer retomar a entrevista?")], mode)
    s = original_request(orch)
    s.active_agent = "credito"
    before, state = csv(orch.repo), protected_state(s)
    response = orch.handle(s, "31 mil, eu já tinha falado")
    fact = s.tool_facts["registrar_intencao_credito"]["result"]
    assert fact["pedido_anterior_preservado"] is True
    assert "maior que zero" not in response
    assert protected_state(s) == state and csv(orch.repo) == before
    assert len(orch.repo.list_requests()) == 1 and not s.ended
    check_protocol(orch, s)


@pytest.mark.parametrize("mode", ["native", "structured"])
def test_different_amount_during_interview_does_not_open_competing_confirmation(repo_ana_620, mode):
    orch = make(repo_ana_620, [action("registrar_intencao_credito", valor=32000, modalidade="limite_total")], mode)
    s = original_request(orch)
    s.active_agent = "credito"
    before, state = csv(orch.repo), protected_state(s)
    response = orch.handle(s, "Quero mudar meu pedido para 32000")
    assert "31.000" in response and "retomar a entrevista" in response
    assert "32.000" not in response and "**confirmo**" not in response
    assert protected_state(s) == state and csv(orch.repo) == before
    assert s.flow.attempts == 0 and not s.ended
    check_protocol(orch, s)


@pytest.mark.parametrize("path", ["contract", "handler", "flow", "effects"])
def test_interview_origin_blocks_financial_operation_without_counting_customer_attempt(repo_ana_620, path):
    from banco_agil.tools import ToolContext, solicitar_aumento_limite

    orch = make(repo_ana_620, [])
    s = original_request(orch)
    s.active_agent = "credito"
    before, state = csv(orch.repo), protected_state(s)
    attempts = s.flow.attempts
    if path == "contract":
        names = {t["function"]["name"] for t in _tool_schemas(AGENTS["credito"], s)}
        assert "solicitar_aumento_limite" not in names
        result = orch.tool_executor.execute_from_llm(AGENTS["credito"], s,
            "solicitar_aumento_limite", '{"novo_limite":31000}')
        assert result["codigo"] == "entrevista_em_andamento"
    elif path == "handler":
        result = solicitar_aumento_limite(ToolContext(s, orch.repo, orch.fx, orch.credit), 31000)
        assert result["codigo"] == "entrevista_em_andamento"
    elif path == "flow":
        result = orch.flow.requested_input_error(s, "limit", {"codigo": "entrevista_em_andamento"})
        assert "31.000,00" in result and "entrevista" in result
    else:
        flow = copy.deepcopy(s.flow)
        orch.flow.apply_tool_effects(s, "solicitar_aumento_limite", {
            "erro": "Conclua a entrevista anterior.", "codigo": "entrevista_em_andamento"})
        assert s.flow == flow
    assert s.flow.attempts == attempts
    assert protected_state(s) == state and csv(orch.repo) == before
    assert len(orch.repo.list_requests()) == 1


@pytest.mark.parametrize("mode", ["native", "structured"])
def test_triage_loop_does_not_present_unperformed_authentication_as_invalid_credentials(repo_ana_620, mode):
    orch = make(repo_ana_620, [action("registrar_entrada_invalida"),
                             action("registrar_entrada_invalida")], mode)
    s = orch.new_session()
    before = csv(orch.repo)
    response = orch.handle(s, "mensagem de autenticação não interpretada")
    assert s.last_failure_category == "progress_stalled"
    assert s.auth_failures == 0 and not s.authenticated and not s.ended
    assert s.flow.unclear_count == 1
    assert "CPF" in response
    assert "validar" not in response and "inválid" not in response
    assert "data de nascimento" not in response
    assert csv(orch.repo) == before
    check_protocol(orch, s)


@pytest.mark.parametrize("mode", ["native", "structured"])
def test_long_retry_after_is_recorded_without_sleep_or_retry_and_preserves_interview(
        repo_ana_620, mode, monkeypatch, caplog):
    orch = make(repo_ana_620, [RateLimited("180")], mode,
                max_rate_limit_wait_seconds=60)
    s = original_request(orch)
    s.flow.interview_question_pending = False
    before, state = csv(orch.repo), protected_state(s)
    waits = []
    monkeypatch.setattr("banco_agil.llm.time.sleep", waits.append)
    with caplog.at_level(logging.INFO, logger="banco_agil"):
        response = orch.handle(s, "Podemos continuar com a entrevista?")
    events = [json.loads(r.message) for r in caplog.records if r.message.startswith("{")]
    failure = next(e for e in events if e.get("event") == "llm_call" and e.get("status_http") == 429)
    assert failure["retry_block_reason"] == "retry_after_exceeds_wait_limit"
    assert failure["result"] == "error" and failure["attempt"] == 1
    assert len(orch.client.requests) == 1 and not waits
    assert "HTTP" not in response and "JSON" not in response
    assert protected_state(s) == state and csv(orch.repo) == before
    assert s.active_agent == "entrevista" and s.authenticated and not s.ended
    assert s.flow.attempts == 0
    check_protocol(orch, s)
