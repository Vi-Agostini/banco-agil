"""Recuperação no mesmo modelo com estado e segurança financeira preservados.

O HTTP simulado verifica os contratos locais, sem avaliar o modelo remoto.
"""
import copy
import json

import pytest

from banco_agil.agents import AGENTS
from banco_agil.exchange import ExchangeService
from banco_agil.flow import FlowController
from banco_agil.llm import LlmRetryState, call_llm
from banco_agil.orchestrator import Orchestrator
from banco_agil.session import Session
from provider_fakes import MODEL, daily, error, local_http, preserved_state
from orchestrator_fakes import logged_in
from structured_fakes import action, message, strict_spec
from support import snapshot_csv as csv


TOOL_FAILURE = "Tool choice is none, but model called a tool"
APPROVED = "Seu pedido foi aprovado e o novo limite de R$ 8.000,00 já está ativo."


def generation_error(code):
    return error(400, code, TOOL_FAILURE if code == "tool_use_failed" else
                 "Failed to validate JSON. Please adjust your prompt.",
                 kind="invalid_request_error")


def interview_state():
    session = Session(active_agent="entrevista", authenticated=True, cpf="local-only",
        nome="Pessoa fictícia", history=[
            {"role": "assistant", "content": "Qual é seu tipo de emprego?"},
            {"role": "user", "content": "formal"},
        ])
    FlowController.start_interview(session)
    session.flow.interview_answers = {"renda_mensal": 6000}
    session.flow.pending_interview_answers = {"tipo_emprego": "formal"}
    session.credit_proposal = dict(valor=90000, modalidade="limite_total",
        novo_limite=90000, confirmada=False)
    session.desired_limit = 90000
    session.last_request_id, session.last_request_status = 7, "reprovado"
    return session


def assert_protocol(requests, mode):
    for sent in requests:
        if mode == "structured":
            strict_spec(sent)
            assert not any(item.get("tool_calls") or item.get("tool_call_id")
                           or item["role"] == "tool" for item in sent["messages"])
        else:
            assert "response_format" not in sent and "tools" in sent


@pytest.mark.parametrize("mode", ["native", "structured"])
@pytest.mark.parametrize("code", ["tool_use_failed", "json_validate_failed"])
def test_one_generation_recovery_preserves_state_and_protocol(local_http, mode, code):
    create, clock = local_http
    provider = create([generation_error(code), message("Você confirma emprego formal?")])
    session, budget = interview_state(), LlmRetryState()
    before = preserved_state(session)
    result = call_llm(provider.client, MODEL, AGENTS["entrevista"], session,
                      llm_mode=mode, retry_state=budget)
    assert result.content == "Você confirma emprego formal?"
    assert len(provider.requests) == 2 and not provider.pending and not clock.waits
    assert provider.client.max_retries == 0 and preserved_state(session) == before
    assert budget.generation_retry_used and not budget.rate_limit_retry_used
    original, repaired = provider.requests
    assert original["model"] == repaired["model"] == MODEL
    if mode == "structured":
        assert strict_spec(original) == strict_spec(repaired)
        assert ("tools" in repaired) is (code == "tool_use_failed")
    else:
        assert original["tools"] == repaired["tools"]
    assert repaired["messages"][1:] == original["messages"][1:]
    assert repaired["messages"][0] != original["messages"][0]
    assert not any(TOOL_FAILURE in item.get("content", "") for item in repaired["messages"])
    assert_protocol(provider.requests, mode)


@pytest.mark.parametrize("mode", ["native", "structured"])
@pytest.mark.parametrize("code", ["tool_use_failed", "json_validate_failed"])
def test_repeated_generation_error_stops_after_one_recovery(local_http, mode, code):
    create, clock = local_http
    provider = create([generation_error(code), generation_error(code)])
    session, budget = interview_state(), LlmRetryState()
    before = preserved_state(session)
    assert call_llm(provider.client, MODEL, AGENTS["entrevista"], session,
                    llm_mode=mode, retry_state=budget) is None
    assert len(provider.requests) == 2 and budget.generation_retry_used
    assert preserved_state(session) == before and not clock.waits and not provider.pending
    assert {r["model"] for r in provider.requests} == {MODEL}


@pytest.mark.parametrize("mode", ["native", "structured"])
def test_daily_error_after_generation_recovery_is_blocking(local_http, mode):
    create, clock = local_http
    provider = create([generation_error("json_validate_failed"), daily()])
    session, budget = interview_state(), LlmRetryState()
    before = preserved_state(session)
    assert call_llm(provider.client, MODEL, AGENTS["entrevista"], session,
                    llm_mode=mode, retry_state=budget) is None
    assert len(provider.requests) == 2 and budget.generation_retry_used
    assert not budget.rate_limit_retry_used and not clock.waits
    assert session.last_failure_cause == "daily_token_limit"
    assert preserved_state(session) == before


@pytest.mark.parametrize("failure", [
    error(400, "invalid_request_error", "Invalid JSON schema", param="response_format"),
    error(400, "json_validate_failed", "Invalid schema: local-only"),
    error(400, "tool_use_failed", param="tools"), error(500, "server_error"),
    TimeoutError("local-only timeout"), {"_raw": "not JSON"},
    {"_raw": json.dumps({"resposta": {"tipo": "mensagem", "mensagem": "Olá", "acao": {}}})},
    {"_raw": "{}", "_finish": "length"},
])
def test_blocking_errors_do_not_gain_generation_retries(local_http, failure):
    create, clock = local_http
    provider = create([failure])
    session, budget = interview_state(), LlmRetryState()
    before = preserved_state(session)
    assert call_llm(provider.client, MODEL, AGENTS["entrevista"], session,
                    llm_mode="structured", retry_state=budget) is None
    assert len(provider.requests) == 1 and not budget.generation_retry_used
    assert not budget.rate_limit_retry_used and not clock.waits
    assert preserved_state(session) == before


def test_later_generation_success_preserves_interview_after_invalid_output(local_http):
    create, clock = local_http
    provider = create([{"_raw": "not JSON"}, message("Podemos continuar a entrevista.")])
    session = interview_state()
    before = preserved_state(session)
    assert call_llm(provider.client, MODEL, AGENTS["entrevista"], session, llm_mode="structured") is None
    assert preserved_state(session) == before
    session.history.append({"role": "user", "content": "Podemos continuar?"})
    assert call_llm(provider.client, MODEL, AGENTS["entrevista"], session,
                    llm_mode="structured").content == "Podemos continuar a entrevista."
    assert len(provider.requests) == 2 and not clock.waits
    assert session.flow.pending_interview_answers == {"tipo_emprego": "formal"}


@pytest.mark.parametrize("mode", ["native", "structured"])
@pytest.mark.parametrize("failure", [TimeoutError("local-only timeout"), error(500, "server_error")])
def test_generation_recovery_does_not_open_generic_retry_budget(local_http, mode, failure):
    create, clock = local_http
    provider = create([generation_error("tool_use_failed"), failure])
    budget = LlmRetryState()
    assert call_llm(provider.client, MODEL, AGENTS["entrevista"], interview_state(),
                    llm_mode=mode, retry_state=budget, max_retries=1) is None
    assert len(provider.requests) == 2 and budget.generation_retry_used
    assert not budget.rate_limit_retry_used and not clock.waits


@pytest.mark.parametrize("mode", ["native", "structured"])
def test_generation_and_tpm_recovery_share_finite_independent_budgets(local_http, mode):
    create, clock = local_http
    provider = create([generation_error("tool_use_failed"),
        error(429, "rate_limit_exceeded", "Tokens per minute (TPM) exceeded", headers={"Retry-After": "2"}),
        message("Podemos continuar.")])
    budget = LlmRetryState()
    result = call_llm(provider.client, MODEL, AGENTS["entrevista"], interview_state(),
                      llm_mode=mode, retry_state=budget)
    assert result.content == "Podemos continuar." and len(provider.requests) == 3
    assert budget.generation_retry_used and budget.rate_limit_retry_used and clock.waits == [20]
    assert {r["model"] for r in provider.requests} == {MODEL}


@pytest.mark.parametrize("mode", ["native", "structured"])
def test_recovered_action_never_creates_financial_consent(local_http, repo_ana_620, mode):
    create, clock = local_http
    provider = create([generation_error("tool_use_failed"), action("solicitar_aumento_limite", novo_limite=8000)])
    orch = Orchestrator(repo_ana_620, ExchangeService(), provider.client, MODEL, llm_mode=mode)
    session = logged_in(orch, agent="credito")
    session.credit_proposal = dict(valor=8000, modalidade="limite_total", novo_limite=8000, confirmada=False)
    before, proposal = csv(orch.repo), copy.deepcopy(session.credit_proposal)
    orch.handle(session, "Primeiro quero conferir os dados da proposta.")
    assert csv(orch.repo) == before and not orch.repo.list_requests()
    assert session.credit_proposal == proposal and session.authenticated
    assert session.active_agent == "credito" and not session.ended
    assert len(provider.requests) == 2 and not clock.waits
    assert_protocol(provider.requests, mode)


@pytest.mark.parametrize("mode", ["native", "structured"])
@pytest.mark.parametrize("stage", ["before_operation", "after_operation"])
@pytest.mark.parametrize("repeat_operation", [False, True])
def test_generation_recovery_never_replays_financial_effects(local_http, repo_ana_620,
                                                             mode, stage, repeat_operation):
    create, clock = local_http
    operation = action("solicitar_aumento_limite", novo_limite=8000)
    closing = operation if repeat_operation else message(APPROVED)
    script = ([generation_error("tool_use_failed"), operation, closing]
              if stage == "before_operation" else [operation, generation_error("tool_use_failed"), closing])
    provider = create(script)
    orch = Orchestrator(repo_ana_620, ExchangeService(), provider.client, MODEL, llm_mode=mode)
    session = logged_in(orch, agent="credito")
    session.credit_proposal = dict(valor=8000, modalidade="limite_total", novo_limite=8000, confirmada=False)
    before = csv(orch.repo)
    answer = orch.handle(session, "confirmo")
    requests = orch.repo.list_requests()
    assert len(requests) == 1 and requests[0]["status_pedido"] == "aprovado"
    assert float(requests[0]["novo_limite_solicitado"]) == 8000
    assert orch.repo.get_client(session.cpf)["limite_credito"] == 8000
    assert session.credit_proposal["executada"] and session.authenticated and not session.ended
    assert csv(orch.repo)["score_limite.csv"] == before["score_limite.csv"]
    assert "8.000,00" in answer and "aprovado" in answer.casefold()
    assert len(provider.requests) == len(script) and not provider.pending and not clock.waits
    assert {r["model"] for r in provider.requests} == {MODEL}
    assert_protocol(provider.requests, mode)
