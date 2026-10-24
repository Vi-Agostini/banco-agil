"""Um único modelo configurado durante falhas, retomada e operações financeiras."""
from concurrent.futures import ThreadPoolExecutor
import json
import logging
from threading import Barrier

import pytest

from banco_agil.agents import AGENTS
from banco_agil.exchange import ExchangeService
from banco_agil.flow import FlowController
from banco_agil.llm import LlmRetryState, call_llm
from banco_agil.orchestrator import Orchestrator
from banco_agil.session import Session
from provider_fakes import MODEL, OTHER_MODEL, daily, error, local_http, preserved_state
from orchestrator_fakes import logged_in
from structured_fakes import action, message
from support import snapshot_csv as csv


@pytest.mark.parametrize("mode", ["native", "structured"])
@pytest.mark.parametrize("model", [MODEL, OTHER_MODEL])
@pytest.mark.parametrize("failure", [daily(), daily(retry_after="2", wait="1s"),
    error(400, "invalid_request_error", param="response_format"),
    TimeoutError("local-only timeout"), ConnectionError("local-only connection")])
def test_blocking_provider_error_never_selects_another_model(local_http, mode, model, failure):
    create, clock = local_http
    provider = create([failure])
    session = Session(authenticated=True, active_agent="entrevista", cpf="fixture",
                      history=[{"role": "user", "content": "Podemos continuar?"}])
    FlowController.start_interview(session)
    session.flow.interview_answers = {"renda_mensal": 6000}
    session.flow.pending_interview_answers = {"tipo_emprego": "formal"}
    before, budget = preserved_state(session), LlmRetryState()
    assert call_llm(provider.client, model, AGENTS["entrevista"], session,
                    llm_mode=mode, retry_state=budget) is None
    assert [r["model"] for r in provider.requests] == [model]
    assert preserved_state(session) == before and not session.ended
    assert not budget.generation_retry_used and not budget.rate_limit_retry_used
    assert not clock.waits and not provider.pending and provider.client.max_retries == 0
    if isinstance(failure, tuple) and failure[0] == 429:
        assert session.last_failure_cause == "daily_token_limit"


@pytest.mark.parametrize("mode", ["native", "structured"])
@pytest.mark.parametrize("model", [MODEL, OTHER_MODEL])
@pytest.mark.parametrize("code", ["tool_use_failed", "json_validate_failed"])
def test_single_generation_recovery_keeps_the_configured_model(local_http, mode, model, code):
    create, clock = local_http
    provider = create([error(400, code, kind="invalid_request_error"), message("Podemos continuar.")])
    session, budget = Session(), LlmRetryState()
    assert call_llm(provider.client, model, AGENTS["triagem"], session,
                    llm_mode=mode, retry_state=budget).content == "Podemos continuar."
    assert [r["model"] for r in provider.requests] == [model, model]
    assert budget.generation_retry_used and not budget.rate_limit_retry_used
    assert not provider.pending and not clock.waits


@pytest.mark.parametrize("mode", ["native", "structured"])
@pytest.mark.parametrize("model", [MODEL, OTHER_MODEL])
def test_tpm_cooldown_keeps_one_configured_model(local_http, mode, model):
    create, clock = local_http
    provider = create([error(429, "rate_limit_exceeded", "Tokens per minute (TPM) exceeded",
                            headers={"Retry-After": "10.5"}), message("Podemos continuar.")])
    budget = LlmRetryState()
    result = call_llm(provider.client, model, AGENTS["triagem"], Session(),
                      llm_mode=mode, retry_state=budget)
    assert result.content == "Podemos continuar."
    assert [r["model"] for r in provider.requests] == [model, model]
    assert clock.waits == [20] and budget.rate_limit_retry_used
    assert not budget.generation_retry_used and not provider.pending


@pytest.mark.parametrize("mode", ["native", "structured"])
def test_daily_failure_does_not_redirect_later_turns_agents_or_sessions(local_http, mode):
    create, clock = local_http
    provider = create([daily(), message("Podemos continuar a entrevista."),
                       message("Podemos falar sobre crédito."), message("Olá, outra conversa.")])
    session = Session(authenticated=True, cpf="fixture", active_agent="entrevista")
    FlowController.start_interview(session)
    session.flow.interview_answers = {"renda_mensal": 6000}
    assert call_llm(provider.client, MODEL, AGENTS["entrevista"], session, llm_mode=mode) is None
    assert call_llm(provider.client, MODEL, AGENTS["entrevista"], session,
                    llm_mode=mode).content == "Podemos continuar a entrevista."
    session.active_agent = "credito"
    assert call_llm(provider.client, MODEL, AGENTS["credito"], session,
                    llm_mode=mode).content == "Podemos falar sobre crédito."
    assert call_llm(provider.client, MODEL, AGENTS["triagem"], Session(),
                    llm_mode=mode).content == "Olá, outra conversa."
    assert all(r["model"] == MODEL for r in provider.requests)
    assert session.authenticated and session.flow.interview_answers == {"renda_mensal": 6000}
    assert not provider.pending and not clock.waits


@pytest.mark.parametrize("mode", ["native", "structured"])
def test_legacy_model_selection_fields_cannot_override_configuration(local_http, mode):
    create, clock = local_http
    provider = create([message("Podemos continuar.")])
    session = Session(authenticated=True, cpf="fixture", active_agent="credito")
    # Uma sessão em cache pode conservar campos da versão anterior.
    session.active_model = OTHER_MODEL
    session.model_notice = "Aviso antigo"
    session.model_switch_announced = True
    result = call_llm(provider.client, MODEL, AGENTS["credito"], session, llm_mode=mode)
    assert result.content == "Podemos continuar."
    assert [r["model"] for r in provider.requests] == [MODEL]
    assert session.authenticated and session.active_agent == "credito" and not clock.waits


@pytest.mark.parametrize("mode", ["native", "structured"])
def test_concurrent_sessions_keep_model_and_conversations_isolated(local_http, mode):
    create, clock = local_http
    provider = create([daily()] * 3)
    sessions = [Session(history=[{"role": "user", "content": label}])
                for label in ("Conversa A", "Conversa B", "Conversa C")]
    originals, barrier = [preserved_state(s) for s in sessions], Barrier(3)

    def run(session):
        barrier.wait(timeout=5)
        return call_llm(provider.client, MODEL, AGENTS["triagem"], session, llm_mode=mode)

    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs = [pool.submit(run, s) for s in sessions]
        assert all(job.result(timeout=5) is None for job in jobs)
    assert [r["model"] for r in provider.requests] == [MODEL] * 3
    labels = {m["content"] for request in provider.requests for m in request["messages"]
              if m["role"] == "user"}
    assert labels == {"Conversa A", "Conversa B", "Conversa C"}
    for request in provider.requests:
        assert len([m for m in request["messages"] if m["role"] == "user"]) == 1
    assert [preserved_state(s) for s in sessions] == originals
    assert not clock.waits and not provider.pending


@pytest.mark.parametrize("mode", ["native", "structured"])
@pytest.mark.parametrize("stage", ["before", "after"])
def test_daily_failure_preserves_financial_result_and_safe_later_resume(local_http, repo_ana_620, mode, stage):
    create, clock = local_http
    operation = action("solicitar_aumento_limite", novo_limite=8000)
    script = ([daily(), operation, message("Seu novo limite aprovado é R$ 8.000,00.")]
              if stage == "before" else [operation, daily(), operation])
    provider = create(script)
    orch = Orchestrator(repo_ana_620, ExchangeService(), provider.client, MODEL, llm_mode=mode)
    session = logged_in(orch, agent="credito")
    session.credit_proposal = dict(valor=8000, modalidade="limite_total", novo_limite=8000, confirmada=False)
    before = csv(orch.repo)
    first = orch.handle(session, "confirmo")
    assert len(orch.repo.list_requests()) == (0 if stage == "before" else 1)
    if stage == "before":
        assert csv(orch.repo) == before
        assert "ainda não foi concluída" in first.casefold() and "preservada" in first.casefold()
    else:
        assert "8.000,00" in first
    orch.handle(session, "confirmo")
    assert len(orch.repo.list_requests()) == 1
    assert orch.repo.get_client(session.cpf)["limite_credito"] == 8000
    assert session.credit_proposal["executada"] and session.authenticated and not session.ended
    assert csv(orch.repo)["score_limite.csv"] == before["score_limite.csv"]
    assert all(r["model"] == MODEL for r in provider.requests)
    assert not provider.pending and not clock.waits


def test_daily_diagnostic_omits_provider_private_text_and_model_switch_event(local_http, caplog):
    create, _ = local_http
    private = "private-conversation-76543210"
    provider = create([daily(message=f"Rate limit on tokens per day (TPD): {private}")])
    with caplog.at_level(logging.INFO, logger="banco_agil"):
        assert call_llm(provider.client, MODEL, AGENTS["triagem"], Session(), llm_mode="structured") is None
    text = "\n".join(record.getMessage() for record in caplog.records)
    assert private not in text and "llm_model_switch" not in text
    events = [json.loads(r.getMessage()) for r in caplog.records if r.getMessage().startswith("{")]
    calls = [e for e in events if e.get("event") == "llm_call"]
    assert len(calls) == 1 and calls[0]["model"] == MODEL
    assert calls[0]["failure_cause"] == "daily_token_limit" and calls[0]["result"] == "error"
