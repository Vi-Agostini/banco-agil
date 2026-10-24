"""Classificação de falhas e espera limitada no modelo configurado, sem rede."""
from types import SimpleNamespace as NS

import pytest

from banco_agil.agents import AGENTS
from banco_agil.llm import LlmRetryState, call_llm
from banco_agil.rate_limit import GovernedClient, RateLimiter, daily_token_limit
from banco_agil.session import Session
from provider_fakes import MODEL, local_http, preserved_state


def test_existing_cooldown_cannot_extend_turn_admission_deadline(local_http):
    create, clock = local_http
    provider = create([])
    bucket = RateLimiter(30, 30000, clock=clock, sleep=clock.sleep)
    bucket.defer(90)
    client = GovernedClient(provider.client, bucket)
    state = LlmRetryState(admission_deadline=clock.now + 30)
    session = Session(authenticated=True, active_agent="credito", cpf="fixture",
                      history=[{"role": "user", "content": "Podemos conversar?"}])
    before = preserved_state(session)
    assert call_llm(client, MODEL, AGENTS["credito"], session, llm_mode="structured", retry_state=state) is None
    assert session.last_failure_category == "local_capacity"
    assert not provider.requests and not clock.waits
    assert preserved_state(session) == before


@pytest.mark.parametrize("body", [None, {}, {"error": "TPD"},
    {"error": {"type": "tokens", "code": "rate_limit_exceeded", "message": None}},
    {"error": {"type": "tokens", "code": "rate_limit_exceeded", "message": "TPD"}},
    {"error": {"type": "tokens", "code": "rate_limit_exceeded", "message": "requests per day (RPD)"}}])
def test_daily_classification_fails_closed_for_incomplete_provider_metadata(body):
    assert not daily_token_limit(NS(status_code=429, body=body))
