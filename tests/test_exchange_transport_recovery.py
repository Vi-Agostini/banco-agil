"""Recuperação do transporte estruturado no câmbio, usando o mesmo modelo.

HTTP em memória e CSVs temporários conferem o fluxo da aplicação,
sem comprovar a qualidade de uma geração remota.
"""
from contextlib import ExitStack
import json
from types import SimpleNamespace as NS

import httpx2 as httpx
from openai import OpenAI
import pytest

from banco_agil.exchange import ExchangeService
from banco_agil.orchestrator import Orchestrator
from fakes import assert_history_valid, Clock
from provider_fakes import MODEL, error
from orchestrator_fakes import logged_in
from structured_fakes import action, message
from support import snapshot_csv as csv


QUESTION = "Quanto tá o dolar americado agora"
QUOTE_REPLY = (
    "A última cotação disponível de USD é R$ 5,1234. Fonte: AwesomeAPI. "
    "Atualização da fonte: 23/10/2026 às 16:12:00 (horário de São Paulo)."
    "\n\nPrecisa de mais alguma coisa?"
)
TOOL_FAILURE = error(400, "tool_use_failed", "Tool choice is none, but model called a tool",
                     kind="invalid_request_error")


@pytest.fixture
def transport_http(monkeypatch):
    """Serializa requisições pelo SDK e devolve respostas sem abrir conexões."""
    clock = Clock()
    monkeypatch.setattr("banco_agil.llm.time.sleep", clock.sleep)
    with ExitStack() as stack:
        def create(script):
            pending, requests = list(script), []

            def sink(request):
                sent = json.loads(request.content)
                requests.append(sent)
                assert pending, "Unexpected HTTP attempt after the shared recovery budget"
                value = pending.pop(0)
                if isinstance(value, tuple):
                    status, body, headers = value
                    return httpx.Response(status, json=body, headers=headers)
                calls, content, finish = None, None, "stop"
                if "response_format" in sent:
                    content = json.dumps({"resposta": value}, ensure_ascii=False)
                else:
                    # A função só transporta o schema; não executa a ação financeira.
                    assert len(sent["tools"]) == 1
                    assert sent["tools"][0]["function"]["name"] == "banco_agil_resposta"
                    calls = [{"id": f"local-carrier-{len(requests)}", "type": "function",
                              "function": {"name": "banco_agil_resposta",
                                           "arguments": json.dumps({"resposta": value}, ensure_ascii=False)}}]
                    finish = "tool_calls"
                body = {"id": "local-exchange", "object": "chat.completion", "created": 0,
                        "model": sent["model"], "usage": {"prompt_tokens": 1500,
                        "completion_tokens": 100, "total_tokens": 1600}, "choices": [{
                            "index": 0, "finish_reason": finish,
                            "message": {"role": "assistant", "content": content}}]}
                if calls is not None:
                    body["choices"][0]["message"]["tool_calls"] = calls
                return httpx.Response(200, json=body)

            client = stack.enter_context(OpenAI(api_key="local-only", base_url="http://memory.invalid/v1",
                max_retries=0, http_client=httpx.Client(transport=httpx.MockTransport(sink), trust_env=False)))
            return NS(client=client, requests=requests, pending=pending)

        yield create, clock


@pytest.fixture
def quote_http():
    requests = []

    def sink(request):
        requests.append(request)
        assert request.method == "GET"
        assert request.url == "https://economia.awesomeapi.com.br/json/last/USD-BRL"
        return httpx.Response(200, json={"USDBRL": {
            "bid": "5.1234", "create_date": "2026-10-23 16:12:00"}})

    with httpx.Client(transport=httpx.MockTransport(sink), trust_env=False) as client:
        yield ExchangeService(http_get=client.get), requests


def carrier_schema(sent):
    assert "response_format" not in sent
    assert sent["parallel_tool_calls"] is False
    assert sent["tool_choice"] == {"type": "function", "function": {"name": "banco_agil_resposta"}}
    assert len(sent["tools"]) == 1
    function = sent["tools"][0]["function"]
    assert function["strict"] is True and function["name"] == "banco_agil_resposta"
    return function["parameters"]


def action_names(schema):
    variants = schema["properties"]["resposta"]["anyOf"]
    return {option["properties"]["nome"]["enum"][0]
            for variant in variants if variant["properties"]["tipo"]["enum"] == ["acao"]
            for option in variant["properties"]["acao"]["anyOf"]}


def test_recovers_transport_and_completes_exchange_with_fresh_agent_catalog(
        transport_http, quote_http, repo_ana_620):
    create, clock = transport_http
    provider = create([TOOL_FAILURE,
        action("transferir_para_agente", destino="cambio"),
        action("consultar_cotacao", moeda="USD"), message(QUOTE_REPLY)])
    fx, quote_requests = quote_http
    orch = Orchestrator(repo_ana_620, fx, provider.client, MODEL, llm_mode="structured")
    session = logged_in(orch)
    correlation_id, before = session.correlation_id, csv(orch.repo)

    answer = orch.handle(session, QUESTION)

    assert answer == QUOTE_REPLY
    assert session.authenticated and session.cpf == "11144477735" and session.auth_failures == 0
    assert session.active_agent == "cambio" and not session.ended
    assert session.correlation_id == correlation_id
    assert session.structured_tool_models == {MODEL}
    assert not session.processing_guard.lock.locked()
    assert csv(orch.repo) == before and not orch.repo.list_requests()
    assert len(provider.requests) == 4 and len(quote_requests) == 1
    assert not provider.pending and not clock.waits
    assert provider.client.max_retries == 0
    original, repaired, quote, closing = provider.requests
    assert original["response_format"]["json_schema"]["strict"] is True
    assert carrier_schema(repaired) == original["response_format"]["json_schema"]["schema"]
    assert "consultar_cotacao" not in action_names(carrier_schema(repaired))
    assert "consultar_cotacao" in action_names(carrier_schema(quote))
    assert "solicitar_aumento_limite" not in action_names(carrier_schema(quote))
    assert "consultar_cotacao" in action_names(carrier_schema(closing))
    # O texto com erro de digitação chega ao modelo antes do encaminhamento.
    # A decisão simulada da IA escolhe o agente e a moeda, sem regra local nova.
    assert any(item.get("content") == QUESTION for item in original["messages"])
    assert session.tool_facts["transferir_para_agente"]["result"]["ok"] is True
    assert session.tool_facts["consultar_cotacao"]["result"] == {
        "moeda": "USD", "valor_em_brl": 5.1234,
        "atualizado_em": "23/10/2026 às 16:12:00 (horário de São Paulo)", "fonte": "AwesomeAPI"}
    assert_history_valid(session.history)
    assert not any(item.get("tool_calls") or item.get("tool_call_id") or item["role"] == "tool"
                   for item in session.history)


def test_repeated_transport_failure_preserves_session_and_allows_a_later_turn(
        transport_http, quote_http, repo_ana_620):
    create, clock = transport_http
    provider = create([TOOL_FAILURE, TOOL_FAILURE, TOOL_FAILURE,
        action("transferir_para_agente", destino="cambio"),
        action("consultar_cotacao", moeda="USD"), message(QUOTE_REPLY)])
    fx, quote_requests = quote_http
    orch = Orchestrator(repo_ana_620, fx, provider.client, MODEL, llm_mode="structured")
    session = logged_in(orch)
    before, correlation_id = csv(orch.repo), session.correlation_id

    failure = orch.handle(session, QUESTION)

    assert "indisponibilidade" in failure.casefold()
    assert not any(term in failure for term in ("HTTP", "JSON", "400", "429"))
    assert len(provider.requests) == 2
    assert not quote_requests and csv(orch.repo) == before
    assert session.authenticated and session.active_agent == "triagem" and not session.ended
    assert not session.structured_tool_models
    assert not session.processing_guard.lock.locked() and not clock.waits
    assert carrier_schema(provider.requests[1]) == provider.requests[0]["response_format"]["json_schema"]["schema"]

    answer = orch.handle(session, QUESTION)

    assert answer == QUOTE_REPLY
    assert session.correlation_id == correlation_id and session.turn == 2
    assert session.authenticated and session.active_agent == "cambio" and not session.ended
    assert session.structured_tool_models == {MODEL} and not session.processing_guard.lock.locked()
    assert len(provider.requests) == 6 and len(quote_requests) == 1
    assert not provider.pending and not clock.waits
    assert csv(orch.repo) == before and not orch.repo.list_requests()
    assert_history_valid(session.history)
