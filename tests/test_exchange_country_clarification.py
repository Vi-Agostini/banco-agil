"""A IA interpreta país e moeda; as taxas vêm do HTTP simulado."""
import json

import pytest
import requests

from banco_agil.agents import AGENTS
from banco_agil.exchange import ExchangeError, ExchangeService
from banco_agil.tools import consultar_cotacao
from provider_fakes import Resp
from orchestrator_fakes import logged_in
from structured_fakes import action, check_protocol, make, message
from support import snapshot_csv as csv


def test_currency_country_reference_and_clarification_are_in_both_wire_protocols(repo_ana_620):
    # Confere a instrução enviada, sem avaliar a compreensão de países pelo modelo.
    for mode in ("native", "structured"):
        orch = make(repo_ana_620, [message("Você se refere ao dólar de qual país?")], mode)
        s = logged_in(orch, agent="cambio")
        before = csv(orch.repo)
        orch.handle(s, "quero dólar, mas não o americano")
        prompt = orch.client.requests[0]["messages"][0]["content"]
        assert "ariary de Madagascar=MGA" in prompt
        assert "não precisa" in prompt and "código" in prompt
        assert "ambiguidade" in prompt and "não suponha" in prompt
        assert s.flow.unclear_count == 0 and not s.ended
        assert csv(orch.repo) == before
        check_protocol(orch, s)


@pytest.mark.parametrize("mode", ["native", "structured"])
def test_madagascar_uses_interpreted_iso_and_real_service_with_mocked_http(repo_ana_620, mode):
    # A taxa fictícia da fixture não representa uma cotação real de Madagascar.
    calls = []

    def get(url, **kwargs):
        calls.append((url, kwargs))
        assert url.endswith("/MGA-BRL") and kwargs["timeout"] == 8
        return Resp({"MGABRL": {"bid": "0.001234", "create_date": "2026-10-22 10:00:00"}})

    orch = make(repo_ana_620, [action("consultar_cotacao", moeda="MGA"),
        message("A última cotação disponível de MGA é R$ 0,0012. Fonte: AwesomeAPI.")], mode)
    orch.fx._get = get
    s = logged_in(orch, agent="cambio")
    before = csv(orch.repo)
    response = orch.handle(s, "Qual a cotação da moeda de Madagascar hoje?")
    assert len(calls) == 1
    assert s.tool_facts["consultar_cotacao"]["result"]["moeda"] == "MGA"
    assert "0,0012" in response
    assert s.flow.unclear_count == 0 and s.auth_failures == 0
    assert s.authenticated and not s.ended and csv(orch.repo) == before
    check_protocol(orch, s)


@pytest.mark.parametrize("failure", [requests.Timeout, requests.ConnectionError])
def test_unavailable_quote_is_provider_problem_not_customer_error(auth_ctx_ana_620, failure):
    calls = []

    def get(url, **kwargs):
        calls.append(url)
        raise failure("private-host-should-not-appear")

    ctx = auth_ctx_ana_620
    ctx.fx = ExchangeService(http_get=get)
    before = csv(ctx.repo)
    result = consultar_cotacao(ctx, "MGA")
    assert len(calls) == 2  # Mantém as duas fontes existentes, sem novo provedor ou retentativa.
    assert result["codigo"] == "cotacao_indisponivel"
    assert "fontes disponíveis" in result["erro"]
    assert "Não atribua a falha ao cliente" in result["instrucao"]
    assert "private-host" not in json.dumps(result)
    assert "valor_em_brl" not in result and "inválida" not in result["erro"]
    assert ctx.session.flow.unclear_count == 0 and ctx.session.auth_failures == 0
    assert csv(ctx.repo) == before


def test_absent_coverage_does_not_claim_a_quote_or_invalid_customer_data(auth_ctx_ana_620):
    calls = []

    def get(url, **kwargs):
        calls.append(url)
        return Resp(status=404)

    ctx = auth_ctx_ana_620
    ctx.fx = ExchangeService(http_get=get)
    result = consultar_cotacao(ctx, "MGA")
    assert len(calls) == 2 and result["codigo"] == "cotacao_indisponivel"
    assert "valor_em_brl" not in result
    assert "fontes disponíveis" in result["erro"]
    assert ctx.session.flow.unclear_count == 0 and not ctx.session.ended


@pytest.mark.parametrize("code", ["??", "MADAGASCAR", "", None, 4])
def test_unidentified_iso_requests_name_or_country_without_demanding_iso(code):
    calls = []
    fx = ExchangeService(http_get=lambda *a, **kw: calls.append(a))
    with pytest.raises(ExchangeError) as error:
        fx.get_rate(code)
    assert error.value.code == "moeda_nao_identificada"
    assert "nome ou o país" in str(error.value)
    assert "código" not in str(error.value) and not calls


def test_quote_clarification_does_not_change_financial_contracts():
    tool = AGENTS["cambio"].get_tool("consultar_cotacao")
    assert tool.parameters["required"] == ["moeda"]
    assert tool.parameters["properties"]["moeda"]["type"] == "string"
    assert AGENTS["cambio"].get_tool("registrar_entrada_invalida") is not None
    assert AGENTS["cambio"].get_tool("solicitar_aumento_limite") is None
    assert AGENTS["cambio"].get_tool("registrar_entrevista_e_calcular_score") is None


@pytest.mark.parametrize("mode", ["native", "structured"])
@pytest.mark.parametrize("currency", ["MGA", "??"])
def test_source_failure_or_unidentified_currency_never_penalizes_session(repo_ana_620, mode, currency):
    text = ("Não consegui consultar essa moeda agora. Posso consultar outra moeda?" if currency == "MGA"
            else "Qual moeda você deseja consultar? Pode informar o nome ou o país.")
    orch = make(repo_ana_620, [action("consultar_cotacao", moeda=currency), message(text)], mode)
    calls = []

    def get(url, **kwargs):
        calls.append(url)
        return Resp(status=404)

    orch.fx._get = get
    s = logged_in(orch, agent="cambio")
    s.flow.unclear_count = 1  # A falha anterior de outro assunto não deve aumentar aqui.
    before = csv(orch.repo)
    response = orch.handle(s, "Qual a cotação da moeda de Madagascar hoje?")
    assert "Tentativa" not in response and "inválida" not in response
    assert s.active_agent == "cambio" and s.authenticated and not s.ended
    assert s.flow.unclear_count <= 1 and s.flow.attempts == 0
    assert len(calls) == (2 if currency == "MGA" else 0)
    assert csv(orch.repo) == before
    check_protocol(orch, s)
