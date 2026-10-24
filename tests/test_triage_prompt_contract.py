"""Prompt, catálogo e transporte reais; HTTP em memória, sem avaliar um modelo real."""
from __future__ import annotations

import copy
import json
import re
from contextlib import contextmanager
from dataclasses import replace

import httpx2 as httpx
import pytest
from jsonschema import Draft202012Validator
from openai import OpenAI

from banco_agil.agents import AGENTS
from banco_agil.exchange import ExchangeService
from banco_agil.flow import Awaiting
from banco_agil.llm import LlmRetryState, call_llm
from banco_agil.orchestrator import Orchestrator
from banco_agil.session import Session
from banco_agil.tool_contracts import schemas_for
from fakes import assert_history_valid
from structured_fakes import action, message, strict_spec
from support import snapshot_csv as csv


MODELS = ("openai/gpt-oss-120b", "openai/gpt-oss-20b")
MODES = ("native", "structured")


def _offered(request):
    wrapper = request.get("tool_choice", {}).get("function", {}).get("name") == "banco_agil_resposta"
    if "response_format" not in request and not wrapper:
        return {item["function"]["name"] for item in request.get("tools", ())}
    variants = strict_spec(request)["schema"]["properties"]["resposta"]["anyOf"]
    return {branch["properties"]["nome"]["enum"][0]
            for variant in variants if "anyOf" in variant["properties"]["acao"]
            for branch in variant["properties"]["acao"]["anyOf"]}


def _completion(request, value, number):
    wrapper = request.get("tool_choice", {}).get("function", {}).get("name") == "banco_agil_resposta"
    native_action = "response_format" not in request and not wrapper and value["tipo"] == "acao"
    if "response_format" in request:
        content = json.dumps({"resposta": value}, ensure_ascii=False)
    else:
        content = value["mensagem"]
    response = {"role": "assistant", "content": content}
    if wrapper:
        response["content"] = None
        response["tool_calls"] = [{"id": f"mock-envelope-{number}", "type": "function", "function": {
            "name": "banco_agil_resposta", "arguments": json.dumps({"resposta": value}, ensure_ascii=False),
        }}]
    elif native_action:
        proposal = value["acao"]
        response["tool_calls"] = [{"id": f"mock-action-{number}", "type": "function", "function": {
            "name": proposal["nome"], "arguments": json.dumps(proposal["argumentos"] or {}),
        }}]
    return {"id": f"mock-completion-{number}", "object": "chat.completion", "created": 0,
            "model": request["model"], "choices": [{"index": 0, "message": response,
                "finish_reason": "tool_calls" if native_action or wrapper else "stop"}]}


@contextmanager
def _provider(script):
    """O SDK serializa cada request, mas o transporte não abre conexões."""
    pending, captured = list(script), []

    def respond(http_request):
        body = json.loads(http_request.content)
        captured.append(body)
        assert pending, "unexpected generation beyond the finite local script"
        value = pending.pop(0)
        if callable(value):
            value = value(body)
        if isinstance(value, tuple):
            status, error = value
            return httpx.Response(status, request=http_request, json={"error": error})
        return httpx.Response(200, request=http_request, json=_completion(body, value, len(captured)))

    with OpenAI(api_key="local-test-only", base_url="http://memory.invalid/v1", max_retries=0,
                http_client=httpx.Client(transport=httpx.MockTransport(respond), trust_env=False)) as client:
        yield client, captured, pending


def _error(code="tool_use_failed", *, param=""):
    return (400, {"type": "invalid_request_error", "code": code, "param": param,
                  "message": "Generated output was rejected; private provider detail."})


def _assert_wire(request, mode):
    if mode == "structured":
        spec = strict_spec(request)
        assert spec["strict"] is True
        validator = Draft202012Validator(spec["schema"])
        validator.check_schema(spec["schema"])
        assert validator.is_valid({"resposta": message("Como posso ajudar?")})
        # Uma mensagem de conversa não pode carregar uma ação financeira junto.
        assert not validator.is_valid({"resposta": {
            "tipo": "mensagem", "mensagem": "Confirmado", "acao": {
                "nome": "solicitar_aumento_limite", "argumentos": {"novo_limite": 9000},
            },
        }})
        assert all(item["role"] in {"system", "assistant", "user"} for item in request["messages"])
        assert not any(item.get("tool_calls") or "tool_call_id" in item for item in request["messages"])
        prefix = request["messages"][0]["content"]
        if "ESCOPO:" in prefix:
            assert prefix.index("JSON") < prefix.index("ESCOPO:")
    else:
        assert "response_format" not in request
        assert "parallel_tool_calls" not in request


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("stage", ("empty", "pending_birth", "authenticated"))
def test_triage_actual_wire_matches_local_authentication_and_current_catalog(repo_ana_620, model, mode, stage):
    session = Session(llm_mode=mode)
    session.history = [{"role": "assistant", "content": "Como posso ajudar?"},
                       {"role": "user", "content": "Gostaria de entender como funciona."}]
    if stage == "pending_birth":
        session.flow.pending_cpf = "11144477735"
        session.flow.awaiting = Awaiting.BIRTH_DATE
    elif stage == "authenticated":
        session.authenticated, session.cpf, session.nome = True, "11144477735", "Ana Souza"
    before = copy.deepcopy((session.history, session.flow, session.authenticated, session.cpf,
                            session.auth_failures, session.credit_proposal, session.tool_facts))
    before_csv = csv(repo_ana_620)
    expected = {tool["function"]["name"] for tool in schemas_for(AGENTS["triagem"], session)}
    text = {"empty": "Qual é o seu CPF?", "pending_birth": "Qual é a sua data de nascimento?",
            "authenticated": "Qual assunto você gostaria de tratar?"}[stage]
    with _provider([message(text)]) as (client, captured, pending):
        result = call_llm(client, model, AGENTS["triagem"], session, llm_mode=mode)
    assert result is not None and len(captured) == 1 and not pending
    request = captured[0]
    _assert_wire(request, mode)
    assert request["model"] == model
    assert _offered(request) == expected
    assert "autenticar_cliente" not in expected
    assert ("transferir_para_agente" in expected) is (stage == "authenticated")
    instructions = "\n".join(item["content"] for item in request["messages"] if item["role"] == "system")
    # O prompt não pode exigir autenticação por uma ação que o catálogo
    # deliberadamente não oferece ao modelo.
    assert not re.search(r"\b(?:chame|use)\s+autenticar_cliente\b", instructions, re.IGNORECASE)
    if stage == "authenticated":
        assert "Cliente autenticado" in instructions and "Ana Souza" in instructions
    elif stage == "pending_birth":
        assert not re.search(r"responda cordialmente e peça o CPF\b", instructions, re.IGNORECASE)
    assert "11144477735" not in json.dumps(request)
    assert before == (session.history, session.flow, session.authenticated, session.cpf,
                      session.auth_failures, session.credit_proposal, session.tool_facts)
    assert csv(repo_ana_620) == before_csv


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("mode", MODES)
def test_mock_provider_rejects_legacy_hidden_auth_command_and_accepts_current_contract(repo_ana_620, model, mode):
    """O provedor simulado reproduz o conflito de contrato, não o raciocínio da IA."""
    legacy_agent = replace(AGENTS["triagem"], scope_prompt=AGENTS["triagem"].scope_prompt +
        "\nSe o cliente mandar CPF e data de nascimento, use autenticar_cliente imediatamente.")
    legacy_session = Session(llm_mode=mode)
    current_session = Session(llm_mode=mode)
    legacy_session.authenticated = current_session.authenticated = True
    legacy_session.cpf = current_session.cpf = "11144477735"
    before = csv(repo_ana_620)
    observed_status = []

    def reject_hidden_auth_command(body):
        prompt = "\n".join(item["content"] for item in body["messages"] if item["role"] == "system")
        conflict = ("autenticar_cliente" not in _offered(body)
                    and re.search(r"\b(?:chame|use)\s+autenticar_cliente\b", prompt, re.IGNORECASE))
        observed_status.append(400 if conflict else 200)
        return _error() if conflict else message("Qual assunto você gostaria de tratar?")

    with _provider([reject_hidden_auth_command] * 3) as (client, captured, pending):
        rejected = call_llm(client, model, legacy_agent, legacy_session, llm_mode=mode)
        accepted = call_llm(client, model, AGENTS["triagem"], current_session, llm_mode=mode)
    assert rejected is None and accepted is not None and not pending
    assert observed_status == [400, 400, 200]
    assert _offered(captured[0]) == _offered(captured[1]) == _offered(captured[2])
    for request in captured:
        _assert_wire(request, mode)
    if mode == "structured":
        assert captured[0]["response_format"] == captured[2]["response_format"]
    assert csv(repo_ana_620) == before
    assert not legacy_session.history and not current_session.history


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("mode", MODES)
def test_combined_authentication_still_uses_ai_handoff_and_real_read_only_credit_tool(repo_ana_620, model, mode):
    before = csv(repo_ana_620)
    with _provider([action("transferir_para_agente", destino="credito"),
                    action("consultar_limite"), message("Seu limite atual é R$ 5.000,00.")]) as (client, captured, pending):
        orch = Orchestrator(repo_ana_620, ExchangeService(), client, model, llm_mode=mode)
        session = orch.new_session()
        response = orch.handle(session, "11144477735 12/05/1990. Quanto posso utilizar no cartão?")
    assert "5.000,00" in response and not pending and len(captured) == 3
    assert "triagem" in captured[0]["messages"][0]["content"].lower() or "recepção" in captured[0]["messages"][0]["content"]
    assert "transferir_para_agente" in _offered(captured[0])
    assert "consultar_limite" not in _offered(captured[0])
    assert "consultar_limite" in _offered(captured[1])
    assert all(request["model"] == model for request in captured)
    for request in captured:
        _assert_wire(request, mode)
    assert session.authenticated and session.auth_failures == 0 and session.active_agent == "credito"
    assert session.tool_facts["transferir_para_agente"]["result"]["ok"] is True
    assert session.tool_facts["consultar_limite"]["result"]["limite_credito_atual"] == 5000
    assert not session.credit_proposal and not repo_ana_620.list_requests()
    assert csv(repo_ana_620) == before
    assert_history_valid(session.history)


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("code", ("tool_use_failed", "json_validate_failed"))
def test_generation_400_has_one_shared_recovery_without_changing_schema_or_session(repo_ana_620, model, mode, code):
    session = Session(llm_mode=mode)
    session.flow.pending_cpf = "11144477735"
    session.flow.awaiting = Awaiting.BIRTH_DATE
    session.history = [{"role": "user", "content": "Pode me explicar o que falta?"}]
    before = copy.deepcopy((session.history, session.flow, session.credit_proposal))
    before_csv, budget = csv(repo_ana_620), LlmRetryState()
    with _provider([_error(code), message("Para continuarmos, informe sua data de nascimento."),
                    _error(code)]) as (client, captured, pending):
        first = call_llm(client, model, AGENTS["triagem"], session, retry_state=budget, llm_mode=mode)
        assert first is not None and len(captured) == 2 and budget.generation_retry_used
        second = call_llm(client, model, AGENTS["triagem"], session, retry_state=budget, llm_mode=mode)
    assert second is None and len(captured) == 3 and not pending
    assert session.last_failure_cause == code
    assert captured[0]["messages"] != captured[1]["messages"]
    assert _offered(captured[0]) == _offered(captured[1]) == _offered(captured[2])
    if mode == "structured":
        assert strict_spec(captured[0]) == strict_spec(captured[1]) == strict_spec(captured[2])
    for request in captured:
        _assert_wire(request, mode)
    assert before == (session.history, session.flow, session.credit_proposal)
    assert not session.authenticated and not session.auth_failures and csv(repo_ana_620) == before_csv


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("mode", MODES)
def test_invalid_request_400_never_retries_unchanged_request(repo_ana_620, model, mode):
    session = Session(llm_mode=mode)
    before = csv(repo_ana_620)
    with _provider([_error("json_validate_failed", param="response_format")]) as (client, captured, pending):
        result = call_llm(client, model, AGENTS["triagem"], session, llm_mode=mode)
    assert result is None and len(captured) == 1 and not pending
    assert session.last_failure_cause == "invalid_request"
    assert not session.authenticated and session.auth_failures == 0 and csv(repo_ana_620) == before


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("mode", MODES)
def test_unoffered_authentication_cannot_be_executed_after_prompt_change(repo_ana_620, model, mode, monkeypatch):
    before = csv(repo_ana_620)
    monkeypatch.setattr(repo_ana_620, "authenticate", lambda *args: pytest.fail("LLM cannot supply authentication credentials"))
    with _provider([action("autenticar_cliente", cpf="11144477735", data_nascimento="12/05/1990")]) as (client, captured, pending):
        orch = Orchestrator(repo_ana_620, ExchangeService(), client, model, llm_mode=mode)
        session = orch.new_session()
        response = orch.handle(session, "Gostaria de conhecer as opções de crédito.")
    assert response and len(captured) == 1 and not pending
    assert session.last_failure_category in {"invalid_tool_response", "invalid_structured_response"}
    assert not session.authenticated and session.auth_failures == 0 and not session.credit_proposal
    assert session.active_agent == "triagem" and csv(repo_ana_620) == before


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("mode", MODES)
def test_three_complete_authentication_failures_still_close_without_provider_or_financial_effect(repo_ana_620, model, mode):
    before = csv(repo_ana_620)
    with _provider([]) as (client, captured, pending):
        orch = Orchestrator(repo_ana_620, ExchangeService(), client, model, llm_mode=mode)
        session = orch.new_session()
        for failures in (1, 2, 3):
            response = orch.handle(session, "11144477735 01/01/2000")
            assert response and session.auth_failures == failures
            assert session.ended is (failures == 3)
    assert not captured and not pending and not session.authenticated
    assert csv(repo_ana_620) == before and not repo_ana_620.list_requests()
