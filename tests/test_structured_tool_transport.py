"""Recuperação do envelope estrito com requisição real do SDK e HTTP simulado.

As respostas são predeterminadas e verificam protocolo e segurança local,
sem garantir a validade de uma geração remota do GPT-OSS.
"""
from __future__ import annotations

import copy
from contextlib import ExitStack
import json
from types import SimpleNamespace as NS

import httpx2 as httpx
from openai import OpenAI
import pytest
from jsonschema import Draft202012Validator

from banco_agil import structured_output as output
from banco_agil.agents import AGENTS
from banco_agil.exchange import ExchangeService
from banco_agil.llm import LlmRetryState, _build_payload, call_llm
from banco_agil.orchestrator import Orchestrator
from banco_agil.rate_limit import GovernedClient, RateLimiter
from banco_agil.session import Session
from banco_agil.tool_contracts import schemas_for
from provider_fakes import MODEL, OTHER_MODEL, daily, error, preserved_state
from orchestrator_fakes import logged_in
from fakes import Clock
from structured_fakes import action, message
from support import snapshot_csv as csv


WRAPPER = "banco_agil_resposta"
TOOL_ERROR = error(400, "tool_use_failed", "Tool choice is none, but model called a tool",
                   kind="invalid_request_error")
APPROVED = "Seu pedido foi aprovado e o novo limite de R$ 8.000,00 já está ativo."


def _outer(value=None, **overrides):
    raw = json.dumps({"resposta": value or message("Podemos continuar.")}, ensure_ascii=False)
    tool = NS(id="local-envelope", type="function",
              function=NS(name=WRAPPER, arguments=raw))
    fields = dict(content=None, tool_calls=[tool], refusal=None, function_call=None)
    fields.update(overrides)
    return NS(**fields)


@pytest.fixture
def envelope_http(monkeypatch):
    """Serializa pelo SDK com todas as respostas HTTP definidas em memória."""
    clock = Clock()
    monkeypatch.setattr("banco_agil.llm.time.sleep", clock.sleep)
    with ExitStack() as stack:
        def create(script, *, governed=False):
            pending, requests = list(script), []

            def sink(request):
                sent = json.loads(request.content)
                requests.append(sent)
                assert pending, "Unexpected HTTP generation beyond the finite test script"
                value = pending.pop(0)
                if isinstance(value, Exception):
                    raise value
                if isinstance(value, tuple):
                    status, body, headers = value
                    return httpx.Response(status, json=body, headers=headers)
                finish = "stop"
                if "_message" in value:
                    answer, finish = value["_message"], value.get("_finish", "stop")
                elif sent.get("tools", [{}])[0].get("function", {}).get("name") == WRAPPER:
                    # Serializa a resposta predeterminada no transporte solicitado.
                    answer = {"role": "assistant", "content": None, "tool_calls": [{
                        "id": "fixture-envelope", "type": "function", "function": {
                            "name": WRAPPER,
                            "arguments": json.dumps({"resposta": value}, ensure_ascii=False),
                        },
                    }]}
                    finish = "tool_calls"
                elif "response_format" in sent:
                    answer = {"role": "assistant", "content": json.dumps({"resposta": value}, ensure_ascii=False)}
                elif value["tipo"] == "acao":
                    proposal = value["acao"]
                    args = {k: v for k, v in (proposal["argumentos"] or {}).items() if v is not None}
                    answer = {"role": "assistant", "content": None, "tool_calls": [{
                        "id": "fixture-native", "type": "function", "function": {
                            "name": proposal["nome"], "arguments": json.dumps(args),
                        },
                    }]}
                    finish = "tool_calls"
                else:
                    answer = {"role": "assistant", "content": value["mensagem"]}
                return httpx.Response(200, json={"id": "fixture", "object": "chat.completion",
                    "created": 0, "model": sent["model"], "usage": {
                        "prompt_tokens": 1000, "completion_tokens": 100, "total_tokens": 1100},
                    "choices": [{"index": 0, "finish_reason": finish, "message": answer}]})

            sdk = stack.enter_context(OpenAI(api_key="local-only", base_url="http://memory.invalid/v1",
                max_retries=0, http_client=httpx.Client(transport=httpx.MockTransport(sink), trust_env=False)))
            bucket = RateLimiter(30, 30000, clock=clock, sleep=clock.sleep)
            client = GovernedClient(sdk, bucket, output_tokens=2048) if governed else sdk
            return NS(client=client, sdk=sdk, requests=requests, pending=pending, bucket=bucket)

        yield create, clock


def _tools(agent="triagem"):
    return schemas_for(AGENTS[agent], Session(active_agent=agent, authenticated=True, cpf="fixture"))


def _assert_wrapper(request, original_format):
    assert "response_format" not in request
    assert set(request["tool_choice"]) == {"type", "function"}
    assert request["tool_choice"] == {"type": "function", "function": {"name": WRAPPER}}
    assert request["parallel_tool_calls"] is False
    assert len(request["tools"]) == 1
    tool = request["tools"][0]
    assert tool["type"] == "function"
    assert tool["function"]["name"] == WRAPPER and tool["function"]["strict"] is True
    assert tool["function"]["parameters"] == original_format["json_schema"]["schema"]
    assert not request.get("disable_tool_validation")


@pytest.mark.parametrize("agent", list(AGENTS))
def test_recovery_request_keeps_exact_strict_exclusive_schema_without_mutation(agent):
    fmt = output.response_format(_tools(agent))
    before = copy.deepcopy(fmt)
    request = output.recovery_request(fmt)
    _assert_wrapper(request, fmt)
    assert fmt == before
    validator = Draft202012Validator(request["tools"][0]["function"]["parameters"])
    assert validator.is_valid({"resposta": message("Qual é sua dúvida?")})
    assert not validator.is_valid({"resposta": {"tipo": "mensagem", "mensagem": "texto",
        "acao": {"nome": "encerrar_atendimento", "argumentos": None}}})


@pytest.mark.parametrize("value", [message("Qual moeda você deseja consultar?"),
                                   action("transferir_para_agente", destino="cambio")])
def test_transport_decodes_same_envelope_as_content_and_keeps_action_arguments(value):
    tools = _tools()
    outer = _outer(value)
    expected = output.decode(NS(content=outer.tool_calls[0].function.arguments), tools, "stop")
    received = output.decode_transport(outer, tools, "tool_calls")
    assert received.envelope == expected.envelope and received.content == expected.content
    assert len(received.tool_calls) == len(expected.tool_calls)
    if received.tool_calls:
        assert received.tool_calls[0].function.name == expected.tool_calls[0].function.name
        assert received.tool_calls[0].function.arguments == expected.tool_calls[0].function.arguments


@pytest.mark.parametrize("finish", [None, "stop", "length", "content_filter", "function_call"])
def test_transport_rejects_wrong_or_incomplete_finalization(finish):
    with pytest.raises(output.StructuredOutputError):
        output.decode_transport(_outer(), _tools(), finish)


@pytest.mark.parametrize("case", ["no_calls", "two_calls", "direct_financial", "unknown_wrapper",
    "non_function", "text_and_call", "refusal", "legacy_call", "non_string_arguments"])
def test_transport_rejects_untrusted_outer_variants(case):
    outer = _outer()
    call = outer.tool_calls[0]
    if case == "no_calls": outer.tool_calls = []
    elif case == "two_calls": outer.tool_calls.append(copy.deepcopy(call))
    elif case == "direct_financial": call.function.name = "solicitar_aumento_limite"
    elif case == "unknown_wrapper": call.function.name = WRAPPER + "_inventada"
    elif case == "non_function": call.type = "custom"
    elif case == "text_and_call": outer.content = "Operação aprovada."
    elif case == "refusal": outer.refusal = "Não posso fazer isso."
    elif case == "legacy_call": outer.function_call = NS(name="solicitar_aumento_limite", arguments="{}")
    else: call.function.arguments = {"resposta": message("Olá")}
    with pytest.raises(output.StructuredOutputError):
        output.decode_transport(outer, _tools(), "tool_calls")


@pytest.mark.parametrize("raw", ["not JSON", "{}", "[]", "null",
    '{"resposta": {"tipo":"mensagem","mensagem":"x","acao":null}, "resposta":null}',
    '{"resposta":{"tipo":"acao","mensagem":null,"acao":{"nome":"solicitar_aumento_limite",'
        '"argumentos":{"novo_limite":NaN}}}}',
    json.dumps({"resposta": {"tipo": "acao", "mensagem": "Já aprovado", "acao": {
        "nome": "transferir_para_agente", "argumentos": {"destino": "cambio"}}}}),
    json.dumps({"resposta": action("solicitar_aumento_limite", novo_limite=8000)}),
    json.dumps({"resposta": {"tipo": "acao", "mensagem": None, "acao": {
        "nome": "transferir_para_agente", "argumentos": {"destino": "cambio", "extra": True}}}}),
])
def test_transport_reuses_existing_decoder_rejections_without_repairing_invalid_json(raw):
    outer = _outer()
    outer.tool_calls[0].function.arguments = raw
    with pytest.raises(output.StructuredOutputError):
        output.decode_transport(outer, _tools(), "tool_calls")


def test_transport_preserves_argument_string_for_duplicate_key_detection(monkeypatch):
    outer = _outer()
    raw = '{"resposta": {"tipo": "mensagem", "mensagem": "acentuação", "acao": null}}'
    outer.tool_calls[0].function.arguments = raw
    observed = []
    original = output.decode

    def observe(message, tools, finish):
        observed.append((message.content, finish))
        return original(message, tools, finish)

    monkeypatch.setattr(output, "decode", observe)
    output.decode_transport(outer, _tools(), "tool_calls")
    assert observed == [(raw, "stop")]


def test_transport_payload_replaces_protocol_only_and_keeps_trusted_state_and_history():
    session = Session(authenticated=True, cpf="fixture", nome="Pessoa fictícia",
                      history=[{"role": "user", "content": "Quanto está a moeda americana?"}])
    payload = _build_payload(AGENTS["triagem"], session)
    original = copy.deepcopy(payload)
    wire = output.build_payload(payload, _tools(), tool_transport=True)
    normal = output.build_payload(payload, _tools())
    assert payload == original
    assert wire[1:] == normal[1:]
    assert WRAPPER in wire[0]["content"]
    assert "sem ferramentas nativas" not in wire[0]["content"]
    assert "Nunca produza tool_calls" not in wire[0]["content"]
    assert AGENTS["triagem"].instructions() in wire[0]["content"]
    assert "fixture" not in json.dumps(wire)


@pytest.mark.parametrize("model", [MODEL, OTHER_MODEL])
@pytest.mark.parametrize("governed", [False, True])
def test_http_tool_failure_uses_one_strict_transport_recovery_and_preserves_session(envelope_http,
                                                                                 model, governed):
    create, clock = envelope_http
    provider = create([TOOL_ERROR, message("Podemos continuar.")], governed=governed)
    session = Session(authenticated=True, cpf="fixture", nome="Pessoa fictícia",
                      history=[{"role": "user", "content": "Quanto está o dólar americano?"}])
    before, budget = preserved_state(session), LlmRetryState()
    result = call_llm(provider.client, model, AGENTS["triagem"], session,
                      llm_mode="structured", retry_state=budget)
    assert result.content == "Podemos continuar." and not provider.pending
    assert len(provider.requests) == 2 and budget.generation_retry_used
    first, recovery = provider.requests
    _assert_wrapper(recovery, first["response_format"])
    assert recovery["model"] == model
    assert recovery["messages"][1:] == first["messages"][1:]
    assert preserved_state(session) == before and not clock.waits
    assert provider.sdk.max_retries == 0
    if governed:
        assert first["max_completion_tokens"] == recovery["max_completion_tokens"] == 2048
        assert len(provider.bucket.entries) == 2
        assert all(not entry.pending for entry in provider.bucket.entries)


def test_native_tool_failure_recovers_action_without_losing_context(envelope_http):
    create, clock = envelope_http
    provider = create([TOOL_ERROR, action("transferir_para_agente", destino="cambio")])
    session = Session(authenticated=True, cpf="fixture", nome="Pessoa fictícia",
                      history=[{"role": "user", "content": "Qual é a cotação do dólar?"}])
    before, budget = preserved_state(session), LlmRetryState()
    result = call_llm(provider.client, MODEL, AGENTS["triagem"], session,
                      llm_mode="structured", retry_state=budget)
    assert result.tool_calls[0].function.name == "transferir_para_agente"
    assert len(provider.requests) == 2 and {r["model"] for r in provider.requests} == {MODEL}
    _assert_wrapper(provider.requests[-1], provider.requests[0]["response_format"])
    assert budget.generation_retry_used and not budget.rate_limit_retry_used
    assert preserved_state(session) == before and not clock.waits


@pytest.mark.parametrize("failure", [TOOL_ERROR,
    error(400, "json_validate_failed", kind="invalid_request_error"),
    error(400, "invalid_request_error", param="tools"),
    daily(),
    TimeoutError("local-only timeout")])
def test_failed_transport_never_adds_another_generation_recovery(envelope_http, failure):
    create, clock = envelope_http
    provider = create([TOOL_ERROR, failure])
    session, budget = Session(), LlmRetryState()
    result = call_llm(provider.client, MODEL, AGENTS["triagem"], session,
                      llm_mode="structured", retry_state=budget, max_retries=1)
    assert result is None and len(provider.requests) == 2
    assert not provider.pending and not clock.waits and budget.generation_retry_used
    assert not budget.rate_limit_retry_used
    assert {r["model"] for r in provider.requests} == {MODEL}


@pytest.mark.parametrize("code,param", [("tool_use_failed", "tools"),
                                        ("tool_use_failed", "response_format"),
                                        ("json_validate_failed", "response_format")])
def test_invalid_requests_never_enter_transport_recovery(envelope_http, code, param):
    create, _ = envelope_http
    provider = create([error(400, code, "Invalid schema", kind="invalid_request_error", param=param)])
    session, budget = Session(), LlmRetryState()
    assert call_llm(provider.client, OTHER_MODEL, AGENTS["triagem"], session,
                    llm_mode="structured", retry_state=budget) is None
    assert len(provider.requests) == 1 and not budget.generation_retry_used
    assert session.last_failure_cause == "invalid_request"


def test_consumed_generation_budget_does_not_gain_transport_retry(envelope_http):
    create, _ = envelope_http
    provider = create([TOOL_ERROR])
    budget = LlmRetryState(generation_retry_used=True)
    assert call_llm(provider.client, OTHER_MODEL, AGENTS["triagem"], Session(),
                    llm_mode="structured", retry_state=budget) is None
    assert len(provider.requests) == 1


def test_successful_transport_is_remembered_only_for_the_same_model_and_session(envelope_http):
    create, _ = envelope_http
    provider = create([TOOL_ERROR, message("Qual é o seu CPF?"),
                       message("Qual assunto deseja tratar?"), message("Qual é o seu CPF?"),
                       message("Qual é o seu CPF?")])
    session, budget = Session(), LlmRetryState()
    first = call_llm(provider.client, OTHER_MODEL, AGENTS["triagem"], session,
                     llm_mode="structured", retry_state=budget)
    assert first.content and OTHER_MODEL in session.structured_tool_models
    assert budget.generation_retry_used

    # A mudança de etapa reconstrói o catálogo conforme a autenticação atual.
    session.authenticated, session.cpf = True, "fixture"
    assert call_llm(provider.client, OTHER_MODEL, AGENTS["triagem"], session,
                    llm_mode="structured", retry_state=budget).content
    current = output.response_format(schemas_for(AGENTS["triagem"], session))
    _assert_wrapper(provider.requests[2], current)
    assert current != provider.requests[0]["response_format"]

    # Outra configuração de modelo ou sessão nova começa com o transporte JSON original.
    assert call_llm(provider.client, MODEL, AGENTS["triagem"], session,
                    llm_mode="structured", retry_state=budget).content
    assert "response_format" in provider.requests[3]
    fresh = Session()
    assert call_llm(provider.client, OTHER_MODEL, AGENTS["triagem"], fresh,
                    llm_mode="structured").content
    assert "response_format" in provider.requests[4]
    assert not fresh.structured_tool_models and session.structured_tool_models == {OTHER_MODEL}
    assert not provider.pending


def test_cached_session_from_before_transport_field_preserves_authentication_and_flow(envelope_http):
    create, _ = envelope_http
    provider = create([TOOL_ERROR, message("Podemos continuar.")])
    session = Session(authenticated=True, cpf="fixture", active_agent="entrevista")
    session.flow.interview_answers = {"renda_mensal": 6000}
    del session.structured_tool_models
    before = preserved_state(session)
    result = call_llm(provider.client, OTHER_MODEL, AGENTS["entrevista"], session, llm_mode="structured")
    assert result.content == "Podemos continuar."
    assert session.structured_tool_models == {OTHER_MODEL} and preserved_state(session) == before


def test_learned_transport_json_recovery_keeps_carrier_and_noncontradictory_protocol(envelope_http):
    create, _ = envelope_http
    provider = create([error(400, "json_validate_failed", kind="invalid_request_error"),
                       message("Podemos continuar.")])
    session = Session(structured_tool_models={OTHER_MODEL})
    budget = LlmRetryState()
    result = call_llm(provider.client, OTHER_MODEL, AGENTS["triagem"], session,
                      llm_mode="structured", retry_state=budget)
    assert result.content and budget.generation_retry_used and len(provider.requests) == 2
    for request in provider.requests:
        _assert_wrapper(request, output.response_format(schemas_for(AGENTS["triagem"], session)))
        assert "sem ferramentas nativas" not in request["messages"][0]["content"]
    assert provider.requests[0]["messages"] != provider.requests[1]["messages"]


def test_transport_recovery_respects_input_budget_before_http(envelope_http, monkeypatch):
    from banco_agil import llm
    create, _ = envelope_http
    provider = create([TOOL_ERROR])
    estimates, observed = iter([100, llm.MAX_INPUT_TOKENS + 1]), []

    def estimate(*args, **kwargs):
        observed.append((args, kwargs))
        return next(estimates)

    monkeypatch.setattr(llm, "_estimate_request_tokens", estimate)
    session, budget = Session(), LlmRetryState()
    assert call_llm(provider.client, OTHER_MODEL, AGENTS["triagem"], session,
                    llm_mode="structured", retry_state=budget) is None
    assert len(provider.requests) == 1 and budget.generation_retry_used
    assert observed[1][0][1:3] == ([], None)
    assert observed[1][1]["request_options"]["tools"][0]["function"]["strict"] is True
    assert session.last_failure_cause == "input_too_long" and not session.structured_tool_models


@pytest.mark.parametrize("outcome", [TOOL_ERROR,
    {"_message": {"role": "assistant", "content": "{}"}, "_finish": "stop"},
    {"_message": {"role": "assistant", "content": None, "tool_calls": [{
        "id": "untrusted", "type": "function", "function": {"name": WRAPPER, "arguments": "{}"}}]},
        "_finish": "tool_calls"}])
def test_failed_transport_never_marks_model_as_working(envelope_http, outcome):
    create, _ = envelope_http
    provider = create([TOOL_ERROR, outcome])
    session = Session()
    assert call_llm(provider.client, OTHER_MODEL, AGENTS["triagem"], session,
                    llm_mode="structured") is None
    assert not session.structured_tool_models and len(provider.requests) == 2


def test_daily_error_after_transport_recovery_stops_without_losing_state(envelope_http):
    create, clock = envelope_http
    provider = create([TOOL_ERROR, daily()])
    session, budget = Session(authenticated=True, cpf="fixture"), LlmRetryState()
    before = preserved_state(session)
    assert call_llm(provider.client, MODEL, AGENTS["triagem"], session,
                    llm_mode="structured", retry_state=budget) is None
    assert len(provider.requests) == 2 and {r["model"] for r in provider.requests} == {MODEL}
    _assert_wrapper(provider.requests[1], provider.requests[0]["response_format"])
    assert budget.generation_retry_used and not budget.rate_limit_retry_used and not clock.waits
    assert not session.structured_tool_models and preserved_state(session) == before
    assert session.last_failure_cause == "daily_token_limit"


def test_failed_generation_is_never_used_as_an_executable_financial_proposal(envelope_http, repo_ana_620):
    create, _ = envelope_http
    failed = error(400, "tool_use_failed", "Tool choice is none, but model called a tool",
        kind="invalid_request_error", failed_generation=json.dumps({"resposta":
            action("solicitar_aumento_limite", novo_limite=8000)}))
    provider = create([failed, message("Podemos conversar sobre a proposta.")])
    orch = Orchestrator(repo_ana_620, ExchangeService(), provider.client, OTHER_MODEL, llm_mode="structured")
    session = logged_in(orch, agent="credito")
    session.credit_proposal = dict(valor=8000, modalidade="limite_total", novo_limite=8000, confirmada=False)
    before = csv(orch.repo)
    orch.handle(session, "Ainda preciso entender os dados.")
    assert csv(orch.repo) == before and not orch.repo.list_requests()
    assert not session.credit_proposal["confirmada"]
    assert "failed_generation" not in json.dumps(provider.requests)
    assert not provider.pending


@pytest.mark.parametrize("model,mode", [("another-model", "structured"), (OTHER_MODEL, "native")])
def test_native_mode_and_other_models_keep_existing_transport(envelope_http, model, mode):
    create, _ = envelope_http
    provider = create([TOOL_ERROR, message("Podemos continuar.")])
    assert call_llm(provider.client, model, AGENTS["triagem"], Session(), llm_mode=mode).content
    assert len(provider.requests) == 2
    assert not any(request.get("tool_choice", {}).get("function", {}).get("name") == WRAPPER
                   for request in provider.requests)


def test_transport_action_does_not_create_financial_consent(envelope_http, repo_ana_620):
    create, clock = envelope_http
    provider = create([TOOL_ERROR, action("solicitar_aumento_limite", novo_limite=8000)])
    orch = Orchestrator(repo_ana_620, ExchangeService(), provider.client, MODEL, llm_mode="structured")
    session = logged_in(orch, agent="credito")
    session.credit_proposal = dict(valor=8000, modalidade="limite_total", novo_limite=8000, confirmada=False)
    before, proposal = csv(orch.repo), copy.deepcopy(session.credit_proposal)
    orch.handle(session, "Ainda quero conferir os dados antes de decidir.")
    assert csv(orch.repo) == before and not orch.repo.list_requests()
    assert session.credit_proposal == proposal and session.authenticated and not session.ended
    assert len(provider.requests) == 2 and not clock.waits
    assert {r["model"] for r in provider.requests} == {MODEL}


@pytest.mark.parametrize("stage", ["before_operation", "after_operation"])
@pytest.mark.parametrize("repeat", [False, True])
def test_transport_recovery_never_duplicates_a_confirmed_financial_effect(envelope_http, repo_ana_620,
                                                                        stage, repeat):
    create, clock = envelope_http
    operation = action("solicitar_aumento_limite", novo_limite=8000)
    close = operation if repeat else message(APPROVED)
    script = ([TOOL_ERROR, operation, close] if stage == "before_operation"
              else [operation, TOOL_ERROR, close])
    provider = create(script)
    orch = Orchestrator(repo_ana_620, ExchangeService(), provider.client, MODEL, llm_mode="structured")
    session = logged_in(orch, agent="credito")
    session.credit_proposal = dict(valor=8000, modalidade="limite_total", novo_limite=8000, confirmada=False)
    before = csv(orch.repo)
    answer = orch.handle(session, "confirmo")
    recorded = orch.repo.list_requests()
    assert len(recorded) == 1 and recorded[0]["status_pedido"] == "aprovado"
    assert float(recorded[0]["novo_limite_solicitado"]) == 8000
    assert orch.repo.get_client(session.cpf)["limite_credito"] == 8000
    assert "8.000,00" in answer and "aprovado" in answer.lower()
    assert session.credit_proposal["executada"] and session.authenticated and not session.ended
    assert csv(orch.repo)["score_limite.csv"] == before["score_limite.csv"]
    assert not provider.pending and not clock.waits
    assert {r["model"] for r in provider.requests} == {MODEL}
    assert not any(item.get("tool_calls") or item["role"] == "tool" for item in session.history)
    assert WRAPPER not in json.dumps(session.history)
