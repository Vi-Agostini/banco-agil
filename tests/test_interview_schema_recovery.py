"""Recuperação das falhas de schema da entrevista com HTTP local."""
import copy
import json
import logging
from types import SimpleNamespace as NS

import httpx2 as httpx
from jsonschema import Draft202012Validator
from openai import OpenAI
import pytest

from banco_agil.agents import AGENTS
from banco_agil.exchange import ExchangeService
from banco_agil.flow import FlowController
from banco_agil.llm import _build_payload, _tool_schemas, call_llm
from banco_agil.orchestrator import Orchestrator
from banco_agil.provider_diagnostics import generation_report
from banco_agil.structured_output import StructuredOutputError, build_payload, decode, response_format
from fakes import failure, Clock
from orchestrator_fakes import logged_in
from support import snapshot_csv as csv
from structured_fakes import make, message


FIELDS = dict(renda_mensal=90000, tipo_emprego="formal", despesas_fixas=2000,
              num_dependentes=0, tem_dividas=False)
COLLECT = "registrar_respostas_entrevista"


def proposal(args):
    # Não usa o helper que preenche automaticamente as chaves ausentes.
    return {"tipo": "acao", "mensagem": None, "acao": {"nome": COLLECT, "argumentos": args}}


def names(request):
    branches = request["response_format"]["json_schema"]["schema"]["properties"]["resposta"]["anyOf"]
    return {b["properties"]["nome"]["enum"][0]
            for variant in branches[1:] for b in variant["properties"]["acao"]["anyOf"]}


@pytest.mark.parametrize("field", FIELDS)
def test_partial_wire_must_include_nullable_keys_but_native_contract_stays_partial(repo_ana_620, field):
    orch = make(repo_ana_620, [])
    session = logged_in(orch, agent="entrevista")
    FlowController.start_interview(session)
    agent = AGENTS["entrevista"]
    tools = _tool_schemas(agent, session)
    original = copy.deepcopy(tools)
    payload = _build_payload(agent, session)
    before = copy.deepcopy(payload)
    structured = build_payload(payload, tools)
    prompt = "\n".join(m["content"] for m in structured if m["role"] == "system")
    assert "nunca envie null" not in prompt
    assert "argumentos deve conter todas as chaves" in prompt
    assert all(key in prompt for key in FIELDS)
    assert "Campos sem resposta nova: null" in prompt
    assert tools == original and payload == before
    native = next(t["function"] for t in tools if t["function"]["name"] == COLLECT)
    assert native["parameters"]["required"] == []
    assert "nunca use null" in native["description"]

    fmt = response_format(tools)
    validator = Draft202012Validator(fmt["json_schema"]["schema"])
    partial = {"resposta": proposal({field: FIELDS[field]})}
    assert not validator.is_valid(partial)
    error = failure("json_validate_failed", failed_generation=json.dumps(partial))
    report = generation_report(error, fmt)
    assert report["generation_status"] == "schema_mismatch"
    assert set(report["schema_issues"]) == {
        f"$.resposta.acao.argumentos.{key}:required" for key in FIELDS if key != field}
    with pytest.raises(StructuredOutputError):
        decode(NS(content=json.dumps(partial)), tools, "stop")
    complete = {"resposta": proposal({key: FIELDS[key] if key == field else None for key in FIELDS})}
    assert validator.is_valid(complete) and fmt["json_schema"]["strict"] is True
    decoded = decode(NS(content=json.dumps(complete)), tools, "stop")
    assert json.loads(decoded.tool_calls[0].function.arguments) == {field: FIELDS[field]}


@pytest.mark.parametrize("throttled", [False, True])
@pytest.mark.parametrize("confirmation_exhausted", [False, True])
def test_sdk_confirmation_and_partial_employment_recover_without_losing_income_or_executing_credit(
        repo_ana_620, caplog, monkeypatch, throttled, confirmation_exhausted):
    clock = Clock()
    monkeypatch.setattr("banco_agil.llm.time.sleep", clock.sleep)
    requests = []
    full_employment = {key: "formal" if key == "tipo_emprego" else None for key in FIELDS}
    script = [
        (400, proposal({key: 90000 if key == "renda_mensal" else None for key in FIELDS})),
        (200, message("Qual é o seu tipo de emprego?")),
        (400, proposal({"tipo_emprego": "formal"})),
        (200, proposal(full_employment)),
        (200, message("Você confirma que seu emprego é formal?")),
        (200, message("Quais são suas despesas fixas mensais?")),
    ]
    if confirmation_exhausted:
        script[1] = script[0]
    if throttled:
        script.insert(3, (429, None))

    def sink(request):
        sent = json.loads(request.content)
        requests.append(sent)
        assert "tools" not in sent and sent["response_format"]["json_schema"]["strict"] is True
        status, value = script.pop(0)
        if status == 429:
            return httpx.Response(429, headers={"retry-after": "10.5"},
                                  json={"error": {"code": "rate_limit_exceeded", "message": "local throttle"}})
        wire = {"resposta": value}
        if status == 400:
            return httpx.Response(400, json={"error": {"type": "invalid_request_error",
                "code": "json_validate_failed", "param": "", "message": "private-provider-detail",
                "failed_generation": json.dumps(wire)}})
        Draft202012Validator(sent["response_format"]["json_schema"]["schema"]).validate(wire)
        return httpx.Response(200, json={"id": "local-schema-regression", "object": "chat.completion",
            "created": 0, "model": "openai/gpt-oss-120b", "choices": [{"index": 0,
            "finish_reason": "stop", "message": {"role": "assistant", "content": json.dumps(wire)}}]})

    with caplog.at_level(logging.INFO, logger="banco_agil"):
        with OpenAI(api_key="local-only", base_url="http://memory.invalid/v1", max_retries=0,
                    http_client=httpx.Client(transport=httpx.MockTransport(sink), trust_env=False)) as client:
            orch = Orchestrator(repo_ana_620, ExchangeService(), client, "openai/gpt-oss-120b", llm_mode="structured")
            session = logged_in(orch, agent="entrevista")
            FlowController.start_interview(session)
            session.flow.pending_interview_answers = {"renda_mensal": 90000}
            session.credit_proposal = dict(valor=8000, modalidade="limite_total", novo_limite=8000, confirmada=False)
            credit_before, data_before = copy.deepcopy(session.credit_proposal), csv(orch.repo)
            answer = orch.handle(session, "confirmo")
            if confirmation_exhausted:
                assert answer.startswith("Seu emprego é formal, autônomo ou está desempregado?")
                assert "Não consegui processar essa resposta com segurança" not in answer
                assert session.last_failure_cause == "json_validate_failed"
            else:
                assert answer.startswith("Qual é o seu tipo de emprego?")
            assert session.flow.interview_answers == {"renda_mensal": 90000}
            assert not session.flow.pending_interview_answers
            for sent in requests:
                assert names(sent) == set()
                assert "já foi processado localmente" in sent["messages"][1]["content"]
            assert "confirma" in orch.handle(session, "formal").lower()
            assert session.flow.pending_interview_answers == {"tipo_emprego": "formal"}
            assert session.flow.interview_answers == {"renda_mensal": 90000}
            assert COLLECT in names(requests[2])
            assert "required" in requests[3]["messages"][0]["content"]
            assert orch.handle(session, "sim").startswith("Quais são suas despesas fixas mensais?")
            assert session.flow.interview_answers == {"renda_mensal": 90000, "tipo_emprego": "formal"}
            assert not session.flow.pending_interview_answers
            assert session.credit_proposal == credit_before and csv(orch.repo) == data_before
            assert session.authenticated and session.active_agent == "entrevista" and not session.ended
            assert not session.last_failure_category and len(requests) == 6 + int(throttled) and not script
            assert clock.waits == ([20] if throttled else [])
    events = [json.loads(r.message) for r in caplog.records if r.message.startswith("{")]
    failed = [e for e in events if e["event"] == "llm_call" and e.get("result") == "retrying" and e["status_http"] == 400]
    assert len(failed) == 2 and all(e["attempt"] == 1 for e in failed)
    assert failed[1]["schema_issues"] == [f"$.resposta.acao.argumentos.{k}:required" for k in FIELDS if k != "tipo_emprego"]
    assert "private-provider-detail" not in caplog.text


@pytest.mark.parametrize("field", FIELDS)
def test_locally_rejected_partial_json_cannot_bypass_decoder_or_erase_confirmed_answers(repo_ana_620, field):
    orch = make(repo_ana_620, [proposal({field: FIELDS[field]})])
    session = logged_in(orch, agent="entrevista")
    FlowController.start_interview(session)
    session.flow.interview_answers = {"renda_mensal": 90000} if field != "renda_mensal" else {"tipo_emprego": "formal"}
    before, confirmed = csv(orch.repo), copy.deepcopy(session.flow.interview_answers)
    orch.handle(session, "Quero acrescentar uma informação à entrevista.")
    assert session.last_failure_category == "invalid_structured_response" and len(orch.client.requests) == 1
    assert session.flow.interview_answers == confirmed and not session.flow.pending_interview_answers
    assert session.authenticated and session.active_agent == "entrevista" and csv(orch.repo) == before


@pytest.mark.parametrize("parameters", [{"type": "object"},
    {"type": "object", "properties": None},
    {"type": "object", "properties": {"valor": {"type": "number"}}, "required": "valor"}])
def test_invalid_catalog_is_blocked_before_building_transport_guidance(repo_ana_620, monkeypatch, parameters):
    orch = make(repo_ana_620, [])
    session = logged_in(orch, agent="entrevista")
    FlowController.start_interview(session)
    bad = {"function": {"name": COLLECT, "parameters": parameters}}
    monkeypatch.setattr("banco_agil.llm._tool_schemas", lambda *_: [bad])
    state, before = copy.deepcopy(session.flow), csv(orch.repo)
    assert call_llm(orch.client, orch.model, AGENTS["entrevista"], session, llm_mode="structured") is None
    assert session.last_failure_cause == "invalid_request" and not orch.client.requests
    assert session.flow == state and csv(orch.repo) == before
