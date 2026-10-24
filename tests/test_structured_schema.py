"""Objetos vazios são rejeitados no schema; o SDK usa HTTP em memória."""
import copy
import json
from types import SimpleNamespace as NS

import httpx2 as httpx
from openai import OpenAI
import pytest
from jsonschema import Draft202012Validator

from banco_agil import llm
from banco_agil.agents import AGENTS
from banco_agil.flow import FlowController
from banco_agil.session import Session
from banco_agil.structured_output import StructuredOutputError, decode, response_format, validate_format
from banco_agil.tool_contracts import parse_arguments
from orchestrator_fakes import logged_in
from support import snapshot_csv as csv
from structured_fakes import make, CATALOG, sample



def action_branches(fmt):
    return fmt["json_schema"]["schema"]["properties"]["resposta"]["anyOf"][1]["properties"]["acao"]["anyOf"]


def assert_exclusive_schema(fmt):
    schema=fmt["json_schema"]["schema"]
    Draft202012Validator.check_schema(schema)
    validator=Draft202012Validator(schema)
    assert schema["type"]=="object" and "anyOf" not in schema
    assert validator.is_valid({"resposta":{"tipo":"mensagem","mensagem":"Como posso ajudar?","acao":None}})
    for branch in action_branches(fmt):
        action=sample(branch)
        assert validator.is_valid({"resposta":{"tipo":"acao","mensagem":None,"acao":action}})
        for kind in ["mensagem","acao"]:
            assert not validator.is_valid({"resposta":{"tipo":kind,"mensagem":"texto","acao":action}})


def offered_states():
    for name in AGENTS:
        s = Session(active_agent=name, authenticated=True, cpf="local-schema-fixture")
        if name == "entrevista":
            FlowController.start_interview(s)
        yield name, s
    yield "triagem_sem_login", Session()
    s = Session(active_agent="credito", authenticated=True, cpf="local-schema-fixture")
    s.credit_proposal = dict(valor=8000, modalidade="limite_total", novo_limite=8000, confirmada=False)
    yield "credito_pendente", copy.deepcopy(s)
    s.credit_proposal["confirmada"] = True
    yield "credito_confirmado", s
    s = Session(active_agent="entrevista", authenticated=True, cpf="local-schema-fixture")
    FlowController.start_interview(s)
    s.flow.pending_interview_answers = {"renda_mensal": 6000}
    yield "entrevista_pendente", copy.deepcopy(s)
    s.flow.pending_interview_answers = {}
    s.flow.interview_answers = dict(renda_mensal=6000, tipo_emprego="formal", despesas_fixas=2000,
                                   num_dependentes=1, tem_dividas=False)
    yield "entrevista_completa", s


def test_original_fifth_branch_and_other_zero_argument_tools_use_null():
    s = Session(active_agent="entrevista", authenticated=True, cpf="fixture")
    FlowController.start_interview(s)
    fmt = response_format(llm._tool_schemas(AGENTS["entrevista"], s))
    branch = action_branches(fmt)[3]
    assert branch["properties"]["nome"]["enum"] == ["encerrar_atendimento"]
    assert branch["properties"]["argumentos"] == {"type": "null"}
    before = copy.deepcopy(fmt)
    branch["properties"]["argumentos"] = {
        "type": "object", "properties": {}, "required": [], "additionalProperties": False}
    with pytest.raises(StructuredOutputError, match="schema_objeto_vazio"):
        validate_format(fmt)
    del branch["properties"]["argumentos"]["properties"]
    with pytest.raises(StructuredOutputError, match="schema_properties_ausente"):
        validate_format(fmt)
    validate_format(before)


@pytest.mark.parametrize("name", sorted(name for name, tool in CATALOG.items() if not tool.parameters["properties"]))
def test_wire_null_maps_only_to_the_existing_zero_argument_contract(name):
    tool = CATALOG[name]
    tools = [tool.schema()]
    fmt = response_format(tools)
    assert action_branches(fmt)[0]["properties"]["argumentos"] == {"type": "null"}
    envelope = {"tipo": "acao", "mensagem": None, "acao": {"nome": name, "argumentos": None}}
    result = decode(NS(content=json.dumps({"resposta": envelope})), tools, "stop")
    assert parse_arguments(tool, result.tool_calls[0].function.arguments) == {}
    envelope["acao"]["argumentos"] = {}
    with pytest.raises(StructuredOutputError):
        decode(NS(content=json.dumps({"resposta": envelope})), tools, "stop")


def test_entire_catalog_and_actual_sdk_wire_have_no_malformed_object_or_removed_action(tmp_path):
    # Audita o catálogo completo; o request real continua usando apenas
    # as ferramentas permitidas pelo estado e pelas autorizações.
    whole = response_format([tool.schema() for tool in CATALOG.values()])
    assert_exclusive_schema(whole)
    names = {b["properties"]["nome"]["enum"][0] for b in action_branches(whole)}
    assert names == set(CATALOG)
    expected, captured = [], []

    def sink(request):
        body = json.loads(request.content)
        validate_format(body["response_format"])
        assert_exclusive_schema(body["response_format"])
        assert not {"tools", "tool_choice", "parallel_tool_calls"}.intersection(body)
        captured.append(body)
        return httpx.Response(200, request=request, json={"id": "local-schema-audit", "object": "chat.completion",
            "created": 0, "model": "openai/gpt-oss-20b", "choices": [{"index": 0, "finish_reason": "stop",
            "message": {"role": "assistant", "content": "{}"}}]})

    with OpenAI(api_key="local-test-only", base_url="http://memory.invalid/v1", max_retries=0,
                http_client=httpx.Client(transport=httpx.MockTransport(sink), trust_env=False)) as client:
        for label, session in offered_states():
            fmt = response_format(llm._tool_schemas(AGENTS[session.active_agent], session))
            expected.append(fmt)
            client.chat.completions.create(model="openai/gpt-oss-20b", messages=[{"role": "user", "content": "local audit"}], response_format=fmt)
            (tmp_path/(label+".json")).write_text(json.dumps(captured[-1], ensure_ascii=False, indent=2))
    assert len(captured) == 9 and [r["response_format"] for r in captured] == expected
    assert all("autenticar_cliente" not in json.dumps(r) for r in captured)


@pytest.mark.parametrize("problem", ["properties_missing", "required_missing", "required_unknown", "required_duplicate",
    "required_partial", "additional_true", "additional_missing", "unsupported_keyword", "invalid_anyof", "nested_object"])
def test_recursive_audit_rejects_malformed_arguments_in_every_branch(problem):
    fmt = response_format([CATALOG["registrar_intencao_credito"].schema()])
    args = action_branches(fmt)[0]["properties"]["argumentos"]
    if problem == "properties_missing": del args["properties"]
    elif problem == "required_missing": del args["required"]
    elif problem == "required_unknown": args["required"].append("inventado")
    elif problem == "required_duplicate": args["required"].append("valor")
    elif problem == "required_partial": args["required"] = ["valor"]
    elif problem == "additional_true": args["additionalProperties"] = True
    elif problem == "additional_missing": del args["additionalProperties"]
    elif problem == "unsupported_keyword": args["allOf"] = []
    elif problem == "invalid_anyof": args["properties"]["valor"] = {"anyOf": []}
    else: args["properties"]["valor"] = {"type": "object", "required": [], "additionalProperties": False}
    with pytest.raises(StructuredOutputError): validate_format(fmt)


def test_malformed_catalog_stops_before_sdk_and_preserves_session_and_finances(repo_ana_620, monkeypatch):
    orch = make(repo_ana_620, [])
    session = logged_in(orch, agent="credito")
    before = csv(orch.repo)
    bad = CATALOG["registrar_intencao_credito"].schema()
    bad["function"]["parameters"] = {"type": "object", "properties": {"valor": {"type": "number", "allOf": []}},
        "required": ["valor"], "additionalProperties": False}
    monkeypatch.setattr(llm, "_tool_schemas", lambda *_: [bad])
    orch.handle(session, "Quais alternativas de limite existem?")
    assert not orch.client.requests and session.last_failure_category == "invalid_request"
    assert session.authenticated and session.auth_failures == 0 and csv(orch.repo) == before


@pytest.mark.parametrize("args", [None, {}, {"novo_limite": None}])
def test_null_or_missing_financial_arguments_never_execute(repo_ana_620, monkeypatch, args):
    value = {"tipo": "acao", "mensagem": None, "acao": {"nome": "solicitar_aumento_limite", "argumentos": args}}
    orch = make(repo_ana_620, [value])
    session = logged_in(orch, agent="credito")
    session.credit_proposal = dict(valor=8000, modalidade="limite_total", novo_limite=8000, confirmada=True)
    before, proposal = csv(orch.repo), copy.deepcopy(session.credit_proposal)
    monkeypatch.setattr(orch.tool_executor, "execute_from_llm", lambda *_: pytest.fail("Financial executor must not run"))
    orch.handle(session, "Podemos conversar sobre a proposta?")
    assert session.last_failure_category in {"invalid_structured_response", "invalid_tool_response"}
    assert csv(orch.repo) == before and session.credit_proposal == proposal
    assert session.authenticated and session.auth_failures == 0


@pytest.mark.parametrize("kind,error",[("mensagem","mensagem_invalida"),("acao","acao_invalida")])
@pytest.mark.parametrize("pending",[False,True])
def test_hybrid_response_never_executes_or_changes_financial_consent(repo_ana_620,monkeypatch,kind,error,pending):
    value={"tipo":kind,"mensagem":"Você deseja solicitar R$ 8.000,00? Por favor, confirme.",
           "acao":{"nome":"registrar_intencao_credito","argumentos":{"valor":8000,"modalidade":"limite_total"}}}
    orch=make(repo_ana_620,[value]);session=logged_in(orch,agent="credito")
    if pending:session.credit_proposal=dict(valor=8000,modalidade="limite_total",novo_limite=8000,confirmada=False)
    before,proposal=csv(orch.repo),copy.deepcopy(session.credit_proposal)
    monkeypatch.setattr(orch.tool_executor,"execute_from_llm",lambda *_:pytest.fail("Hybrid cannot reach executor"))
    orch.handle(session,"Meu objetivo é um limite total de R$ 8.000,00. Ainda não autorizo a execução; primeiro quero conferir a proposta.")
    assert session.last_failure_category=="invalid_structured_response" and session.last_tool_error_code==error
    assert session.credit_proposal==proposal and csv(orch.repo)==before
    assert session.authenticated and session.auth_failures==0 and len(orch.client.requests)==1


def test_new_wrapper_does_not_accept_or_normalize_legacy_flat_output():
    value={"tipo":"mensagem","mensagem":"Podemos conversar.","acao":None}
    with pytest.raises(StructuredOutputError,match="envelope_invalido"):
        decode(NS(content=json.dumps(value)),[],"stop")
    valid={"resposta":value}
    result=decode(NS(content=json.dumps(valid)),[],"stop")
    assert result.content==value["mensagem"] and result.envelope==valid and not result.tool_calls
