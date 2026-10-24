"""Diagnósticos de erro do provedor sem persistir as gerações privadas."""
import json
import logging
from types import SimpleNamespace

import pytest

from banco_agil.agents import AGENTS
from banco_agil.llm import _provider_diagnostics, call_llm
from banco_agil.session import Session
from structured_fakes import ScriptedClient, completion, message


@pytest.mark.parametrize('code',['json_validate_failed','tool_use_failed'])
@pytest.mark.parametrize('generation,valid',[('{"resposta":',False),('{"private":"secret"}',True)])
def test_safe_diagnostics(code,generation,valid):
    error=type('ProviderError',(Exception,),{'body':{'error':{
        'code':code,'message':'private provider text','failed_generation':generation}}})()
    assert _provider_diagnostics(error)==(code,valid)


@pytest.mark.parametrize('body', [None, 'private', {'error':'private'}, {'error':{'code':[]}},
    {'code':'secret','param':'secret'}, {'code':['unhashable'],'message':'secret'}])
def test_malformed_or_unknown_diagnostics_do_not_leak(body):
    code,valid=_provider_diagnostics(SimpleNamespace(body=body))
    assert code in {None,'other_provider_error'} and valid is None


def test_schema_rejection_classification():
    assert _provider_diagnostics(SimpleNamespace(body={'error':{
        'param':'response_format','message':'private','code':''}}))==('invalid_response_schema',None)


def test_http_400_logging_keeps_only_safe_metadata(caplog):
    error=type('ProviderError',(Exception,),{'status_code':400,'body':{
        'error':{'code':'json_validate_failed','message':'CPF 11144477735 private-token',
                 'failed_generation':'{"private":"11144477735"'}}})('private-token')
    client=ScriptedClient([error,error,completion(message('Podemos continuar.'))])
    session=Session()
    caplog.set_level(logging.INFO,logger='banco_agil')
    assert call_llm(client,'fake',AGENTS['triagem'],session,llm_mode='structured') is None
    assert len(client.requests)==2  # Uma única recuperação seletiva de geração.
    assert session.last_provider_error_code=='json_validate_failed'
    assert session.last_generation_json_valid is False
    events=[json.loads(r.message) for r in caplog.records if r.message.startswith('{')]
    assert events[-1]['provider_error_code']=='json_validate_failed'
    assert events[-1]['output_mode']=='structured'
    assert events[-1]['generation_json_valid'] is False
    assert '11144477735' not in caplog.text and 'private-token' not in caplog.text
    assert '11144477735' not in session.last_llm_response
    assert call_llm(client,'fake',AGENTS['triagem'],session,llm_mode='structured').content=='Podemos continuar.'
    assert session.last_provider_error_code is None and session.last_generation_json_valid is None


def test_reported_400_without_failed_generation_is_not_misdiagnosed(repo):
    """Formato do erro observado, sem geração rejeitada ou tokens informados."""
    import httpx2 as httpx
    from openai import OpenAI
    from banco_agil.orchestrator import Orchestrator
    from banco_agil.exchange import ExchangeService
    from banco_agil.rate_limit import RateLimiter, GovernedClient
    from fakes import Clock
    from support import snapshot_csv as csv
    payload={
        'request_id':'req_01m5jf49ejemrb5yk1kb542cws',
        'created_at':'2026-10-22T18:49:13.426Z',
        'error':{'message':"Failed to validate JSON. Please adjust your prompt. See 'failed_generation' for more details.",
                 'type':'invalid_request_error','param':'','code':'json_validate_failed'},
    }
    clock=Clock()
    bucket=RateLimiter(30,8000,clock=clock,sleep=clock.sleep)
    requests=[]
    def sink(request):
        requests.append(json.loads(request.content))
        return httpx.Response(400,json=payload)
    before=csv(repo)
    with OpenAI(api_key='local',base_url='http://memory.invalid/v1',max_retries=0,
                http_client=httpx.Client(transport=httpx.MockTransport(sink))) as sdk:
        orch=Orchestrator(repo,ExchangeService(),GovernedClient(sdk,bucket),'openai/gpt-oss-20b',
                          llm_mode='structured',max_llm_retries=1)
        session=orch.new_session()
        orch.handle(session,'Quais serviços existem?')
    assert len(requests)==2 and not clock.waits
    assert requests[0]['response_format']['json_schema']['strict'] is True
    assert requests[0]['max_completion_tokens']==2048 and 'tools' not in requests[0]
    assert session.last_failure_category=='http_400'
    assert session.last_provider_error_code=='json_validate_failed'
    assert session.last_generation_json_valid is None  # Não há dado suficiente para afirmar que a geração foi truncada.
    assert session.last_provider_diagnostic['generation_status']=='not_provided'
    assert len(session.last_provider_diagnostic['schema_sha256'])==64
    assert session.turn==1 and not session.ended and csv(repo)==before


@pytest.mark.parametrize('value,expected',[
    ({'resposta':{'tipo':'acao','mensagem':None,'acao':{'nome':'consultar_limite','argumentos':None}}}, []),
    ({'resposta':{'tipo':'acao','mensagem':'private-token','acao':{'nome':'consultar_limite','argumentos':None}}}, ['$.resposta.mensagem:type']),
    ({'resposta':{'tipo':'acao','mensagem':None,'acao':{'nome':'consultar_limite','argumentos':{}}}}, ['$.resposta.acao.argumentos:type']),
    ({'resposta':{'tipo':'mensagem','acao':None}}, ['$.resposta.mensagem:required']),
    ({'resposta':{'tipo':'mensagem','mensagem':'private-token','acao':None,'11144477735':'secret'}}, ['$.resposta:additional_properties']),
])
def test_generation_diagnostic_uses_matching_branch_without_leaking(value,expected):
    from banco_agil.provider_diagnostics import generation_report
    from banco_agil.structured_output import response_format
    tool=AGENTS['credito'].get_tool('consultar_limite').schema()
    fmt=response_format([tool])
    error=SimpleNamespace(body={'error':{'failed_generation':json.dumps(value)}})
    report=generation_report(error,fmt)
    assert report['schema_issues']==expected
    assert report['generation_status']==('schema_mismatch' if expected else 'schema_valid')
    assert len(report['schema_sha256'])==64
    assert 'private-token' not in json.dumps(report) and '11144477735' not in json.dumps(report)


@pytest.mark.parametrize('generation,status',[
    ('{"resposta":','invalid_json'),
    ('{"resposta":null,"resposta":null}','duplicate_key'),
    ('{"resposta":NaN}','nonfinite_number'),
    ('a'*100001,'size_limit'),
    ({'secret':'private'},'unsupported_format'),
    (None,'not_provided'),
])
def test_absence_and_malformed_generation_are_explicit(generation,status):
    from banco_agil.provider_diagnostics import generation_report
    from banco_agil.structured_output import response_format
    error=SimpleNamespace(body={'error':{'failed_generation':generation}})
    report=generation_report(error,response_format([]))
    assert report['generation_status']==status and not report['schema_issues']


def test_reporter_agrees_with_transmitted_catalog_schema():
    from banco_agil.provider_diagnostics import generation_report
    from banco_agil.structured_output import response_format
    from structured_fakes import CATALOG, sample
    from jsonschema import Draft202012Validator
    fmt=response_format([t.schema() for t in CATALOG.values()])
    validator=Draft202012Validator(fmt['json_schema']['schema'])
    for branch in fmt['json_schema']['schema']['properties']['resposta']['anyOf'][1]['properties']['acao']['anyOf']:
        value={'resposta':{'tipo':'acao','mensagem':None,'acao':sample(branch)}}
        report=generation_report(SimpleNamespace(body={'failed_generation':json.dumps(value)}),fmt)
        assert validator.is_valid(value) and report['generation_status']=='schema_valid'
        value['resposta']['mensagem']='private-token'
        report=generation_report(SimpleNamespace(body={'error':{'failed_generation':json.dumps(value)}}),fmt)
        assert not validator.is_valid(value) and report['generation_status']=='schema_mismatch'


def test_http_diagnostic_paths_reach_logs_without_generation(caplog):
    error=type('ProviderError',(Exception,),{'status_code':400,'body':{'error':{
        'code':'json_validate_failed','failed_generation':json.dumps({'resposta':{
            'tipo':'acao','mensagem':'private-token','acao':{'nome':'encerrar_atendimento','argumentos':None}}})}}})()
    client=ScriptedClient([error,error])
    session=Session()
    caplog.set_level(logging.INFO,logger='banco_agil')
    assert call_llm(client,'fake',AGENTS['triagem'],session,llm_mode='structured') is None
    assert session.last_provider_diagnostic['generation_status']=='schema_mismatch'
    assert session.last_provider_diagnostic['schema_issues']==['$.resposta.mensagem:type']
    assert 'private-token' not in caplog.text
    assert 'failed_generation' not in caplog.text
    assert 'schema_mismatch' in caplog.text
    assert len(client.requests)==2
