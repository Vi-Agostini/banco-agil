"""A recuperação seletiva mantém o schema, os contratos e as condições financeiras."""
import copy
import json
import logging

import httpx2 as httpx
from openai import OpenAI
import pytest

from banco_agil.agents import AGENTS
from banco_agil.llm import LlmRetryState, call_llm
from banco_agil.messages import LLM_UNAVAILABLE
from banco_agil.rate_limit import GovernedClient, RateLimiter
from banco_agil.session import Session
from fakes import Clock, failure
from structured_fakes import make, action, message, completion, ScriptedClient, strict_spec
from support import snapshot_csv as csv
from orchestrator_fakes import logged_in



@pytest.mark.parametrize('mode',['native','structured'])
@pytest.mark.parametrize('code',['tool_use_failed','json_validate_failed'])
def test_safe_http_generation_recovery(mode,code,caplog):
    clock=Clock()
    bucket=RateLimiter(30,30000,clock=clock,sleep=clock.sleep)
    sent=[]
    def sink(request):
        sent.append(json.loads(request.content))
        if len(sent)==1:
            return httpx.Response(400,json={'error':{'code':code,'param':'','message':'private-token'}})
        content='Podemos continuar.' if mode=='native' else json.dumps({'resposta':message('Podemos continuar.')})
        answer, finish = {'role':'assistant','content':content}, 'stop'
        if sent[-1].get('tool_choice',{}).get('function',{}).get('name')=='banco_agil_resposta':
            answer={'role':'assistant','content':None,'tool_calls':[{'id':'local-envelope','type':'function',
                'function':{'name':'banco_agil_resposta','arguments':content}}]}
            finish='tool_calls'
        return httpx.Response(200,json={'id':'local','object':'chat.completion','created':0,'model':'fake',
            'choices':[{'index':0,'finish_reason':finish,'message':answer}]})
    session=Session(history=[{'role':'user','content':'Podemos conversar?'}])
    before=copy.deepcopy(session.history)
    with caplog.at_level(logging.INFO,logger='banco_agil'):
        with OpenAI(api_key='local',base_url='http://memory.invalid/v1',max_retries=0,
                    http_client=httpx.Client(transport=httpx.MockTransport(sink))) as sdk:
            result=call_llm(GovernedClient(sdk,bucket),'openai/gpt-oss-20b',AGENTS['triagem'],session,llm_mode=mode)
    assert result.content=='Podemos continuar.' and len(sent)==2 and not clock.waits
    assert session.history==before and not session.last_failure_category
    assert sent[1]['messages'][1:]==sent[0]['messages'][1:]
    if mode=='structured':
        assert strict_spec(sent[0])==strict_spec(sent[1])
        assert ('tools' in sent[1]) is (code=='tool_use_failed')
    else:
        assert sent[0]['tools']==sent[1]['tools']
    events=[json.loads(r.message) for r in caplog.records if r.message.startswith('{')]
    assert [e['result'] for e in events]==['retrying','success']
    assert events[0]['failure_cause']==code and events[1]['recovery_cause']==code
    assert all(e['model']=='openai/gpt-oss-20b' for e in events)
    assert [e['attempt'] for e in events]==[1,2] and 'private-token' not in caplog.text


@pytest.mark.parametrize('error',[
    failure('',param='response_format',message='invalid JSON schema'),
    failure('json_validate_failed',param='response_format'),
    failure('json_validate_failed',message='invalid JSON schema for response_format'),
    failure('tool_use_failed',param='messages'),
    failure('unsupported_parameter'), failure('unknown'),
])
def test_invalid_request_never_retries_identically(error):
    client=ScriptedClient([error])
    session=Session()
    assert call_llm(client,'fake',AGENTS['triagem'],session,llm_mode='structured') is None
    assert len(client.requests)==1 and session.last_failure_cause=='invalid_request'


@pytest.mark.parametrize('second',['same','mixed'])
def test_generation_budget_shared_across_classes(second):
    client=ScriptedClient([failure('json_validate_failed'),failure('json_validate_failed' if second=='same' else 'tool_use_failed')])
    session=Session()
    assert call_llm(client,'fake',AGENTS['triagem'],session,llm_mode='structured') is None
    assert len(client.requests)==2 and session.last_failure_cause in {'tool_use_failed','json_validate_failed'}


def test_budget_shared_across_pipeline_steps_and_agents():
    state=LlmRetryState()
    first=ScriptedClient([failure('json_validate_failed'),completion(message('Continuamos.'))])
    session=Session()
    assert call_llm(first,'fake',AGENTS['triagem'],session,llm_mode='structured',retry_state=state)
    other=ScriptedClient([failure('tool_use_failed')])
    assert call_llm(other,'fake',AGENTS['credito'],session,llm_mode='structured',retry_state=state) is None
    assert len(other.requests)==1


@pytest.mark.parametrize('code',['tool_use_failed','json_validate_failed'])
def test_exhaustion_has_friendly_reply_and_zero_financial_effects(repo_ana_620,code):
    orch=make(repo_ana_620,[failure(code),failure(code)])
    session=logged_in(orch,agent='credito')
    before=csv(orch.repo)
    answer=orch.handle(session,'Quero conversar sobre meu crédito.')
    assert answer==LLM_UNAVAILABLE
    assert not any(word in answer.lower() for word in ['http','json','schema'])
    assert len(orch.client.requests)==2 and csv(orch.repo)==before
    assert not orch.repo.list_requests() and session.authenticated and not session.ended


@pytest.mark.parametrize('code',['tool_use_failed','json_validate_failed'])
@pytest.mark.parametrize('mode',['native','structured'])
def test_recovered_generation_cannot_bypass_financial_confirmation(repo_ana_620,code,mode):
    orch=make(repo_ana_620,[failure(code),action('solicitar_aumento_limite',novo_limite=8000)],mode)
    session=logged_in(orch,agent='credito')
    session.credit_proposal=dict(valor=8000,modalidade='limite_total',novo_limite=8000,confirmada=False)
    before,proposal=csv(orch.repo),copy.deepcopy(session.credit_proposal)
    orch.handle(session,'Ainda quero conferir os dados.')
    assert session.last_failure_cause==('invalid_structured_response' if mode=='structured' else 'tool_contract_violation')
    assert csv(orch.repo)==before and session.credit_proposal==proposal and not orch.repo.list_requests()
    assert len(orch.client.requests)==2


@pytest.mark.parametrize('stage',['before','after'])
def test_completed_operation_is_never_replayed(repo_ana_620,stage):
    script=[failure('json_validate_failed'),action('solicitar_aumento_limite',novo_limite=8000),message('Seu limite atual é R$ 8.000,00.')] if stage=='before' else [
        action('solicitar_aumento_limite',novo_limite=8000),failure('json_validate_failed'),message('Seu limite atual é R$ 8.000,00.')]
    orch=make(repo_ana_620,script)
    session=logged_in(orch,agent='credito')
    session.credit_proposal=dict(valor=8000,modalidade='limite_total',novo_limite=8000,confirmada=True)
    answer=orch.handle(session,'sim')
    assert '8.000,00' in answer and len(orch.client.requests)==3
    assert len(orch.repo.list_requests())==1 and orch.repo.get_client(session.cpf)['limite_credito']==8000


@pytest.mark.parametrize('bad',[
    'not json', json.dumps({'tipo':'mensagem','mensagem':'flat','acao':None}),
    {'tipo':'acao','mensagem':'hybrid','acao':{'nome':'consultar_limite','argumentos':None}},
])
def test_local_invalid_structured_response_stays_blocking(repo_ana_620,bad):
    orch=make(repo_ana_620,[completion(bad)])
    session=logged_in(orch,agent='credito')
    before=csv(orch.repo)
    orch.handle(session,'Podemos conversar?')
    assert session.last_failure_cause=='invalid_structured_response'
    assert len(orch.client.requests)==1 and csv(orch.repo)==before


def test_recovery_still_requires_rate_limit_admission():
    clock=Clock()
    bucket=RateLimiter(1,30000,clock=clock,sleep=clock.sleep)
    sent=[]
    def sink(request):
        sent.append(json.loads(request.content))
        bucket.defer(20)  # Representa um 429 já observado, sem prever consumo de RPM.
        return httpx.Response(400,json={'error':{'code':'json_validate_failed'}})
    with OpenAI(api_key='local',base_url='http://memory.invalid/v1',max_retries=0,
                http_client=httpx.Client(transport=httpx.MockTransport(sink))) as sdk:
        session=Session()
        assert call_llm(GovernedClient(sdk,bucket,max_wait=10),'fake',AGENTS['triagem'],session,llm_mode='structured') is None
    assert len(sent)==1 and not clock.waits
    assert session.last_failure_cause=='local_capacity'
    assert session.last_provider_diagnostic=={}  # Uma falha local não reaproveita o diagnóstico anterior do provedor.


def test_corrective_prompt_cannot_bypass_input_budget(monkeypatch):
    from banco_agil import llm
    estimates=iter([100,llm.MAX_INPUT_TOKENS+1])
    monkeypatch.setattr(llm,'_estimate_request_tokens',lambda *_, **__: next(estimates))
    client=ScriptedClient([failure('json_validate_failed')])
    session=Session()
    assert call_llm(client,'fake',AGENTS['triagem'],session,llm_mode='structured') is None
    assert len(client.requests)==1 and session.last_failure_cause=='input_too_long'


def test_invalid_recovered_response_is_never_executed(repo_ana_620):
    bad={'tipo':'acao','mensagem':'Não confirmada','acao':{'nome':'solicitar_aumento_limite','argumentos':{'novo_limite':8000}}}
    orch=make(repo_ana_620,[failure('json_validate_failed'),completion(bad)])
    session=logged_in(orch,agent='credito')
    before=csv(orch.repo)
    orch.handle(session,'Quero avaliar minhas opções.')
    assert session.last_failure_cause=='invalid_structured_response'
    assert len(orch.client.requests)==2 and not orch.repo.list_requests() and csv(orch.repo)==before
