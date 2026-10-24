"""Contabilização e requisição real do SDK, sem rede ou espera no relógio real."""
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from uuid import uuid4
import json

import httpx2 as httpx
from openai import OpenAI
import pytest

from banco_agil.rate_limit import CapacityError, RateLimiter, GovernedClient, shared_limiter, duration
from banco_agil.llm import call_llm, LlmRetryState
from banco_agil.agents import AGENTS
from banco_agil.session import Session
from structured_fakes import message

from fakes import Clock


def limiter(rpm=2, tpm=100, **kwargs):
    clock = Clock()
    return RateLimiter(rpm, tpm, clock=clock, sleep=clock.sleep, **kwargs), clock


@pytest.mark.parametrize('rpm,tpm,tokens', [(1,100,10), (10,100,80)])
def test_usage_thresholds_do_not_cause_preventive_waits(rpm,tpm,tokens):
    bucket, clock = limiter(rpm,tpm)
    first = bucket.acquire(tokens, 0)
    bucket.settle(first)
    second = bucket.acquire(tokens, 0)
    assert second.at == 0 and clock.waits == []
    assert sum(e.tokens for e in bucket.entries) == 2*tokens
    bucket.settle(second)
    clock.now = 60
    bucket.acquire(tokens, 0)  # Um prazo vencido não impede um envio que já tem capacidade disponível.
    assert len(bucket.entries) == 1 and not clock.waits


def test_pending_and_unknown_usage_remain_accounted_without_waiting():
    bucket, clock = limiter(1)
    first = bucket.acquire(80,0)
    clock.now = 100
    bucket.acquire(30,100)
    assert first in bucket.entries and sum(e.tokens for e in bucket.entries) == 110
    bucket.settle(first)
    assert first.tokens == 80 and first.at == 100 and not clock.waits


def test_confirmed_usage_and_oversized_request():
    bucket, clock = limiter(10)
    entry = bucket.acquire(90,0)
    bucket.settle(entry, {'total_tokens':20})
    bucket.acquire(80,0)
    with pytest.raises(CapacityError, match='exceeds'):
        bucket.acquire(101,120)
    assert clock.waits == []


def test_atomic_concurrent_admission():
    bucket, clock = limiter(3,100)
    barrier = Barrier(10)
    def attempt(_):
        barrier.wait()
        try:
            return bucket.acquire(30,0)
        except CapacityError:
            return None
    with ThreadPoolExecutor(max_workers=10) as pool:
        admitted = [v for v in pool.map(attempt, range(10)) if v is not None]
    assert len(admitted) == 10 and sum(e.tokens for e in bucket.entries) == 300
    assert clock.waits == []


def test_shared_scope_and_config_cannot_enlarge_bucket():
    org = uuid4().hex
    first = shared_limiter(org,'model',10,1000)
    assert first is shared_limiter(org,'model',20,2000)
    assert (first.rpm,first.tpm) == (10,1000)
    assert first is not shared_limiter(org,'other-model',10,1000)
    assert first is not shared_limiter(org+'other','model',10,1000)


def test_headers_are_additional_caps_and_never_restore_capacity():
    bucket, clock = limiter(10,100)
    one = bucket.acquire(10,0)
    pending = bucket.acquire(30,0)
    bucket.settle(one, {'total_tokens':10}, {
        'x-ratelimit-limit-tokens':'90', 'x-ratelimit-remaining-tokens':'50',
        'x-ratelimit-reset-tokens':'2s', 'x-ratelimit-remaining-requests':'0',
        'x-ratelimit-reset-requests':'1m'})
    assert bucket.remote['tokens'][0] == 20 and bucket.tpm == 90
    bucket.settle(pending, headers={'x-ratelimit-remaining-tokens':'80','x-ratelimit-reset-tokens':'1s'})
    assert bucket.remote['tokens'] == [20,2]
    bucket.acquire(1,0)
    assert clock.waits == []  # Cabeçalhos de sucesso não abrem cooldown.


@pytest.mark.parametrize('value,expected',[('2m59.56s',179.56),('100ms',.1),('nan',None),('garbage',None)])
def test_header_durations(value,expected):
    assert duration(value) == expected


@pytest.mark.parametrize('mode',['native','structured'])
def test_sdk_headers_usage_and_single_cooldown(mode):
    bucket, clock = limiter(30,8000)
    sent=[]
    def sink(request):
        sent.append(json.loads(request.content))
        if len(sent)==1:
            return httpx.Response(429, headers={'retry-after':'25',
                'x-ratelimit-remaining-tokens':'0','x-ratelimit-reset-tokens':'25s'},
                json={'error':{'message':'busy'}})
        content = 'Tudo bem.' if mode=='native' else json.dumps({'resposta':message('Tudo bem.')})
        return httpx.Response(200, headers={'x-ratelimit-remaining-tokens':'4000','x-ratelimit-reset-tokens':'5s'},
            json={'id':'local','object':'chat.completion','created':0,'model':'fake',
            'usage':{'prompt_tokens':10,'completion_tokens':5,'total_tokens':15},
            'choices':[{'index':0,'finish_reason':'stop','message':{'role':'assistant','content':content}}]})
    with OpenAI(api_key='local',base_url='http://memory.invalid/v1',max_retries=0,
                http_client=httpx.Client(transport=httpx.MockTransport(sink))) as sdk:
        client=GovernedClient(sdk,bucket)
        result=call_llm(client,'fake',AGENTS['triagem'],Session(),llm_mode=mode)
    assert result.content == 'Tudo bem.'
    assert len(sent)==2 and sent[0]==sent[1]
    assert sent[0]['max_completion_tokens']==2048
    assert clock.waits==[25]  # A espera do cabeçalho e o cooldown se sobrepõem, sem somar os prazos.
    assert bucket.entries[-1].tokens==15
    if mode=='structured':
        assert 'tools' not in sent[0] and sent[0]['response_format']['json_schema']['strict']


def test_blocked_admission_does_not_send_and_preserves_session():
    bucket, clock = limiter(1,8000)
    bucket.defer(61)
    session=Session(authenticated=True,cpf='fixture')
    class Never:
        @property
        def chat(self):
            pytest.fail('No HTTP permitted')
    result=call_llm(GovernedClient(Never(),bucket), 'fake', AGENTS['credito'],session)
    assert result is None and session.authenticated and session.turn==0
    assert session.last_failure_category=='local_capacity'
    assert clock.waits==[]


def test_deadline_shared_across_steps():
    bucket, clock=limiter(30,8000)
    state=LlmRetryState(admission_deadline=10)
    bucket.defer(11)
    result=call_llm(GovernedClient(object(),bucket), 'fake',AGENTS['triagem'],Session(),retry_state=state)
    assert result is None and clock.waits==[]


def test_shared_cooldown_blocks_other_session_without_a_retry():
    bucket, clock=limiter(30,8000)
    bucket.defer(25)
    bucket.acquire(100,60)
    assert clock.waits==[25]
    bucket.acquire(100,60)
    assert clock.waits==[25]


def test_governed_financial_continuation_is_not_reexecuted(repo_ana_620):
    from banco_agil.orchestrator import Orchestrator
    from banco_agil.exchange import ExchangeService
    from orchestrator_fakes import logged_in
    bucket,clock=limiter(30,30000)
    sent=[]
    def sink(request):
        sent.append(json.loads(request.content))
        if len(sent)==2:
            return httpx.Response(429,headers={'retry-after':'20'},json={'error':{'message':'busy'}})
        content = {'resposta': {'tipo':'acao','mensagem':None,'acao':{'nome':'solicitar_aumento_limite',
            'argumentos':{'novo_limite':8000}}}} if len(sent)==1 else {'resposta':message('Seu limite atual é R$ 8.000,00.')}
        return httpx.Response(200,json={'id':'local','object':'chat.completion','created':0,'model':'fake',
            'usage':{'prompt_tokens':100,'completion_tokens':50,'total_tokens':150},
            'choices':[{'index':0,'finish_reason':'stop','message':{'role':'assistant','content':json.dumps(content)}}]})
    with OpenAI(api_key='local',base_url='http://memory.invalid/v1',max_retries=0,
                http_client=httpx.Client(transport=httpx.MockTransport(sink))) as sdk:
        orch=Orchestrator(repo_ana_620,ExchangeService(),GovernedClient(sdk,bucket),'fake',llm_mode='structured')
        session=logged_in(orch,agent='credito')
        session.credit_proposal=dict(valor=8000,modalidade='limite_total',novo_limite=8000,confirmada=True)
        answer=orch.handle(session,'sim')
    assert '8.000,00' in answer and clock.waits==[20]
    assert len(sent)==3 and sent[1]==sent[2]
    assert len(orch.repo.list_requests())==1
    assert orch.repo.get_client(session.cpf)['limite_credito']==8000


@pytest.mark.parametrize('config', ['[]','{"org":{"model":{"rpm":0,"tpm":8000}}}', '{broken'])
def test_bad_configuration_fails_closed(config,monkeypatch):
    from banco_agil import bootstrap
    monkeypatch.setattr(bootstrap,'load_dotenv',lambda *_: None)
    monkeypatch.setattr(bootstrap,'setup_logging',lambda *_: None)
    monkeypatch.setattr(bootstrap,'OpenAI',lambda **_: object())
    monkeypatch.setenv('LLM_API_KEY','local')
    monkeypatch.setenv('LLM_BASE_URL','https://api.groq.com/openai/v1')
    monkeypatch.setenv('GROQ_ORGANIZATION','org')
    monkeypatch.setenv('LLM_MODEL','model')
    monkeypatch.setenv('GROQ_RATE_LIMITS_JSON',config)
    with pytest.raises(bootstrap.ConfigError,match='limitador'):
        bootstrap.build_orchestrator()


def test_bootstrap_selects_organization_model_config(repo,monkeypatch):
    from types import SimpleNamespace
    from banco_agil import bootstrap
    captured={}
    monkeypatch.setattr(bootstrap,'load_dotenv',lambda *_: None)
    monkeypatch.setattr(bootstrap,'setup_logging',lambda *_: None)
    def sdk(**kwargs):
        captured['sdk']=kwargs
        return object()
    def orchestrator(_repo,_fx,client,model,**kwargs):
        captured.update(client=client,model=model)
        return SimpleNamespace(credit=SimpleNamespace(reconcile=lambda:None))
    monkeypatch.setattr(bootstrap,'OpenAI',sdk)
    monkeypatch.setattr(bootstrap,'Orchestrator',orchestrator)
    monkeypatch.setenv('LLM_API_KEY','local')
    monkeypatch.setenv('LLM_BASE_URL','https://api.groq.com/openai/v1')
    monkeypatch.setenv('DATA_DIR',str(repo.dir))
    organization=uuid4().hex
    monkeypatch.setenv('GROQ_ORGANIZATION',organization)
    monkeypatch.setenv('LLM_MODEL','configured-model')
    monkeypatch.setenv('GROQ_RATE_LIMITS_JSON',json.dumps({organization:{'configured-model':{'rpm':4,'tpm':9000,'output_tokens':3072}}}))
    monkeypatch.setenv('GROQ_ADMISSION_TIMEOUT_SECONDS','35')
    bootstrap.build_orchestrator()
    client=captured['client']
    assert isinstance(client,GovernedClient) and client.max_wait==35
    assert client.output_tokens==3072
    assert (client.limiter.rpm,client.limiter.tpm)==(4,9000)
    assert captured['model']=='configured-model' and captured['sdk']['max_retries']==0


@pytest.mark.parametrize('output_tokens',[2048,3072])
def test_configured_output_budget_is_reserved_and_sent(output_tokens):
    bucket, clock=limiter(30,8000)
    sent=[]
    def sink(request):
        body=json.loads(request.content)
        sent.append(body)
        assert body['max_completion_tokens']==output_tokens
        assert sum(e.tokens for e in bucket.entries)==1000+output_tokens
        return httpx.Response(200,json={'id':'local','object':'chat.completion','created':0,'model':'fake',
            'usage':{'prompt_tokens':1000,'completion_tokens':1001,'total_tokens':2001},
            'choices':[{'index':0,'finish_reason':'stop','message':{'role':'assistant','content':'{}'}}]})
    with OpenAI(api_key='local',base_url='http://memory.invalid/v1',max_retries=0,
                http_client=httpx.Client(transport=httpx.MockTransport(sink))) as sdk:
        client=GovernedClient(sdk,bucket,output_tokens=output_tokens)
        client.create({'model':'fake','messages':[]},1000,60)
    assert len(sent)==1 and bucket.entries[0].tokens==2001


def test_larger_output_budget_cannot_bypass_tpm():
    bucket,clock=limiter(30,2000)
    client=GovernedClient(object(),bucket,output_tokens=2048)
    with pytest.raises(CapacityError,match='exceeds_tpm'):
        client.create({'model':'fake'},1000,60)
    assert not bucket.entries and not clock.waits


@pytest.mark.parametrize('value',[0,255,8193,True,'2048'])
def test_output_budget_rejects_invalid_values(value):
    bucket,_=limiter()
    with pytest.raises(ValueError):
        GovernedClient(object(),bucket,output_tokens=value)
