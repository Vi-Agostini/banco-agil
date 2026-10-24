"""Respostas normais não têm espera artificial; a cotação sobrevive à falha de redação final."""
import json

import httpx2 as httpx
from openai import OpenAI
import pytest

from banco_agil.agents import AGENTS
from banco_agil.llm import call_llm
from banco_agil.rate_limit import RateLimiter, GovernedClient
from banco_agil.session import Session
from fakes import Clock
from structured_fakes import make, action, message
from support import snapshot_csv as csv
from orchestrator_fakes import logged_in


@pytest.mark.parametrize('mode',['native','structured'])
def test_success_headers_and_usage_never_delay_a_healthy_request(mode):
    clock=Clock()
    bucket=RateLimiter(1,8000,clock=clock,sleep=clock.sleep)
    sent=[]
    def sink(request):
        sent.append(json.loads(request.content))
        content='Tudo bem.' if mode=='native' else json.dumps({'resposta':message('Tudo bem.')})
        return httpx.Response(200,headers={'x-ratelimit-remaining-tokens':'0',
                'x-ratelimit-reset-tokens':'60s','x-ratelimit-remaining-requests':'0',
                'x-ratelimit-reset-requests':'24h'},
            json={'id':'local','object':'chat.completion','created':0,'model':'fake',
                'usage':{'prompt_tokens':4000,'completion_tokens':2000,'total_tokens':6000},
                'choices':[{'index':0,'finish_reason':'stop','message':{'role':'assistant','content':content}}]})
    with OpenAI(api_key='local',base_url='http://memory.invalid/v1',max_retries=0,
                http_client=httpx.Client(transport=httpx.MockTransport(sink))) as sdk:
        client=GovernedClient(sdk,bucket)
        for _ in range(3):
            assert call_llm(client,'fake',AGENTS['triagem'],Session(),llm_mode=mode)
    assert len(sent)==3 and not clock.waits
    assert sum(e.tokens for e in bucket.entries)==18000


@pytest.mark.parametrize('mode',['native','structured'])
def test_successful_exchange_is_reported_after_connection_failure(repo_ana_620,mode):
    orch=make(repo_ana_620,[action('transferir_para_agente',destino='cambio'),
        action('consultar_cotacao',moeda='USD'),ConnectionError('private')],mode)
    rates=[]
    def quote(currency):
        rates.append(currency)
        return dict(moeda=currency,valor_em_brl=5.1234,fonte='API simulada',atualizado_em='22/10/2026 às 16:42')
    orch.fx.get_rate=quote
    session=logged_in(orch,agent='credito')
    before=csv(orch.repo)
    answer=orch.handle(session,'Qual a cotação do dólar?')
    assert 'USD' in answer and '5,1234' in answer and 'API simulada' in answer
    assert '16:42' in answer and 'indisponibilidade' not in answer
    assert rates==['USD'] and len(orch.client.requests)==3
    assert session.active_agent=='cambio' and session.authenticated and not session.ended
    assert csv(orch.repo)==before


@pytest.mark.parametrize('source',['old','error','absent'])
def test_failed_exchange_does_not_reuse_previous_quote(repo_ana_620,source):
    from banco_agil.messages import LLM_UNAVAILABLE
    orch=make(repo_ana_620,[ConnectionError('private')])
    session=logged_in(orch,agent='cambio')
    if source=='old':
        session.tool_facts['consultar_cotacao']={'turn':0,'result':dict(moeda='USD',valor_em_brl=5,fonte='old')}
    elif source=='error':
        session.tool_facts['consultar_cotacao']={'turn':1,'result':{'erro':'failed'}}
    assert orch.handle(session,'Qual a cotação do dólar?')==LLM_UNAVAILABLE


@pytest.mark.parametrize('code',['tool_use_failed','json_validate_failed'])
def test_quote_is_preserved_after_selective_recovery_is_exhausted(repo_ana_620,code):
    from fakes import failure
    orch=make(repo_ana_620,[action('consultar_cotacao',moeda='USD'),failure(code),failure(code)])
    calls=[]
    def quote(currency):
        calls.append(currency)
        return dict(moeda=currency,valor_em_brl=5.1234,fonte='API simulada',atualizado_em='22/10/2026')
    orch.fx.get_rate=quote
    session=logged_in(orch,agent='cambio')
    before=csv(orch.repo)
    answer=orch.handle(session,'Qual a cotação do dólar?')
    assert 'USD' in answer and '5,1234' in answer and 'API simulada' in answer
    assert calls==['USD'] and len(orch.client.requests)==3
    assert csv(orch.repo)==before and not session.ended


def test_exchange_handoff_and_quote_do_not_wait_on_success_headers(repo_ana_620):
    from banco_agil.orchestrator import Orchestrator
    from banco_agil.exchange import ExchangeService
    clock=Clock()
    bucket=RateLimiter(1,8000,clock=clock,sleep=clock.sleep)
    sent=[]
    script=[action('transferir_para_agente',destino='cambio'),action('consultar_cotacao',moeda='USD'),
            message('A cotação do USD é R$ 5,1234. Posso ajudar com outro assunto?')]
    def sink(request):
        sent.append(json.loads(request.content))
        clock.now+=25  # O processamento HTTP pode ultrapassar o prazo inicial de admissão.
        return httpx.Response(200,headers={'x-ratelimit-remaining-tokens':'0','x-ratelimit-reset-tokens':'60s'},
            json={'id':'local','object':'chat.completion','created':0,'model':'fake',
                'usage':{'prompt_tokens':4000,'completion_tokens':1000,'total_tokens':5000},
                'choices':[{'index':0,'finish_reason':'stop','message':{'role':'assistant','content':json.dumps({'resposta':script.pop(0)})}}]})
    rates=[]
    def quote(currency):
        rates.append(currency)
        return dict(moeda=currency,valor_em_brl=5.1234,fonte='API simulada',atualizado_em='22/10/2026')
    with OpenAI(api_key='local',base_url='http://memory.invalid/v1',max_retries=0,
                http_client=httpx.Client(transport=httpx.MockTransport(sink))) as sdk:
        fx=ExchangeService()
        fx.get_rate=quote
        orch=Orchestrator(repo_ana_620,fx,GovernedClient(sdk,bucket),'fake',llm_mode='structured')
        session=logged_in(orch,agent='credito')
        before=csv(orch.repo)
        answer=orch.handle(session,'Qual a cotação do dólar?')
    assert rates==['USD'] and len(sent)==3 and not script and not clock.waits
    assert '5,1234' in answer and 'indisponibilidade' not in answer
    assert session.active_agent=='cambio' and session.authenticated
    assert csv(orch.repo)==before
