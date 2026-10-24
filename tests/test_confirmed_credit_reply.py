"""Uma falha ao redigir a resposta final não pode esconder um resultado de crédito confirmado."""
import pytest

from banco_agil.rate_limit import CapacityError
from banco_agil.messages import LLM_UNAVAILABLE
from banco_agil.response_guard import confirmed_result_reply
from structured_fakes import make, action
from support import snapshot_csv as csv
from orchestrator_fakes import logged_in


@pytest.mark.parametrize('mode',['native','structured'])
@pytest.mark.parametrize('failure',[ConnectionError('private'), TimeoutError('private'),CapacityError('admission_deadline')])
def test_consultation_is_reported_after_closing_failure(repo_ana_620,mode,failure):
    orch=make(repo_ana_620,[action('consultar_limite'),failure],mode)
    session=logged_in(orch,agent='credito')
    before=csv(orch.repo)
    response=orch.handle(session,'qual meu limite')
    assert response=='Seu limite de crédito atual é R$ 5.000,00.'
    assert csv(orch.repo)==before and not orch.repo.list_requests()
    assert len(orch.client.requests)==2 and not orch.client.script
    assert session.transcript[-1]==('assistant',response)
    assert session.history[-1]['content']==response
    assert session.authenticated and not session.ended


@pytest.mark.parametrize('mode',['native','structured'])
@pytest.mark.parametrize('failure',[ConnectionError('private'),CapacityError('admission_deadline')])
def test_financial_effect_is_reported_once_when_closing_generation_fails(repo_ana_620,mode,failure):
    orch=make(repo_ana_620,[action('solicitar_aumento_limite',novo_limite=8000),failure],mode)
    session=logged_in(orch,agent='credito')
    session.credit_proposal=dict(valor=8000,modalidade='limite_total',novo_limite=8000,confirmada=True)
    response=orch.handle(session,'sim')
    assert response=='Seu pedido foi aprovado e o novo limite de R$ 8.000,00 já está ativo.'
    assert len(orch.repo.list_requests())==1
    assert orch.repo.get_client(session.cpf)['limite_credito']==8000
    assert len(orch.client.requests)==2 and not orch.client.script
    assert session.authenticated and not session.ended
    assert session.last_failure_category in {'connection','local_capacity'}


@pytest.mark.parametrize('source',['old_result','error_result','no_result'])
def test_failure_before_execution_never_reports_old_or_failed_result(repo_ana_620,source):
    orch=make(repo_ana_620,[CapacityError('request_exceeds_tpm')])
    session=logged_in(orch,agent='credito')
    if source=='old_result':
        session.tool_facts['solicitar_aumento_limite']={'turn':0,'result':{
            'status_pedido':'aprovado','limite_aplicado':True,'limite_credito_atual':8000}}
    if source=='error_result':
        session.tool_facts['solicitar_aumento_limite']={'turn':1,'result':{'erro':'failed'}}
    before=csv(orch.repo)
    assert orch.handle(session,'Podemos continuar?')==LLM_UNAVAILABLE
    assert confirmed_result_reply(session) is None
    assert csv(orch.repo)==before and not orch.repo.list_requests()


def test_awaiting_confirmation_is_not_hidden_by_known_consultation(repo_ana_620):
    orch=make(repo_ana_620,[action('consultar_limite'),
        action('registrar_intencao_credito',valor=8000,modalidade='limite_total'),
        CapacityError('admission_deadline')])
    session=logged_in(orch,agent='credito')
    before=csv(orch.repo)
    response=orch.handle(session,'Quero conferir a proposta de um limite de 8000 antes de confirmar.')
    assert '5.000,00' in response and 'Dados da proposta' in response and '8.000,00' in response
    assert 'me envie **confirmo** para continuar' in response
    assert not session.credit_proposal['confirmada']
    assert csv(orch.repo)==before and not orch.repo.list_requests()


@pytest.mark.parametrize('closing_failure',['connection','capacity'])
def test_actual_sdk_closing_failure_reports_tool_result(repo_ana_620,closing_failure):
    import json
    import httpx2 as httpx
    from openai import OpenAI
    from banco_agil.rate_limit import GovernedClient, RateLimiter
    from banco_agil.orchestrator import Orchestrator
    from banco_agil.exchange import ExchangeService
    from fakes import Clock
    clock=Clock()
    bucket=RateLimiter(30,8000,clock=clock,sleep=clock.sleep)
    requests=[]
    tool='consultar_limite' if closing_failure=='connection' else 'solicitar_aumento_limite'
    def sink(request):
        requests.append(json.loads(request.content))
        if len(requests)>1:
            assert closing_failure=='connection'
            raise httpx.ConnectError('private-network-detail',request=request)
        clock.now=45
        if closing_failure=='capacity':
            bucket.defer(60)  # Cooldown aberto por um erro anterior do provedor.
        value=action(tool,**({'novo_limite':8000} if closing_failure=='capacity' else {}))
        return httpx.Response(200,
            headers={'x-ratelimit-remaining-tokens':'0','x-ratelimit-reset-tokens':'60s'} if closing_failure=='capacity' else {},
            json={'id':'local','object':'chat.completion','created':0,'model':'fake',
                'usage':{'prompt_tokens':100,'completion_tokens':100,'total_tokens':200},
                'choices':[{'index':0,'finish_reason':'stop','message':{'role':'assistant','content':json.dumps({'resposta':value})}}]})
    with OpenAI(api_key='local',base_url='http://memory.invalid/v1',max_retries=0,
                http_client=httpx.Client(transport=httpx.MockTransport(sink))) as sdk:
        orch=Orchestrator(repo_ana_620,ExchangeService(),GovernedClient(sdk,bucket),'fake',llm_mode='structured')
        session=logged_in(orch,agent='credito')
        if closing_failure=='capacity':
            session.credit_proposal=dict(valor=8000,modalidade='limite_total',novo_limite=8000,confirmada=True)
        response=orch.handle(session,'sim' if closing_failure=='capacity' else 'qual meu limite')
    assert 'indisponibilidade' not in response
    assert not clock.waits
    if closing_failure=='capacity':
        assert 'aprovado' in response and '8.000,00' in response
        assert len(requests)==1 and len(orch.repo.list_requests())==1
        assert orch.repo.get_client(session.cpf)['limite_credito']==8000
        assert session.last_failure_category=='local_capacity'
    else:
        assert '5.000,00' in response
        assert len(requests)==2 and not orch.repo.list_requests()
        assert session.last_failure_category=='connection'
