"""Gerações incompletas não executam ferramentas nem contam como erro do cliente."""
import copy
import json
import logging
from types import SimpleNamespace as NS

import pytest

from banco_agil.agents import AGENTS
from banco_agil.flow import FlowController
from banco_agil.messages import EMPTY_REPLY, LLM_UNAVAILABLE
from banco_agil.llm import call_llm
from banco_agil.session import Session
from fakes import FakeLLM, reply, assert_history_valid
from orchestrator_fakes import make, logged_in
from support import snapshot_csv as csv_bytes


def completion(text='', calls=(), *, finish='length', usage=True):
    value=reply(text,calls)
    value.choices[0].finish_reason=finish
    if usage:
        value.usage=NS(prompt_tokens=1170,completion_tokens=512,total_tokens=1682)
    return value


@pytest.mark.parametrize('text',[None,'','Texto parcial que não deve ser apresentado.'])
@pytest.mark.parametrize('calls',[
    [('solicitar_aumento_limite',{'novo_limite':8000})],
    [('solicitar_aumento_limite','{"novo_limite":')],
    [('registrar_intencao_credito',{'valor':8000,'modalidade':'limite_total'})],
])
def test_length_precedes_tool_validation_and_execution(repo_ana_620,text,calls,caplog):
    orch=make(repo_ana_620,[completion(text,calls)],max_llm_retries=1)
    session=logged_in(orch,agent='credito')
    before=csv_bytes(orch.repo)
    with caplog.at_level(logging.INFO,logger='banco_agil'):
        response=orch.handle(session,'quero aumentar para 8000')
    assert response==LLM_UNAVAILABLE and response!=EMPTY_REPLY
    assert session.last_failure_category=='truncated_response'
    assert session.last_tool_error_code is None
    assert session.authenticated and session.auth_failures==0 and not session.ended
    assert session.credit_proposal is None and session.last_request_id is None
    assert csv_bytes(orch.repo)==before
    assert len(orch.client.requests)==1 and not orch.client.script
    assert not any(m.get('tool_calls') for m in session.history)
    assert_history_valid(session.history)
    events=[json.loads(r.message) for r in caplog.records if r.message.startswith('{')]
    event=next(e for e in events if e['event']=='llm_call')
    assert event['status_http']==200 and event['finish_reason']=='length'
    assert event['result']=='incomplete' and event['output_tokens']==512
    assert event['input_tokens']==1170 and event['reported_tokens']==1682
    assert not any(e['event']=='tool_execution' for e in events)
    assert 'Texto parcial' not in caplog.text and 'novo_limite' not in caplog.text


@pytest.mark.parametrize('agent',['triagem','credito','entrevista','cambio'])
def test_truncation_preserves_pending_data_and_authenticated_session(repo_ana_620,agent):
    orch=make(repo_ana_620,[completion()])
    session=logged_in(orch,agent=agent)
    session.credit_proposal={'valor':8000,'modalidade':'limite_total','novo_limite':8000,'confirmada':False}
    if agent=='entrevista':
        FlowController.start_interview(session)
        session.flow.pending_interview_answers={'renda_mensal':6000,'despesas_fixas':2000}
    proposal=copy.deepcopy(session.credit_proposal)
    pending=copy.deepcopy(session.flow.pending_interview_answers)
    before=csv_bytes(orch.repo)
    response=orch.handle(session,'Pode esclarecer a etapa em andamento?')
    if agent == 'entrevista':
        assert 'entrevista continua ativa' in response and 'ainda aguardam sua confirmação' in response
        assert 'R$ 6.000,00' in response and 'R$ 2.000,00' in response
    else:
        assert response.startswith(LLM_UNAVAILABLE)
    assert session.credit_proposal==proposal and session.flow.pending_interview_answers==pending
    assert session.authenticated and session.active_agent==agent and not session.ended
    assert session.auth_failures==0 and csv_bytes(orch.repo)==before
    assert any(m.get('content')=='Pode esclarecer a etapa em andamento?' for m in session.history)


def test_truncation_does_not_replay_a_completed_financial_operation(repo_ana_620):
    orch=make(repo_ana_620,[reply(calls=[('solicitar_aumento_limite',{'novo_limite':8000})]),
        completion(calls=[('solicitar_aumento_limite',{'novo_limite':8000})])],max_llm_retries=1)
    session=logged_in(orch,agent='credito')
    response=orch.handle(session,'quero aumentar para 8000')
    assert response==LLM_UNAVAILABLE and session.last_failure_category=='truncated_response'
    assert len(orch.client.requests)==2
    requests=orch.repo.list_requests()
    assert len(requests)==1 and requests[0]['status_pedido']=='aprovado'
    assert orch.repo.get_client(session.cpf)['limite_credito']==8000
    assert len([c for m in session.history for c in m.get('tool_calls',[])])==1
    assert_history_valid(session.history)


def test_pending_reanalysis_is_not_misreported_as_success_after_truncation(repo_ana_620):
    orch=make(repo_ana_620,[completion()])
    session=logged_in(orch,agent='credito')
    session.automatic_reanalysis_response_pending=True
    session.automatic_reanalysis_result={'id':3,'status_pedido':'aprovado','limite_credito_atual':8000}
    before=csv_bytes(orch.repo)
    response=orch.handle(session,'Pode explicar o resultado?')
    assert response==LLM_UNAVAILABLE
    assert session.automatic_reanalysis_response_pending
    assert session.automatic_reanalysis_result['id']==3 and csv_bytes(orch.repo)==before


def test_stop_at_token_limit_remains_a_success(repo_ana_620):
    text='Como você prefere esclarecer o pedido de limite?'
    orch=make(repo_ana_620,[completion(text,finish='stop')])
    session=logged_in(orch,agent='credito')
    assert orch.handle(session,'Podemos conversar sobre meu limite?')==text
    assert session.last_failure_category is None and len(orch.client.requests)==1


def test_stop_can_execute_complete_authorized_tools_at_token_limit(repo_ana_620):
    orch=make(repo_ana_620,[completion(calls=[('solicitar_aumento_limite',{'novo_limite':8000})],finish='stop'),
        reply('Seu limite aprovado é R$ 8.000,00.')])
    session=logged_in(orch,agent='credito')
    orch.handle(session,'quero aumentar para 8000')
    assert session.last_failure_category is None
    assert len(orch.repo.list_requests())==1 and orch.repo.get_client(session.cpf)['limite_credito']==8000


def test_missing_usage_is_not_invented_in_truncation_diagnostic():
    session=Session()
    client=FakeLLM([completion(usage=False)])
    assert call_llm(client,'local',AGENTS['triagem'],session,max_retries=1) is None
    assert json.loads(session.last_llm_response)=={'status_http':200,'finish_reason':'length'}
    assert len(client.requests)==1


def test_empty_stop_is_different_from_length(repo):
    orch=make(repo,[completion(finish='stop')])
    session=logged_in(orch,agent='credito')
    assert orch.handle(session,'Preciso de ajuda')=='Desculpe, não consegui entender.\n\nPode reformular a sua mensagem?'
    assert session.last_failure_category=='empty_response'


@pytest.mark.parametrize('status,expected',[(400,'http_400'),(None,'connection')])
def test_http_and_connection_failures_remain_distinct(repo,status,expected):
    class ProviderFailure(Exception):status_code=status
    error=ProviderFailure('private provider detail') if status else ConnectionError('private connection detail')
    class Client:
        chat=NS(completions=NS())
        calls=0
        def create(self,**kwargs):
            self.calls+=1
            raise error
    from banco_agil.orchestrator import Orchestrator
    from banco_agil.exchange import ExchangeService
    client=Client();client.chat.completions.create=client.create
    orch=Orchestrator(repo,ExchangeService(),client,'local',max_llm_retries=0)
    session=logged_in(orch,agent='credito')
    assert orch.handle(session,'Preciso de ajuda')==LLM_UNAVAILABLE
    assert session.last_failure_category==expected and client.calls==1
