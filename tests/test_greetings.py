"""Saudações isoladas não interferem na coleta local de credenciais."""
import pytest
from banco_agil.intents import is_isolated_greeting
from banco_agil.flow import Awaiting
from orchestrator_fakes import make
from fakes import reply


def csv_bytes(repo):
    return [p.read_bytes() for p in [repo.clients_path,repo.score_path,repo.requests_path]]


@pytest.mark.parametrize('text',['oi','olá','bom dia','boa tarde','boa noite','  OLÁ!!!  ','Bom\nDia!'])
@pytest.mark.parametrize('birth_pending',[False,True])
def test_greeting_preserves_authentication_stage_and_security(repo,text,birth_pending):
    orch=make(repo,[]);session=orch.new_session()
    if birth_pending:
        session.flow.pending_cpf='11144477735';session.flow.awaiting=Awaiting.BIRTH_DATE
    session.auth_failures=1;session.flow.unclear_count=2
    before=csv_bytes(repo);state=(session.flow.awaiting,session.flow.pending_cpf)
    response=orch.handle(session,text)
    assert 'Olá!' in response
    assert ('nascimento' in response) if birth_pending else ('CPF' in response)
    assert not session.authenticated and not session.ended
    assert session.auth_failures==1 and session.flow.unclear_count==2
    assert (session.flow.awaiting,session.flow.pending_cpf)==state
    assert not orch.client.requests and not any(m.get('tool_calls') for m in session.history)
    assert csv_bytes(repo)==before


@pytest.mark.parametrize('text',['foi','coisa','oito','oi quero crédito','oi 11144477735','bom dia quero encerrar','boa noite 01/01/1990','olá quero cotação','sair do rotativo','','xyzzy'])
def test_mixed_text_is_not_an_isolated_greeting(text):
    assert not is_isolated_greeting(text)


def test_repeated_greetings_do_not_trigger_repeated_reply_lockout(repo):
    orch=make(repo,[]);session=orch.new_session()
    for _ in range(4):orch.handle(session,'oi')
    assert not session.ended and session.auth_failures==session.flow.unclear_count==0
    assert not orch.client.requests


def test_credentials_continue_after_greeting_between_fields(repo):
    orch=make(repo,[]);session=orch.new_session();before=csv_bytes(repo)
    orch.handle(session,'oi');orch.handle(session,'11144477735')
    assert session.flow.awaiting is Awaiting.BIRTH_DATE
    orch.handle(session,'boa tarde')
    assert not session.authenticated
    orch.handle(session,'12/05/1990')
    assert session.authenticated and session.auth_failures==0
    assert not orch.client.requests and csv_bytes(repo)==before


def test_invalid_complete_credentials_still_consume_failure(repo):
    orch=make(repo,[]);session=orch.new_session();before=csv_bytes(repo)
    orch.handle(session,'oi');orch.handle(session,'11144477735 01/01/2000')
    assert not session.authenticated and session.auth_failures==1
    assert session.flow.unclear_count==0 and csv_bytes(repo)==before


def test_garbage_keeps_three_invalid_entries_policy(repo):
    script=[]
    for _ in range(2):script.extend([reply(calls=[('registrar_entrada_invalida',{})]),reply('Não entendi; reformule.')])
    script.append(reply(calls=[('registrar_entrada_invalida',{})]))
    orch=make(repo,script);session=orch.new_session();before=csv_bytes(repo)
    for value in ['xyzzy','blorb','asdfgh']:orch.handle(session,value)
    assert session.ended and session.flow.unclear_count==3 and session.auth_failures==0
    assert csv_bytes(repo)==before


@pytest.mark.parametrize('agent',['credito','entrevista','cambio'])
def test_greeting_interceptor_does_not_replace_other_agents(repo,agent):
    orch=make(repo,[reply('Resposta do agente no seu contexto.')]);session=orch.new_session()
    session.active_agent=agent;session.authenticated=True
    assert orch.handle(session,'oi')=='Resposta do agente no seu contexto.'
    assert len(orch.client.requests)==1
