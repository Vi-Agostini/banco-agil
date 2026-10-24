"""A pausa de apresentação de cinco segundos é simulada e não repete operações."""
from threading import Event
from pathlib import Path


from banco_agil.processing import ProcessingJob, BUSY
from structured_fakes import make, message
from support import snapshot_csv as csv
from orchestrator_fakes import logged_in


def test_five_second_pause_is_once_per_reply_and_keeps_server_locked(repo_ana_620,monkeypatch):
    orch=make(repo_ana_620,[message('Podemos continuar.')])
    session=logged_in(orch,agent='credito')
    before=csv(orch.repo)
    entered,release=Event(),Event()
    waits=[]
    def pause(seconds):
        waits.append(seconds)
        entered.set()
        assert release.wait(5)
    monkeypatch.setattr('banco_agil.processing.sleep_seconds',pause)
    job=ProcessingJob(orch,session,'Vamos conversar?')
    try:
        assert entered.wait(5) and waits==[5.0]
        assert not job.future.done()
        assert ('assistant','Podemos continuar.') not in job.visible_transcript
        assert ('user','Vamos conversar?') in job.visible_transcript
        assert orch.handle(session,'duplicada')==BUSY
        assert session.turn==1 and len(orch.client.requests)==1
    finally:
        release.set()
        job.thread.join(5)
    assert job.future.result()=='Podemos continuar.'
    assert not session.processing_guard.lock.locked() and csv(orch.repo)==before


def test_direct_orchestrator_calls_do_not_sleep(repo_ana_620,monkeypatch):
    waits=[]
    monkeypatch.setattr('banco_agil.processing.sleep_seconds',waits.append)
    orch=make(repo_ana_620,[message('Olá!')])
    session=logged_in(orch,agent='credito')
    assert orch.handle(session,'Podemos conversar?')=='Olá!'
    assert not waits


def test_pacing_applies_to_error_reply_and_never_retries(repo_ana_620,monkeypatch):
    waits=[]
    monkeypatch.setattr('banco_agil.processing.sleep_seconds',waits.append)
    orch=make(repo_ana_620,[ConnectionError('private')])
    session=logged_in(orch,agent='credito')
    before=csv(orch.repo)
    job=ProcessingJob(orch,session,'Podemos conversar?')
    job.thread.join(5)
    assert waits==[5.0] and 'indisponibilidade' in job.future.result()
    assert len(orch.client.requests)==1 and csv(orch.repo)==before


def test_ui_hides_reply_during_pacing_and_survives_rerun(repo_ana_620,monkeypatch):
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    from banco_agil import bootstrap
    orch=make(repo_ana_620,[message('Resposta pronta para exibir.')])
    session=logged_in(orch,agent='credito')
    monkeypatch.setattr(orch,'new_session',lambda:session)
    monkeypatch.setattr(bootstrap,'build_orchestrator',lambda:orch)
    entered,release=Event(),Event()
    waits=[]
    def pause(seconds):
        waits.append(seconds)
        entered.set()
        assert release.wait(10)
    monkeypatch.setattr('banco_agil.processing.sleep_seconds',pause)
    st.cache_resource.clear()
    try:
        app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'app.py')).run()
        app.chat_input[0].set_value('Podemos conversar?').run()
        assert entered.wait(5)
        job=app.session_state['processing_job']
        assert not app.chat_input[0].disabled and app.sidebar.button[0].disabled
        assert BUSY in app.get('html')[0].value
        assert not any('Resposta pronta' in m.value for m in app.markdown)
        app.run()
        assert app.session_state['processing_job'] is job and waits==[5.0]
        assert not any('Resposta pronta' in m.value for m in app.markdown)
        assert len(orch.client.requests)==1
        release.set()
        job.thread.join(5)
        app.run()
        assert not app.exception and not app.chat_input[0].disabled
        assert not app.sidebar.button[0].disabled and not any(BUSY in element.value for element in app.get('html'))
        assert app.chat_message[-1].markdown[0].value=='Resposta pronta para exibir.'
    finally:
        release.set()
        st.cache_resource.clear()


def test_five_seconds_are_added_once_after_429_recovery(repo_ana_620,monkeypatch):
    import json
    import httpx2 as httpx
    from openai import OpenAI
    from banco_agil.orchestrator import Orchestrator
    from banco_agil.exchange import ExchangeService
    from banco_agil.rate_limit import RateLimiter, GovernedClient
    from fakes import Clock
    clock=Clock()
    phases=[]
    def cooldown(seconds):
        phases.append(('429',seconds))
        clock.sleep(seconds)
    def presentation(seconds):
        phases.append(('reply',seconds))
        clock.now+=seconds
    monkeypatch.setattr('banco_agil.processing.sleep_seconds',presentation)
    bucket=RateLimiter(30,8000,clock=clock,sleep=cooldown)
    sent=[]
    def sink(request):
        sent.append(json.loads(request.content))
        if len(sent)==1:
            return httpx.Response(429,headers={'retry-after':'20'},json={'error':{'message':'busy'}})
        return httpx.Response(200,json={'id':'local','object':'chat.completion','created':0,'model':'fake',
            'choices':[{'index':0,'finish_reason':'stop','message':{'role':'assistant','content':json.dumps({'resposta':message('Podemos continuar.')})}}]})
    with OpenAI(api_key='local',base_url='http://memory.invalid/v1',max_retries=0,
                http_client=httpx.Client(transport=httpx.MockTransport(sink))) as sdk:
        orch=Orchestrator(repo_ana_620,ExchangeService(),GovernedClient(sdk,bucket),'fake',llm_mode='structured')
        session=logged_in(orch,agent='credito')
        before=csv(orch.repo)
        job=ProcessingJob(orch,session,'Podemos conversar?')
        job.thread.join(5)
    assert job.future.result()=='Podemos continuar.'
    assert phases==[('429',20.0),('reply',5.0)]
    assert len(sent)==2 and sent[0]==sent[1]
    assert csv(orch.repo)==before and not session.processing_guard.lock.locked()
