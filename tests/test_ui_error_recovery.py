"""Recuperação na interface com AppTest, HTTP e relógio simulados e dados isolados."""
from pathlib import Path
from threading import Event
from types import SimpleNamespace
import time
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest
from banco_agil import bootstrap, llm
from banco_agil.processing import BUSY
from fakes import failure, RateLimited
from orchestrator_fakes import logged_in
from structured_fakes import make
from support import snapshot_csv as csv

ROOT = Path(__file__).resolve().parents[1]

@pytest.mark.parametrize('kind,category,calls,cooldown', [
    ('http400_invalid','http_400',1,[]),
    ('http400_generation','http_400',2,[]),
    ('http429_exhausted','http_429',2,[20.0]),
    ('timeout','timeout',1,[]),
    ('unexpected',None,0,[]),
])
def test_app_releases_after_simulated_errors_and_never_duplicates(repo_ana_620,monkeypatch,kind,category,calls,cooldown):
    if kind=='http400_invalid': script=[failure('unsupported_parameter')]
    elif kind=='http400_generation': script=[failure('json_validate_failed'),failure('json_validate_failed')]
    elif kind=='http429_exhausted': script=[RateLimited('2'),RateLimited('2')]
    elif kind=='timeout': script=[TimeoutError('private timeout detail')]
    else: script=[]
    orch=make(repo_ana_620,script)
    session=logged_in(orch,agent='credito')
    monkeypatch.setattr(orch,'new_session',lambda:session)
    monkeypatch.setattr(bootstrap,'build_orchestrator',lambda:orch)
    monkeypatch.setattr(bootstrap,'setup_logging',lambda *_:None)
    waits,pauses=[],[]
    monkeypatch.setattr(llm,'time',SimpleNamespace(sleep=waits.append, time=time.time, perf_counter=time.perf_counter, monotonic=time.monotonic))
    ready,release=Event(),Event()
    def pacing(seconds):
        pauses.append(seconds)
        ready.set()
        assert release.wait(10)
    monkeypatch.setattr('banco_agil.processing.sleep_seconds',pacing)
    if kind=='unexpected':
        def fail_interceptor(*_): raise RuntimeError('private unexpected detail')
        monkeypatch.setattr(orch.interceptors,'try_handle',fail_interceptor)
    before=csv(orch.repo)
    st.cache_resource.clear()
    try:
        app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=15).run()
        app.chat_input[0].set_value('Podemos conversar?').run()
        assert ready.wait(5)
        job=app.session_state['processing_job']
        assert not app.chat_input[0].disabled
        assert next(b for b in app.sidebar.button if b.label=='Novo atendimento').disabled
        html=app.get('html')[0].value
        assert BUSY in html and html.count('<span>●</span>')==3
        assert 'animation:bank-pulse' in html and 'role="status"' in html
        assert not any('indisponibilidade' in m.value for m in app.markdown)
        assert orch.handle(session,'duplicada')==BUSY
        assert session.turn==1
        app.run()
        assert app.session_state['processing_job'] is job
        assert session.turn==1 and len(orch.client.requests)==calls
        release.set()
        job.thread.join(5)
        app.run()
        assert not app.exception and not app.chat_input[0].disabled
        assert not next(b for b in app.sidebar.button if b.label=='Novo atendimento').disabled
        assert not any(BUSY in element.value for element in app.get('html')) and not session.processing_guard.lock.locked()
        assert 'indisponibilidade' in app.chat_message[-1].markdown[0].value
        assert 'private' not in app.chat_message[-1].markdown[0].value
        assert len(orch.client.requests)==calls and waits==cooldown and pauses==[5.0]
        if category: assert session.last_failure_category==category
        assert session.authenticated and not session.ended and csv(orch.repo)==before
    finally:
        release.set()
        if 'job' in locals(): job.thread.join(5)
        st.cache_resource.clear()
