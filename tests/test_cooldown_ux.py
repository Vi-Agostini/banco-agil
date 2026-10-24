"""Recuperação de 429 e proteção contra reenvio em testes locais."""
import json
from threading import Event
from email.utils import formatdate

import httpx2 as httpx
from openai import OpenAI
import pytest

from banco_agil import llm
from banco_agil.agents import AGENTS
from banco_agil.session import Session
from banco_agil.processing import BUSY, ProcessingJob
from structured_fakes import make, message, action
from support import snapshot_csv as csv
from orchestrator_fakes import logged_in


@pytest.fixture(autouse=True)
def simulated_reply_pacing(monkeypatch):
    monkeypatch.setattr("banco_agil.processing.sleep_seconds", lambda _: None)


@pytest.mark.parametrize('mode', ['native', 'structured'])
@pytest.mark.parametrize('header,delay', [(None,20), ('2',20), ('35',35), ('invalid',20), ('121',None)])
def test_http_cooldown(mode, header, delay, monkeypatch):
    requests, waits = [], []
    monkeypatch.setattr(llm.time, 'sleep', waits.append)
    def sink(request):
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            return httpx.Response(429, headers={} if header is None else {'Retry-After':header}, json={'error':{'message':'busy'}})
        content = 'Tudo bem.' if mode == 'native' else json.dumps({'resposta':message('Tudo bem.')})
        return httpx.Response(200, json={'id':'local','object':'chat.completion','created':0,'model':'fake',
            'choices':[{'index':0,'finish_reason':'stop','message':{'role':'assistant','content':content}}]})
    with OpenAI(api_key='local', base_url='http://memory.invalid/v1', max_retries=0,
                http_client=httpx.Client(transport=httpx.MockTransport(sink))) as client:
        result = llm.call_llm(client, 'fake', AGENTS['triagem'], Session(), llm_mode=mode)
    assert waits == ([] if delay is None else [delay])
    assert len(requests) == (1 if delay is None else 2)
    if delay is not None:
        assert requests[0] == requests[1] and result.content == 'Tudo bem.'
    else:
        assert result is None


def test_http_date_and_safe_config(monkeypatch):
    monkeypatch.setattr(llm.time, 'time', lambda: 1000)
    error = type('Error', (Exception,), {'headers':{'Retry-After':formatdate(1045, usegmt=True)}})()
    assert llm._retry_after_seconds(error, 60) == 45
    assert llm._retry_after_seconds(error, 30) is None
    assert llm.configured_rate_limit_wait_seconds('1') == 20
    assert llm.configured_rate_limit_wait_seconds('99999') == 120


def test_session_admission_and_release(repo):
    orch = make(repo, [message('Podemos continuar.')])
    session = logged_in(orch, agent='credito')
    entered, release = Event(), Event()
    original = orch.client.create
    def blocked(**kwargs):
        entered.set()
        assert release.wait(5)
        return original(**kwargs)
    orch.client.chat.completions.create = blocked
    job = ProcessingJob(orch, session, 'Vamos conversar?')
    try:
        assert entered.wait(5)
        before = (session.turn, list(session.history), csv(repo))
        assert orch.handle(session, 'duplicada') == BUSY
        assert before == (session.turn, session.history, csv(repo))
    finally:
        release.set()
        job.thread.join(5)
    assert job.future.result() == 'Podemos continuar.'
    assert not session.processing_guard.lock.locked()
    assert len(orch.client.requests) == 1


@pytest.mark.parametrize('mode', ['native', 'structured'])
def test_429_after_financial_effect_never_replays_operation(repo_ana_620, mode, monkeypatch):
    from fakes import RateLimited
    monkeypatch.setattr(llm.time, 'sleep', lambda _: None)
    orch = make(repo_ana_620, [action('solicitar_aumento_limite', novo_limite=8000),
        RateLimited(), message('Seu limite atual é R$ 8.000,00.')], mode)
    session = logged_in(orch, agent='credito')
    session.credit_proposal = dict(valor=8000, modalidade='limite_total', novo_limite=8000, confirmada=True)
    orch.handle(session, 'sim')
    assert len(orch.repo.list_requests()) == 1
    assert orch.repo.get_client(session.cpf)['limite_credito'] == 8000
    assert len(orch.client.requests) == 3


def test_busy_ui_survives_rerun_and_unlocks(repo, monkeypatch):
    from pathlib import Path
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    from banco_agil import bootstrap
    orch = make(repo, [message('Podemos continuar.')])
    entered, release = Event(), Event()
    original = orch.client.create
    def blocked(**kwargs):
        entered.set()
        assert release.wait(10)
        return original(**kwargs)
    orch.client.chat.completions.create = blocked
    monkeypatch.setattr(bootstrap, 'build_orchestrator', lambda: orch)
    st.cache_resource.clear()
    try:
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1]/'app.py')).run()
        app.chat_input[0].set_value('Quais serviços existem?').run()
        assert entered.wait(5)
        job = app.session_state['processing_job']
        assert not app.chat_input[0].disabled
        assert next(b for b in app.sidebar.button if b.label == 'Novo atendimento').disabled
        assert BUSY in app.get('html')[0].value
        app.run()  # O rerun não pode enviar a mesma entrada pela segunda vez.
        assert app.session_state['processing_job'] is job
        assert app.session_state['session'].turn == 1
        release.set()
        job.thread.join(5)
        app.run()
        assert not app.exception
        assert not app.chat_input[0].disabled
        assert not next(b for b in app.sidebar.button if b.label == 'Novo atendimento').disabled
        assert not any(BUSY in element.value for element in app.get('html'))
        assert len(orch.client.requests) == 1
        assert app.chat_message[-1].markdown[0].value == 'Podemos continuar.'
    finally:
        release.set()
        st.cache_resource.clear()
