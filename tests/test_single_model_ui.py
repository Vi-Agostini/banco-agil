"""Modelo único e processamento na interface, ignorando avisos de troca antigos."""
import copy
from pathlib import Path
from threading import Event

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from banco_agil import bootstrap
from banco_agil.processing import BUSY
from orchestrator_fakes import logged_in
from structured_fakes import make, message
from support import snapshot_csv as csv

ROOT = Path(__file__).resolve().parents[1]
LEGACY_NOTICE = "Modelo alternativo selecionado. Seu atendimento permanece ativo."


@pytest.fixture(autouse=True)
def clean_cache(monkeypatch):
    monkeypatch.setattr(bootstrap, "setup_logging", lambda *_: None)
    st.cache_resource.clear()
    yield
    st.cache_resource.clear()


def reject_model_toasts(monkeypatch):
    def toast(*args, **kwargs):
        raise AssertionError("Removed model selection must not display a toast")
    monkeypatch.setattr(st, "toast", toast)


def test_legacy_session_model_fields_are_ignored_without_changing_conversation(repo_ana_620, monkeypatch):
    orch = make(repo_ana_620, [])
    session = logged_in(orch, agent="entrevista")
    session.flow.interview_started = True
    session.flow.interview_answers = {"renda_mensal": 6000}
    session.flow.pending_interview_answers = {"tipo_emprego": "formal"}
    session.credit_proposal = dict(valor=8000, modalidade="limite_total", novo_limite=8000, confirmada=False)
    # Simula campos antigos ainda presentes numa sessão em execução.
    session.active_model = "openai/gpt-oss-20b"
    session.model_notice = LEGACY_NOTICE
    session.model_switch_announced = True
    before = (copy.deepcopy(session.history), copy.deepcopy(session.transcript),
              copy.deepcopy(session.flow), copy.deepcopy(session.credit_proposal), csv(orch.repo))
    monkeypatch.setattr(orch, "new_session", lambda: session)
    monkeypatch.setattr(bootstrap, "build_orchestrator", lambda: orch)
    reject_model_toasts(monkeypatch)
    app = AppTest.from_file(str(ROOT / "app.py")).run()
    assert not app.exception and not app.get("toast")
    assert session.authenticated and session.active_agent == "entrevista"
    assert before == (session.history, session.transcript, session.flow, session.credit_proposal, csv(orch.repo))
    assert LEGACY_NOTICE not in " ".join(item.value for item in app.markdown)
    assert not any("stToastContainer" in item.value for item in app.markdown)
    app.run()
    assert not app.exception and not orch.client.requests


def test_processing_keeps_editable_input_animation_pacing_and_duplicate_protection(repo_ana_620, monkeypatch):
    orch = make(repo_ana_620, [message("Sua entrevista continua ativa.")])
    session = logged_in(orch, agent="entrevista")
    session.flow.interview_started = True
    session.flow.interview_answers = {"renda_mensal": 6000}
    entered, release = Event(), Event()

    def pause(seconds):
        assert seconds == 5
        entered.set()
        assert release.wait(10)

    monkeypatch.setattr("banco_agil.processing.sleep_seconds", pause)
    monkeypatch.setattr(orch, "new_session", lambda: session)
    monkeypatch.setattr(bootstrap, "build_orchestrator", lambda: orch)
    reject_model_toasts(monkeypatch)
    before = csv(orch.repo)
    app = AppTest.from_file(str(ROOT / "app.py")).run()
    try:
        app.chat_input[0].set_value("Podemos continuar?").run()
        assert entered.wait(5)
        job = app.session_state["processing_job"]
        session.model_notice = LEGACY_NOTICE
        app.run()
        assert not app.exception and not app.get("toast")
        assert not app.chat_input[0].disabled and app.sidebar.button[0].disabled
        assert app.session_state["processing_job"] is job
        assert any(BUSY in element.value for element in app.get("html"))
        assert orch.handle(session, "duplicada") == BUSY and session.turn == 1
        app.run()
        assert len(orch.client.requests) == 1
        assert not any("Sua entrevista continua ativa" in item.value for item in app.markdown)
        release.set()
        job.thread.join(5)
        app.run()
        assert not app.exception and not app.chat_input[0].disabled and not app.sidebar.button[0].disabled
        assert not any(BUSY in element.value for element in app.get("html"))
        assert app.chat_message[-1].markdown[0].value == "Sua entrevista continua ativa."
        assert session.authenticated and session.flow.interview_answers == {"renda_mensal": 6000}
        assert len(orch.client.requests) == 1 and csv(orch.repo) == before
    finally:
        release.set()
        if "job" in locals():
            job.thread.join(5)


def test_new_attendance_does_not_recreate_removed_model_state(repo_ana_620, monkeypatch):
    orch = make(repo_ana_620, [])
    initial = orch.new_session()
    initial.active_model, initial.model_notice = "openai/gpt-oss-20b", LEGACY_NOTICE
    original_new = orch.new_session
    sessions = [initial]

    def new():
        if sessions:
            return sessions.pop()
        return original_new()

    monkeypatch.setattr(orch, "new_session", new)
    monkeypatch.setattr(bootstrap, "build_orchestrator", lambda: orch)
    reject_model_toasts(monkeypatch)
    app = AppTest.from_file(str(ROOT / "app.py")).run()
    assert not app.exception and not app.get("toast")
    app.sidebar.button[0].click().run()
    session = app.session_state["session"]
    assert not app.exception and session is not initial
    assert not hasattr(session, "model_notice") and not hasattr(session, "active_model")
    assert not hasattr(session, "model_switch_announced")
    assert not session.authenticated and not orch.client.requests and len(session.transcript) == 1
