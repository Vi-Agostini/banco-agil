"""O cliente pode escrever durante o processamento, sem iniciar outro turno."""
from pathlib import Path
from threading import Event

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from banco_agil import bootstrap
from orchestrator_fakes import logged_in
from support import snapshot_csv as csv
from structured_fakes import make, message


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("first_result", [message("Podemos continuar."), TimeoutError()])
def test_early_submit_restores_draft_without_queueing_or_losing_session(repo_ana_620, monkeypatch, first_result):
    orch = make(repo_ana_620, [first_result, message("Recebi sua próxima mensagem.")])
    session = logged_in(orch, agent="credito")
    entered, release = Event(), Event()
    original = orch.client.create

    def blocked(**kwargs):
        if not entered.is_set():
            entered.set()
            assert release.wait(10)
        return original(**kwargs)

    orch.client.chat.completions.create = blocked
    monkeypatch.setattr(bootstrap, "build_orchestrator", lambda: orch)
    monkeypatch.setattr(orch, "new_session", lambda: session)
    monkeypatch.setattr("banco_agil.processing.sleep_seconds", lambda _: None)
    before = csv(orch.repo)
    st.cache_resource.clear()
    try:
        app = AppTest.from_file(str(ROOT / "app.py")).run()
        widget_id = app.chat_input[0].id
        app.chat_input[0].set_value("Podemos conversar?").run()
        assert entered.wait(5)
        job = app.session_state["processing_job"]
        assert app.chat_input[0].id == widget_id and not app.chat_input[0].disabled
        assert app.sidebar.button[0].disabled

        # Simula um envio antes da trava do navegador ou contornando essa trava.
        draft = "Quero preparar minha próxima mensagem"
        app.chat_input[0].set_value(draft).run()
        assert not app.exception
        assert app.chat_input[0].proto.set_value and app.chat_input[0].proto.value == draft
        assert app.session_state["processing_job"] is job and session.turn == 1
        assert [text for role, text in session.transcript if role == "user"] == ["Podemos conversar?"]
        app.run()
        assert session.turn == 1 and app.chat_input[0].id == widget_id

        release.set()
        job.thread.join(5)
        app.run()
        assert not app.exception and not app.chat_input[0].disabled
        assert session.turn == 1 and session.authenticated and not session.ended
        assert csv(orch.repo) == before and len(orch.client.requests) == 1
        # Concluir o processamento não envia o rascunho.
        app.run()
        assert session.turn == 1 and len(orch.client.requests) == 1

        app.chat_input[0].set_value(draft).run()
        next_job = app.session_state.get("processing_job")
        if next_job is not None:
            next_job.thread.join(5)
        app.run()
        assert not app.exception and session.turn == 2
        assert len(orch.client.requests) == 2 and not orch.client.script
        assert csv(orch.repo) == before
        assert app.chat_message[-1].markdown[0].value == "Recebi sua próxima mensagem."
    finally:
        release.set()
        if "job" in locals():
            job.thread.join(5)
        st.cache_resource.clear()


def test_completion_before_rerun_does_not_admit_an_early_submission(repo_ana_620, monkeypatch):
    orch = make(repo_ana_620, [message("Podemos continuar.")])
    session = logged_in(orch, agent="credito")
    monkeypatch.setattr(bootstrap, "build_orchestrator", lambda: orch)
    monkeypatch.setattr(orch, "new_session", lambda: session)
    ready, release = Event(), Event()

    def pacing(_):
        ready.set()
        assert release.wait(10)

    monkeypatch.setattr("banco_agil.processing.sleep_seconds", pacing)
    before = csv(orch.repo)
    st.cache_resource.clear()
    try:
        app = AppTest.from_file(str(ROOT / "app.py")).run()
        app.chat_input[0].set_value("Podemos conversar?").run()
        assert ready.wait(5)
        job = app.session_state["processing_job"]
        release.set()
        job.thread.join(5)
        assert job.future.done()
        # A tela ainda pode mostrar o processamento anterior; essa confirmação
        # antecipada não autoriza a operação seguinte.
        app.chat_input[0].set_value("confirmo").run()
        assert not app.exception and session.turn == 1
        assert app.chat_input[0].proto.set_value and app.chat_input[0].proto.value == "confirmo"
        assert len(orch.client.requests) == 1 and csv(orch.repo) == before
    finally:
        release.set()
        if "job" in locals():
            job.thread.join(5)
        st.cache_resource.clear()


def test_new_attendance_clears_the_previous_draft(repo_ana_620, monkeypatch):
    orch = make(repo_ana_620, [])
    monkeypatch.setattr(bootstrap, "build_orchestrator", lambda: orch)
    before = csv(orch.repo)
    st.cache_resource.clear()
    try:
        app = AppTest.from_string(f"""
import runpy
import streamlit as st
if 'draft_seeded' not in st.session_state:
    st.session_state.customer_message = 'Rascunho do atendimento anterior'
    st.session_state.draft_seeded = True
runpy.run_path({str(ROOT / 'app.py')!r})
""").run()
        previous_session = app.session_state["session"]
        assert app.chat_input[0].proto.value == "Rascunho do atendimento anterior"
        app.sidebar.button[0].click().run()
        assert not app.exception
        assert app.session_state["session"] is not previous_session
        assert app.chat_input[0].proto.set_value and app.chat_input[0].proto.value == ""
        assert not orch.client.requests and csv(orch.repo) == before
    finally:
        st.cache_resource.clear()
