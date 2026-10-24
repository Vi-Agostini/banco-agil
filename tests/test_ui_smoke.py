"""Fumaça da UI Streamlit com LLM inalcançável: valida render, fallback registrado e sem vazamentos."""
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("streamlit")
import streamlit as st  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

from banco_agil import bootstrap

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def fresh_cache(monkeypatch):
    monkeypatch.setattr("banco_agil.processing.sleep_seconds", lambda _: None)
    monkeypatch.setattr(bootstrap, "setup_logging", lambda *_: None)
    st.cache_resource.clear()  # o orquestrador é cacheado por processo
    yield
    st.cache_resource.clear()


def texts(at):
    return [m.markdown[0].value for m in at.chat_message if m.markdown]


def visible_text(at):
    elements = [*at.caption, *at.sidebar.caption, *at.markdown, *at.sidebar.markdown,
                *at.code, *at.sidebar.code, *at.text, *at.sidebar.text]
    return "\n".join([*texts(at), *(element.value for element in elements if hasattr(element, "value"))])


def test_relative_data_dir_is_resolved_from_project_root(tmp_path, monkeypatch):
    captured = {}

    class FakeOrchestrator:
        def __init__(self, repo, *_args, **_kwargs):
            captured["repo"] = repo
            self.credit = SimpleNamespace(reconcile=lambda: None)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATA_DIR", "./data")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setattr(bootstrap, "load_dotenv", lambda *_args: False)
    monkeypatch.setattr(bootstrap, "setup_logging", lambda *_args: None)
    monkeypatch.setattr(bootstrap, "OpenAI", lambda **_kwargs: object())
    monkeypatch.setattr(bootstrap, "Orchestrator", FakeOrchestrator)

    bootstrap.build_orchestrator()

    assert captured["repo"].dir == bootstrap.ROOT / "data"


def test_ui_renders_and_survives_llm_outage(tmp_path, monkeypatch):
    for f in (ROOT / "data").glob("*.csv"):
        shutil.copy(f, tmp_path / f.name)
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("LLM_API_KEY", "chave-falsa-secreta")
    monkeypatch.setenv("LLM_BASE_URL", "http://127.0.0.1:9/v1")  # porta fechada: falha imediata
    monkeypatch.setenv("LLM_MAX_RETRIES", "0")
    monkeypatch.setenv("BANCO_AGIL_ENV", "development")
    monkeypatch.setenv("BANCO_AGIL_ENABLE_DEBUG", "1")
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    assert not at.exception
    assert "Banco Ágil" in texts(at)[0]
    at.chat_input[0].set_value("meu cpf é 111.444.777-35").run()
    if "processing_job" in at.session_state:
        at.session_state["processing_job"].thread.join(timeout=5)
        at.run()
    assert not at.exception
    shown = texts(at)
    assert any("data de nascimento" in t for t in shown)
    assert any("***.***.***-35" in t for t in shown) and not any("111.444.777" in t for t in shown)
    at.chat_input[0].set_value("(12/05/1990)").run()
    if "processing_job" in at.session_state:
        at.session_state["processing_job"].thread.join(timeout=5)
        at.run()
    at.chat_input[0].set_value("quero consultar meu limite").run()
    if "processing_job" in at.session_state:
        at.session_state["processing_job"].thread.join(timeout=5)
        at.run()
    shown = texts(at)
    assert any("indisponibilidade técnica" in t for t in shown)
    at.chat_input[0].set_value(
        "Meu CPF é 11144477735 e nasci em 12/05/1990. Quero pedir aumento para R$ 7.000,00"
    ).run()
    if "processing_job" in at.session_state:
        at.session_state["processing_job"].thread.join(timeout=5)
        at.run()
    assert not at.exception
    session = at.session_state["session"]
    assert "11144477735" not in visible_text(at)
    assert "12/05/1990" not in visible_text(at)
    # AppTest expõe o Markdown, não a página final renderizada pelo navegador.
    assert r"R\$ 7.000,00" in visible_text(at)
    assert any("R$ 7.000,00" in text for _, text in session.transcript)
    assert "11144477735" not in (session.last_llm_payload or "")
    assert "12/05/1990" not in (session.last_llm_payload or "")
    assert "R$ 7.000,00" in (session.last_llm_payload or "")
    assert "chave-falsa-secreta" not in " ".join(shown) and "Traceback" not in " ".join(shown)
    assert at.session_state["session"].history[-1]["role"] == "assistant"  # fallback registrado no histórico
    assert at.session_state["session"].last_failure_reason
    at.sidebar.checkbox[0].check().run()
    assert any("Categoria da última falha" in c.value for c in at.caption)
    rendered = visible_text(at)
    assert "Payload enviado ao LLM" not in rendered
    assert "Último retorno da API" not in rendered
    assert "chave-falsa-secreta" not in rendered
    assert "11144477735" not in rendered and "12/05/1990" not in rendered
    next(b for b in at.sidebar.button if b.label == "Novo atendimento").click().run()
    assert len(texts(at)) == 1  # reset leva de volta à saudação


def test_production_ui_hides_debug_and_demo_credentials(monkeypatch):
    monkeypatch.delenv("BANCO_AGIL_ENABLE_DEBUG", raising=False)
    monkeypatch.setenv("BANCO_AGIL_ENV", "production")
    monkeypatch.setenv("LLM_API_KEY", "delivery-secret")
    monkeypatch.setattr("banco_agil.bootstrap.load_dotenv", lambda *a, **k: False)
    monkeypatch.setattr("banco_agil.bootstrap.OpenAI", lambda **_kwargs: object())
    monkeypatch.setattr("banco_agil.bootstrap.Orchestrator", lambda *_a, **_k: SimpleNamespace(
        new_session=lambda: __import__("banco_agil.session", fromlist=["Session"]).Session(),
        credit=SimpleNamespace(reconcile=lambda: None),
    ))

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()

    assert not at.exception
    rendered = visible_text(at)
    assert not at.sidebar.checkbox
    assert "111.444.777-35" not in rendered
    assert "529.982.247-25" not in rendered
    assert "123.456.789-09" not in rendered
    assert "delivery-secret" not in rendered


def test_ui_without_api_key_shows_config_error(monkeypatch, tmp_path):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setattr("banco_agil.bootstrap.load_dotenv", lambda *a, **k: False)
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    assert any("LLM_API_KEY" in e.value for e in at.error)
