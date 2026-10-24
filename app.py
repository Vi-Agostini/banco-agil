import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import streamlit as st  # noqa: E402

from banco_agil.bootstrap import ConfigError, build_orchestrator  # noqa: E402
from banco_agil.composer import chat_input_controls  # noqa: E402
from banco_agil.processing import ProcessingJob
from banco_agil.presentation import chat_markdown  # noqa: E402

st.set_page_config(page_title="Banco Ágil", page_icon="🏦")
st.title("🏦 Banco Ágil")


@st.cache_resource
def get_orchestrator():
    return build_orchestrator()


try:
    orch = get_orchestrator()
except ConfigError as e:
    st.error(str(e))
    st.stop()

if "session" not in st.session_state:
    st.session_state.session = orch.new_session()
session = st.session_state.session


job = st.session_state.get("processing_job")
if job is not None and job.future.done():
    del st.session_state["processing_job"]
    job = None
busy = job is not None

with st.sidebar:
    if st.button("Novo atendimento", disabled=busy) and not busy:
        st.session_state.session = orch.new_session()
        st.session_state.customer_message = ""
        st.rerun()
    debug_enabled = (
        os.getenv("BANCO_AGIL_ENV", "production").casefold() == "development"
        and os.getenv("BANCO_AGIL_ENABLE_DEBUG") == "1"
    )
    debug = st.checkbox("Diagnóstico técnico local") if debug_enabled else False
    if debug:
        st.write({"agente_ativo": session.active_agent, "autenticado": session.authenticated,
                  "falhas_auth": session.auth_failures})
        st.caption("Categoria da última falha")
        st.text(session.last_failure_category or "Nenhuma falha registrada")
        if session.last_provider_error_code:
            st.caption("Código do provedor")
            st.text(session.last_provider_error_code)
            if session.last_provider_diagnostic:
                st.json(session.last_provider_diagnostic)

visible_transcript = job.visible_transcript if busy else session.transcript
for role, content in visible_transcript:
    with st.chat_message(role):
        st.markdown(chat_markdown(content))

if session.ended and not busy:
    st.info("Atendimento encerrado.")
else:
    def preserve_busy_draft():
        # Uma submissão durante o processamento permanece como rascunho, mesmo em reruns.
        if st.session_state.get("processing_job") is not None:
            # Preenche o campo novamente, sem enviar a mensagem.
            st.session_state.customer_message = st.session_state.customer_message

    prompt = st.chat_input("Digite sua mensagem", key="customer_message", disabled=False,
                           on_submit=preserve_busy_draft)
    if prompt and not busy:
        st.session_state.processing_job = ProcessingJob(orch, session, prompt)
        st.rerun()

if busy:
    @st.fragment(run_every=0.5)
    def processing_status():
        if job.future.done():
            st.rerun(scope="app")
        with st.chat_message("assistant"):
            st.html("""<style>
            .bank-dots span {display:inline-block; animation:bank-pulse 1.2s infinite;}
            .bank-dots span:nth-child(2) {animation-delay:.2s;}
            .bank-dots span:nth-child(3) {animation-delay:.4s;}
            @keyframes bank-pulse {0%,80%,100% {opacity:.2;} 40% {opacity:1;}}
            </style><div role="status" aria-live="polite">
            Aguarde um momento, estou trabalhando nisso.
            <span class="bank-dots" aria-hidden="true"><span>●</span><span>●</span><span>●</span></span>
            </div>""")
    processing_status()

if not session.ended or busy:
    st.html(chat_input_controls(busy), unsafe_allow_javascript=True)
