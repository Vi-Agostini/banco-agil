"""Entrada do atendimento e criação de sessões."""
from __future__ import annotations

import logging
from pathlib import Path

from .credit_service import CreditService
from .exchange import ExchangeService
from .flow import Awaiting, FlowController
from .interceptors import ConversationInterceptors
from .intents import extract_credentials
from .messages import CLOSED, GREETING
from .pipeline import PipelineServices, run_pipeline
from .privacy import has_birth_marker, has_unrelated_date_context, redact_text
from .reply_recorder import ReplyRecorder
from .repository import BankRepository
from .session import Session
from .structured_output import validate_mode
from .tool_executor import ToolExecutor

log = logging.getLogger("banco_agil")


def setup_logging(log_dir: Path) -> None:
    if log.handlers:
        return
    log_dir.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_dir / "errors.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(handler)
    log.setLevel(logging.INFO)


class Orchestrator:
    """Coordena a sessão, os tratamentos locais e o pipeline da IA."""

    def __init__(self, repo: BankRepository, fx: ExchangeService, client, model: str, max_steps: int = 8,
                 max_llm_retries: int = 0, test_id: str | None = None,
                 max_rate_limit_wait_seconds: float = 60.0,
                 *, llm_mode: str = "native"):
        self.llm_mode = validate_mode(llm_mode)
        credit = CreditService(repo)
        self.repo, self.fx = repo, fx
        self.credit = credit
        self.tool_executor = ToolExecutor(repo, fx, credit)
        self.reply_recorder = ReplyRecorder()
        self.flow = FlowController()
        self.client, self.model, self.max_steps = client, model, max_steps
        self.max_llm_retries, self.test_id = max_llm_retries, test_id
        self.max_rate_limit_wait_seconds = max_rate_limit_wait_seconds
        self.interceptors = ConversationInterceptors(self.tool_executor, self.reply_recorder, self.flow)

    def new_session(self) -> Session:
        session = Session(test_id=self.test_id)
        session.history.append({"role": "assistant", "content": GREETING})
        session.transcript.append(("assistant", GREETING))
        return session

    def handle(self, session: Session, text: str, *, before_reply=None) -> str:
        from .processing import BUSY
        if not session.processing_guard.lock.acquire(blocking=False):
            return BUSY
        try:
            return self._handle(session, text)
        finally:
            try:
                if before_reply is not None:
                    before_reply()
            finally:
                session.processing_guard.lock.release()

    def _handle(self, session: Session, text: str) -> str:
        if session.llm_mode is not None and session.llm_mode != self.llm_mode:
            raise ValueError("Não altere o modo LLM de uma sessão existente.")
        session.llm_mode = self.llm_mode
        if session.ended:
            return CLOSED
        session.turn += 1
        session.flow.interview_question_pending = False
        text = text if isinstance(text, str) else ""
        # O texto original é usado apenas para interpretar credenciais e controles locais.
        # O estado armazenado permanece mascarado, inclusive após a autenticação.
        credentials = extract_credentials(text, awaiting_birth_date=session.flow.awaiting is Awaiting.BIRTH_DATE)
        redact_all_dates = session.flow.awaiting is Awaiting.BIRTH_DATE or (
            credentials.cpf is not None
            and credentials.birth_date is not None
            and not has_birth_marker(text)
            and not has_unrelated_date_context(text)
        )
        safe_text = redact_text(text, redact_birth_dates=redact_all_dates)
        session.transcript.append(("user", safe_text))
        session.history.append({"role": "user", "content": safe_text})
        reply = self.interceptors.try_handle(session, text)
        if reply is None:
            services = PipelineServices(
                self.max_steps, self.max_llm_retries, self.client, self.model,
                self.tool_executor, self.reply_recorder, self.flow,
                max_rate_limit_wait_seconds=self.max_rate_limit_wait_seconds,
                llm_mode=self.llm_mode,
            )
            reply = run_pipeline(services, session)
        session.transcript.append(("assistant", reply))
        return reply

