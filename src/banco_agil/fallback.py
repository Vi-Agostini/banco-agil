"""Respostas determinísticas quando o provedor de linguagem está indisponível."""
from __future__ import annotations

import json
from collections.abc import Callable

from .agents import AGENTS
from .flow import Awaiting, FlowController
from .intents import extract_credentials
from .interview import next_interview_prompt
from .messages import (
    TRIAGE_AUTHENTICATED,
    TRIAGE_BIRTH_DATE,
    TRIAGE_CPF,
    LLM_INPUT_TOO_LONG,
    LLM_UNAVAILABLE,
    LLM_RATE_LIMITED,
    agent_fallback_message,
    auth_restart_message,
)
from .privacy import redact_history
from .session import Session


def agent_fallback(agent_name: str, session: Session, execute: Callable) -> str:
    if session.last_failure_category in {"truncated_response", "invalid_structured_response", "invalid_request", "local_capacity"}:
        return LLM_UNAVAILABLE
    if session.last_failure_category == "input_too_long":
        return LLM_INPUT_TOO_LONG
    if agent_name == "credito" and session.last_failure_category == "invalid_tool_response":
        return FlowController().requested_input_error(
            session, "limit", {"codigo": session.last_tool_error_code or "argumentos_invalidos"},
        )
    if (agent_name == "entrevista" and session.authenticated and session.flow.interview_started
            and session.last_failure_category in {"http_400", "invalid_tool_response"}):
        return next_interview_prompt(
            session.flow.interview_answers, session.flow.pending_interview_answers,
        )
    if session.authenticated:
        if session.last_failure_category == "http_429":
            return LLM_RATE_LIMITED
        return LLM_UNAVAILABLE
    if agent_name != "triagem":
        return agent_fallback_message(agent_name)
    latest = next(
        (item.get("content", "") for item in reversed(session.history) if item.get("role") == "user"),
        "",
    )
    credentials = extract_credentials(latest, awaiting_birth_date=session.flow.awaiting is Awaiting.BIRTH_DATE)
    cpf = credentials.cpf or session.flow.pending_cpf
    if cpf and credentials.birth_date:
        return _fallback_authenticate(session, execute, cpf, credentials.birth_date)
    if credentials.cpf:
        session.flow.pending_cpf = credentials.cpf
        session.flow.awaiting = Awaiting.BIRTH_DATE
        return TRIAGE_BIRTH_DATE
    return TRIAGE_CPF


def _fallback_authenticate(session: Session, execute: Callable, cpf: str, birth_date: str) -> str:
    result = execute(
        AGENTS["triagem"],
        session,
        "autenticar_cliente",
        json.dumps({"cpf": cpf, "data_nascimento": birth_date}, ensure_ascii=False),
    )
    redact_history(session.history)
    session.flow.pending_cpf = None
    if session.flow.awaiting is Awaiting.BIRTH_DATE:
        session.flow.awaiting = Awaiting.NONE
    if result.get("encerrado"):
        return result["mensagem_final"]
    if result.get("sucesso"):
        session.flow.unclear_count = 0
        return TRIAGE_AUTHENTICATED.format(name=session.nome)
    return auth_restart_message(result)
