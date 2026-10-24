"""Eventos de diagnóstico sem conteúdo da conversa."""
from __future__ import annotations

import json
import logging

from .session import Session

_SAFE_FIELDS = {
    "agent", "operation", "attempt", "duration_ms", "result", "status_http",
    "admission_wait_ms", "http_ms", "http_attempted",
    "model", "failure_cause", "recovery_cause",
    "capacity_reason", "generation_status", "schema_issues", "schema_sha256",
    "provider_error_code", "generation_json_valid", "output_mode", "max_completion_tokens",
    "structured_strict", "generation_phase", "structured_transport",
    "retry_block_reason",
    "exception_type", "error_category", "fallback_reason", "tool_name",
    "from_agent", "to_agent",
    "finish_reason", "input_tokens", "output_tokens", "reported_tokens", "cached_input_tokens",
}


def log_event(logger: logging.Logger, event: str, session: Session, **fields) -> None:
    """Registra apenas os metadados permitidos, sem texto ou argumentos da conversa."""
    record = {
        "event": event,
        "correlation_id": session.correlation_id,
        "test_id": session.test_id,
        "turn": session.turn,
        **{key: value for key, value in fields.items() if key in _SAFE_FIELDS},
    }
    logger.info(json.dumps(record, ensure_ascii=True, separators=(",", ":")))
