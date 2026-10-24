"""Registra as respostas e controla repetições no atendimento."""
from __future__ import annotations

from .intents import normalize_text
from .messages import REPEAT_GUARD_REPLY
from .session import Session
from .agent_context import remember_question
from .presentation import customer_reply


class ReplyRecorder:
    """Guarda as mensagens do assistente e aplica o limite de respostas repetidas."""

    def say(self, session: Session, text: str, track: bool = True) -> str:
        text = customer_reply(session, text)
        if not track:
            session.flow.assistant_reply_states.clear()
            session.history.append({"role": "assistant", "content": text})
            return text
        normalized = normalize_text(text)
        signature = self.flow_signature(session)
        recent = session.flow.assistant_reply_states[-2:]
        if (
            not session.ended
            and len(recent) == 2
            and all(reply == normalized and state == signature for reply, state in recent)
        ):
            session.ended = True
            text = REPEAT_GUARD_REPLY
            normalized = normalize_text(text)
            signature = self.flow_signature(session)
        session.flow.assistant_reply_states.append((normalized, signature))
        remember_question(session, text)
        session.history.append({"role": "assistant", "content": text})
        return text

    @staticmethod
    def flow_signature(session: Session) -> tuple:
        flow = session.flow
        return (
            session.active_agent,
            session.authenticated,
            session.auth_failures,
            flow.unclear_count,
            flow.awaiting,
            flow.pending_cpf,
            flow.input_kind,
            flow.attempts,
            tuple(sorted(flow.interview_answers.items())),
            tuple(sorted(flow.pending_interview_answers.items())),
            flow.interview_revision_requested,
            session.desired_limit,
            session.last_request_status,
            session.last_request_id,
            session.automatic_reanalysis_source_id,
            session.automatic_reanalysis_response_pending,
        )
