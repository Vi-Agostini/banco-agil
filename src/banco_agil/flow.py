"""Estado coeso da máquina de fluxo de uma conversa."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from .messages import (
    CURRENCY_RETRY_PROMPT,
    INTERVIEW_RETRY_PROMPT,
    LIMIT_RETRY_PROMPT,
    LLM_UNAVAILABLE,
    requested_input_error,
    interview_credit_resume_message,
)
from .statuses import STATUS_APROVADO, STATUS_REPROVADO

if TYPE_CHECKING:
    from .session import Session


class Awaiting(str, Enum):
    NONE = "none"
    INTERVIEW_OFFER = "interview_offer"
    LIMIT_AMOUNT = "limit_amount"
    BIRTH_DATE = "birth_date"
    CLOSE_CONFIRM = "close_confirm"


TECHNICAL_FAILURE_CODES = {"dados_indisponiveis", "erro_tecnico", "pedido_estado_incerto", "reanalise_indisponivel"}


@dataclass
class FlowState:
    """Guarda os dados transitórios necessários para conduzir o atendimento."""

    awaiting: Awaiting = Awaiting.NONE
    attempts: int = 0
    pending_cpf: str | None = None
    unclear_count: int = 0
    input_kind: str | None = None
    input_prompt: str | None = None
    interview_answers: dict[str, object] = field(default_factory=dict)
    pending_interview_answers: dict[str, object] = field(default_factory=dict)
    interview_revision_requested: bool = False
    interview_started: bool = False
    interview_question_pending: bool = False
    assistant_reply_states: list[tuple[str, tuple]] = field(default_factory=list)


def apply_tool_effects(session: Session, name: str, result: dict) -> None:
    """Atualiza o fluxo após a execução bem-sucedida de uma ferramenta."""
    if name == "autenticar_cliente":
        if result.get("sucesso") or result.get("motivo") != "dados_incompletos":
            session.flow.pending_cpf = None
            if session.flow.awaiting is Awaiting.BIRTH_DATE:
                session.flow.awaiting = Awaiting.NONE
    elif name == "solicitar_aumento_limite":
        if "erro" in result:
            if result.get("codigo") in TECHNICAL_FAILURE_CODES | {"entrevista_em_andamento"}:
                return  # Falha do provedor ou da gravação não conta como resposta inválida do cliente.
            if session.flow.input_kind != "limit":
                session.flow.attempts = 0
            session.flow.awaiting = Awaiting.LIMIT_AMOUNT
            session.flow.input_kind = "limit"
            session.flow.attempts += 1
            session.flow.input_prompt = LIMIT_RETRY_PROMPT
        elif result.get("status_pedido") == STATUS_APROVADO:
            _clear_requested_input(session)
        elif result.get("status_pedido") == STATUS_REPROVADO:
            _clear_requested_input(session)
            _clear_interview_answers(session)
            session.flow.awaiting = Awaiting.INTERVIEW_OFFER
    elif name == "registrar_entrevista_e_calcular_score":
        if "erro" not in result and "novo_score" in result:
            _clear_requested_input(session)
            _clear_interview_answers(session)
            reanalysis = result.get("reanalise_automatica") or {}
            if reanalysis.get("status_pedido") == STATUS_REPROVADO:
                session.flow.awaiting = Awaiting.INTERVIEW_OFFER
    elif name == "consultar_cotacao":
        return
    elif name == "transferir_para_agente":
        if result.get("ok"):
            _clear_requested_input(session)
            session.flow.pending_cpf = None
            # Mudar de agente não cancela a entrevista: dados e propostas confirmados
            # sobrevivem a uma consulta em outro agente na mesma entrevista.
            if session.active_agent == "entrevista":
                session.flow.interview_started = True
    elif name == "encerrar_atendimento":
        _clear_requested_input(session)
        session.flow.pending_cpf = None
        _clear_interview_answers(session)
        session.credit_proposal = None


def _clear_requested_input(session: Session) -> None:
    session.flow.awaiting = Awaiting.NONE
    session.flow.attempts = 0
    session.flow.input_kind = None
    session.flow.input_prompt = None


def _clear_interview_answers(session: Session) -> None:
    session.flow.interview_answers.clear()
    session.flow.pending_interview_answers.clear()
    session.flow.interview_revision_requested = False
    session.flow.interview_started = False
    session.flow.interview_question_pending = False


class FlowController:
    """Controla novas tentativas de coleta e as mensagens de erro ao cliente."""

    @staticmethod
    def apply_tool_effects(session: Session, name: str, result: dict) -> None:
        apply_tool_effects(session, name, result)

    @staticmethod
    def clear_requested_input(session: Session) -> None:
        flow = session.flow
        flow.input_kind = None
        flow.input_prompt = None
        flow.attempts = 0
        if flow.awaiting in (Awaiting.LIMIT_AMOUNT, Awaiting.INTERVIEW_OFFER, Awaiting.CLOSE_CONFIRM):
            flow.awaiting = Awaiting.NONE

    @staticmethod
    def finish_interview(session: Session) -> None:
        _clear_interview_answers(session)

    @staticmethod
    def start_interview(session: Session) -> None:
        """Inicia uma coleta nova, sem reutilizar dados de entrevistas anteriores."""
        _clear_interview_answers(session)
        session.flow.interview_started = True
        session.flow.input_kind = "interview"
        session.flow.input_prompt = None
        session.flow.attempts = 0

    def requested_input_error(self, session: Session, kind: str, result: dict) -> str:
        flow = session.flow
        if result.get("codigo") == "entrevista_em_andamento":
            return interview_credit_resume_message(session.desired_limit)
        if kind == "currency" and result.get("codigo") in {"moeda_nao_identificada", "cotacao_indisponivel"}:
            return result.get("erro") or "Não consegui consultar essa cotação agora. Qual outra moeda deseja consultar?"
        if result.get("codigo") in TECHNICAL_FAILURE_CODES:
            return LLM_UNAVAILABLE
        if kind.startswith("interview") and result.get("codigo") == "entrevista_aguardando_confirmacao":
            flow.input_kind = "interview_confirmation"
            flow.input_prompt = "Confirme os dados resumidos antes de calcular o score."
            return flow.input_prompt
        if kind.startswith("interview") and result.get("codigo") == "entrevista_incompleta":
            flow.input_kind = "interview"
            flow.input_prompt = "Ainda faltam informações confirmadas da entrevista."
            flow.attempts = 0
            return flow.input_prompt
        if kind != "limit":
            if flow.input_kind != kind:
                flow.attempts = 0
            flow.input_kind = kind
            flow.attempts += 1
        attempt = flow.attempts
        if attempt > 3:
            self.clear_requested_input(session)
            session.active_agent = "triagem"
        if kind == "limit" and attempt <= 3:
            flow.awaiting = Awaiting.LIMIT_AMOUNT
            flow.input_kind = "limit"
            flow.input_prompt = LIMIT_RETRY_PROMPT
        elif kind.startswith("interview") and attempt <= 3:
            flow.input_prompt = flow.input_prompt or INTERVIEW_RETRY_PROMPT
        elif kind == "currency" and attempt <= 3:
            flow.input_prompt = CURRENCY_RETRY_PROMPT
        return requested_input_error(kind, attempt, result.get("codigo", "erro_tecnico"), flow.input_prompt)
