from __future__ import annotations

from dataclasses import dataclass, field
from uuid import uuid4

from .flow import FlowState
from .processing import ProcessingGuard


@dataclass
class Session:
    """Estado compartilhado de um atendimento (um único histórico para todos os agentes).

    `history` fornece o contexto do LLM e o diagnóstico; só uma janela recente
    é enviada ao provedor. `transcript` contém as mensagens exibidas na UI.
    Ambos mantêm CPF e datas de nascimento mascarados, preservando outras datas.
    """

    processing_guard: ProcessingGuard = field(default_factory=ProcessingGuard, repr=False, compare=False)
    active_agent: str = "triagem"
    authenticated: bool = False
    cpf: str | None = None
    nome: str | None = None
    auth_failures: int = 0
    flow: FlowState = field(default_factory=FlowState)
    ended: bool = False
    desired_limit: float | None = None
    credit_proposal: dict | None = None
    credit_confirmation_turn: int | None = None
    credit_clarification_turn: int | None = None
    credit_operation: dict | None = None
    tool_facts: dict[str, dict] = field(default_factory=dict)
    agent_questions: dict[str, dict[str, str]] = field(default_factory=dict)
    turn: int = 0
    last_request_status: str | None = None
    last_request_id: int | None = None
    automatic_reanalysis_source_id: int | None = None
    automatic_reanalysis_result: dict | None = None
    automatic_reanalysis_response_pending: bool = False
    last_llm_payload: str | None = None
    last_llm_response: str | None = None
    last_failure_reason: str | None = None
    last_failure_category: str | None = None
    last_failure_cause: str | None = None
    last_provider_error_code: str | None = None
    last_generation_json_valid: bool | None = None
    last_provider_diagnostic: dict = field(default_factory=dict)
    last_tool_error_code: str | None = None
    correlation_id: str = field(default_factory=lambda: uuid4().hex)
    test_id: str | None = None
    history: list[dict] = field(default_factory=list)
    transcript: list[tuple[str, str]] = field(default_factory=list)
    llm_mode: str | None = None
    # Só memoriza o transporte após uma recuperação validada, nesta sessão e modelo.
    structured_tool_models: set[str] = field(default_factory=set)

    def record_invalid_input(self) -> int:
        self.flow.unclear_count += 1
        return self.flow.unclear_count
