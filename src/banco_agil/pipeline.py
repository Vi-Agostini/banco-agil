"""Loop híbrido entre o LLM e as decisões determinísticas das ferramentas."""
from __future__ import annotations

import json
import logging
import time
from collections import Counter
from dataclasses import dataclass, field

from .agents import AGENTS
from .interview import missing_interview_fields, next_interview_prompt, has_interview_request
from .fallback import agent_fallback
from .flow import FlowController, TECHNICAL_FAILURE_CODES
from .llm import LlmRetryState, call_llm
from .messages import (
    EMPTY_REPLY,
    LOOP_GUARD_REPLY,
    auth_restart_message,
    loop_guard_reply,
    rejected_limit_offer_message,
    interview_credit_resume_message,
    TRIAGE_CONTINUE,
    TRIAGE_BIRTH_DATE,
)
from .observability import log_event
from .privacy import redact_history
from .reply_recorder import ReplyRecorder
from .response_guard import (confirmed_result_reply, grounded_reply, interview_recovery_reply, pending_summary,
                             unfinished_credit_confirmation_reply)
from .tool_contracts import ToolContractError, credit_clarification_pending, credit_confirmation_pending, validate_batch
from .session import Session
from .statuses import STATUS_REPROVADO
from .tool_executor import ToolExecutor
from .tools import MSG_FAREWELL
from . import structured_output

log = logging.getLogger("banco_agil")
MAX_IDENTICAL_CALLS = 1
MAX_TRANSFERS_PER_TURN = 3
CREDIT_COMPLETION_INSTRUCTION = (
    "A decisão anterior não concluiu o pedido confirmado. Use solicitar_aumento_limite com o novo_limite "
    "da proposta preservada antes de responder ao cliente. Não transfira nem registre novamente a intenção. "
    "Aguarde o resultado real. Não há processamento em segundo plano nem retorno futuro agendado."
)


@dataclass(frozen=True)
class PipelineServices:
    """Dependências usadas pelo pipeline de mensagens e ferramentas."""

    max_steps: int
    max_llm_retries: int
    client: object
    model: str
    tools: ToolExecutor
    replies: ReplyRecorder
    flow: FlowController
    max_rate_limit_wait_seconds: float = 60.0
    llm_mode: str = "native"


@dataclass
class TurnState:
    seen: Counter = field(default_factory=Counter)
    invalid_input_tool_called: bool = False
    transfers: int = 0
    rate_limit_retry_state: LlmRetryState = field(default_factory=LlmRetryState)
    credit_repair_used: bool = False


@dataclass
class BatchOutcome:
    final_message: str | None = None
    repeated_failed_call: bool = False
    rejected_limit: float | None = None
    auth_restart: dict | None = None
    credit_tool_error: dict | None = None
    requested_input_error: tuple[str, dict] | None = None


def run_pipeline(services: PipelineServices, session: Session) -> str:
    """Executa etapas limitadas até obter uma resposta ou um resultado local."""
    turn = TurnState()
    for _ in range(services.max_steps):
        reply = _run_step(services, session, turn)
        if reply is not None:
            return reply
    log.error("Limite de passos excedido (agente=%s)", session.active_agent)
    session.last_failure_reason = (
        f"O limite de {services.max_steps} etapas foi atingido; "
        f"possível ciclo interno (agente={session.active_agent})."
    )
    unfinished = unfinished_credit_confirmation_reply(session)
    clarification = pending_summary(session) if credit_clarification_pending(session) else None
    return services.replies.say(session, unfinished or clarification or LOOP_GUARD_REPLY,
                                track=not bool(unfinished or clarification))


def _run_step(services: PipelineServices, session: Session, turn: TurnState) -> str | None:
    agent = AGENTS[session.active_agent]
    message = call_llm(
        services.client,
        services.model,
        agent,
        session,
        services.max_llm_retries,
        max_rate_limit_wait_seconds=services.max_rate_limit_wait_seconds,
        retry_state=turn.rate_limit_retry_state,
        llm_mode=services.llm_mode,
        decision_instruction=(CREDIT_COMPLETION_INSTRUCTION
                              if turn.credit_repair_used and credit_confirmation_pending(session) else None),
    )
    if message is None:
        log_event(log, "llm_fallback", session, agent=agent.name, operation="chat_completion",
                  fallback_reason=session.last_failure_category or "unknown")
        if (session.last_tool_error_code == "pedido_confirmado_aguarda_analise"
                and _repair_credit_decision(session, turn)):
            return None
        if session.automatic_reanalysis_response_pending and session.last_failure_category != "truncated_response":
            return _automatic_reanalysis_reply(services, session)
        can_present_result = session.last_failure_category in {
            "connection", "timeout", "local_capacity", "http_429", "http_500", "http_502", "http_503", "http_504",
        } or (session.last_failure_category == "http_400" and session.last_failure_cause in {
            "tool_use_failed", "json_validate_failed",
        })
        unfinished = unfinished_credit_confirmation_reply(session)
        confirmed = confirmed_result_reply(session) if (can_present_result or session.last_failure_category in {
            "invalid_tool_response", "invalid_structured_response",
        }) and not unfinished else None
        if unfinished and session.last_failure_category in {"truncated_response", "invalid_structured_response"}:
            unfinished = agent_fallback(agent.name, session, services.tools.execute) + "\n\n" + unfinished
        interview_reply = interview_recovery_reply(session)
        clarification = (pending_summary(session) if credit_clarification_pending(session)
                         and session.last_failure_category in {"invalid_tool_response", "invalid_structured_response"}
                         else None)
        fallback = (unfinished or confirmed or interview_reply or clarification
                    or agent_fallback(agent.name, session, services.tools.execute))
        if confirmed:
            log_event(log, "response_recovery", session, agent=agent.name,
                      operation="confirmed_tool_result", result="success",
                      fallback_reason=session.last_failure_category)
        if pending_summary(session) and not interview_reply and pending_summary(session) not in fallback:
            # Uma falha do provedor ao resumir não pode deixar uma confirmação
            # pendente escondida do cliente.
            fallback += "\n\n" + pending_summary(session)
        return services.replies.say(session, fallback, track=False)
    calls = list(getattr(message, "tool_calls", None) or [])
    raw_content = getattr(message, "content", None)
    content = raw_content.strip() if isinstance(raw_content, str) else ""
    if not calls:
        return _plain_reply(services, session, agent.name, content, turn)
    structured = isinstance(message, structured_output.StructuredMessage)
    if structured:
        structured_output.record_proposal(session, message)
    else:
        _append_assistant_tool_calls(session, raw_content, calls)
    outcome = _execute_batch(services, session, agent, calls, turn, structured=structured)
    if not turn.invalid_input_tool_called:
        session.flow.unclear_count = 0
    return _post_tool_reply(services, session, agent.name, outcome, turn)


def _append_assistant_tool_calls(session: Session, content: str | None, calls: list) -> None:
    # O protocolo distingue uma chamada sem texto (content=null) de uma mensagem
    # com texto vazio. Mantém essa diferença ao enviar o histórico ao próximo agente.
    assistant_content = content if isinstance(content, str) else None
    session.history.append({
        "role": "assistant",
        "content": assistant_content,
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.function.name, "arguments": call.function.arguments or "{}"},
            }
            for call in calls
        ],
    })


def _execute_batch(services: PipelineServices, session: Session, agent, calls: list, turn: TurnState,
                   *, structured: bool = False) -> BatchOutcome:
    outcome = BatchOutcome()
    try:
        validate_batch(agent, session, calls)
    except ToolContractError:
        # Defesa independente para chamadas internas que não passaram pelo LLM.
        for call in calls:
            if structured:
                structured_output.record_result(session, call.function.name,
                    {"erro": "Lote inválido; nenhuma ferramenta executada."})
            else:
                session.history.append({"role": "tool", "tool_call_id": call.id,
                                        "content": '{"erro":"Lote inválido; nenhuma ferramenta executada."}'})
        outcome.final_message = "Não executei esse lote porque os dados ou permissões não conferem."
        outcome.repeated_failed_call = True
        return outcome
    for call in calls:
        name, raw = call.function.name, call.function.arguments or "{}"
        result, repeated = _execute_call(services, session, agent, name, raw, turn, outcome)
        outcome.repeated_failed_call |= repeated
        if structured:
            structured_output.record_result(session, name, result)
        else:
            session.history.append({
                "role": "tool",
                "tool_call_id": call.id,
                "name": name,
                "content": json.dumps(result, ensure_ascii=False),
            })
        if name == "autenticar_cliente":
            redact_history(session.history)
    return outcome


def _execute_call(services: PipelineServices, session, agent, name: str, raw: str, turn: TurnState, outcome: BatchOutcome):
    # Argumentos equivalentes representam a mesma operação, qualquer que seja a ordem do JSON.
    signature = (name, json.dumps(json.loads(raw), sort_keys=True, separators=(",", ":")))
    if session.ended:
        return {"erro": "Atendimento encerrado; chamada não executada."}, False
    if turn.seen[signature] >= MAX_IDENTICAL_CALLS:
        log_event(log, "pipeline_stop", session, agent=agent.name, operation="progress_guard",
                  tool_name=name, result="blocked", fallback_reason="repeated_tool_call")
        return {"erro": "Chamada repetida bloqueada."}, True
    if name == "transferir_para_agente" and turn.transfers >= MAX_TRANSFERS_PER_TURN:
        log_event(log, "pipeline_stop", session, agent=agent.name, operation="progress_guard",
                  tool_name=name, result="blocked", fallback_reason="transfer_limit")
        return {"erro": "Limite de transferências atingido neste atendimento."}, True
    turn.seen[signature] += 1
    turn.transfers += name == "transferir_para_agente"
    previous_agent = session.active_agent
    started = time.perf_counter()
    result = services.tools.execute_from_llm(agent, session, name, raw)
    log_event(log, "tool_execution", session, agent=agent.name, operation="tool_call",
              tool_name=name, result="error" if "erro" in result else "success",
              duration_ms=round((time.perf_counter() - started) * 1000))
    if session.active_agent != previous_agent:
        log_event(log, "agent_transition", session, from_agent=previous_agent,
                  to_agent=session.active_agent, operation="tool_call")
    _record_tool_effects(session, name, result, turn, outcome)
    return result, False


def _record_tool_effects(session, name, result, turn, outcome) -> None:
    if name == "registrar_entrada_invalida":
        turn.invalid_input_tool_called = True
    if name == "autenticar_cliente":
        _record_auth_result(result, outcome)
    if name == "solicitar_aumento_limite":
        _record_credit_result(result, outcome)
    if name == "registrar_intencao_credito" and result.get("codigo") == "entrevista_em_andamento":
        outcome.requested_input_error = ("limit", result)
    if name == "registrar_entrevista_e_calcular_score" and "erro" in result:
        outcome.requested_input_error = ("interview", result)
    if name == "consultar_cotacao" and "erro" in result and result.get("codigo") not in {
            "moeda_nao_identificada", "cotacao_indisponivel"}:
        outcome.requested_input_error = ("currency", result)
    if session.ended:
        outcome.final_message = result.get("mensagem_final", MSG_FAREWELL)


def _record_auth_result(result: dict, outcome: BatchOutcome) -> None:
    if result.get("sucesso") is False:
        if result.get("motivo") != "dados_incompletos" and not result.get("encerrado", False):
            outcome.auth_restart = result


def _record_credit_result(result: dict, outcome: BatchOutcome) -> None:
    if "erro" in result:
        outcome.credit_tool_error = result
        outcome.requested_input_error = ("limit", result)
    if result.get("status_pedido") == STATUS_REPROVADO:
        outcome.rejected_limit = result.get("novo_limite_solicitado")


def _post_tool_reply(services: PipelineServices, session: Session, agent_name: str, outcome: BatchOutcome,
                     turn: TurnState) -> str | None:
    if session.ended:
        return services.replies.say(session, outcome.final_message or MSG_FAREWELL)
    if outcome.credit_tool_error is not None:
        return services.replies.say(
            session,
            services.flow.requested_input_error(session, "limit", outcome.credit_tool_error),
            track=outcome.credit_tool_error.get("codigo") not in TECHNICAL_FAILURE_CODES,
        )
    if outcome.requested_input_error is not None:
        kind, result = outcome.requested_input_error
        if kind == "interview" and session.flow.input_kind and session.flow.input_kind.startswith("interview"):
            kind = session.flow.input_kind
        return services.replies.say(session, services.flow.requested_input_error(session, kind, result),
                                    track=result.get("codigo") not in TECHNICAL_FAILURE_CODES)
    if outcome.auth_restart is not None:
        return services.replies.say(session, auth_restart_message(outcome.auth_restart))
    if outcome.rejected_limit is not None:
        return services.replies.say(session, rejected_limit_offer_message(outcome.rejected_limit))
    if outcome.repeated_failed_call:
        session.last_failure_category = "progress_stalled"
        session.last_failure_reason = (
            f"Um ciclo sem progresso foi bloqueado (agente={agent_name})."
        )
        if _repair_credit_decision(session, turn):
            return None
        result = (unfinished_credit_confirmation_reply(session)
                  or confirmed_result_reply(session) or interview_recovery_reply(session))
        if not result and agent_name == "credito":
            result = (interview_credit_resume_message(session.desired_limit) if has_interview_request(session)
                      else pending_summary(session))
        if not result and agent_name == "triagem":
            if session.authenticated:
                proposal = session.credit_proposal or {}
                if proposal and not proposal.get("executada") and session.cpf:
                    # Retoma a proposta por uma transferência real, dentro do mesmo limite do turno.
                    # Isso não autoriza operação financeira nem repete a coleta.
                    handoff, _ = _execute_call(services, session, AGENTS["triagem"],
                        "transferir_para_agente", '{"destino":"credito"}', turn, outcome)
                    structured_output.record_result(session, "transferir_para_agente", handoff)
                    if handoff.get("ok"):
                        result = (interview_credit_resume_message(session.desired_limit)
                                  if has_interview_request(session) else pending_summary(session))
                result = result or TRIAGE_CONTINUE
            elif session.flow.pending_cpf:
                result = TRIAGE_BIRTH_DATE
        if not result and agent_name == "cambio":
            fact = session.tool_facts.get("consultar_cotacao", {})
            failure = fact.get("result", {})
            if fact.get("turn") == session.turn and failure.get("codigo") in {"cotacao_indisponivel", "moeda_nao_identificada"}:
                result = failure["erro"]
        result = result or loop_guard_reply(agent_name)
        if pending_summary(session) and not session.flow.pending_interview_answers and pending_summary(session) not in result:
            result += "\n\n" + pending_summary(session)
        return services.replies.say(session, result, track=False)
    return None


def _repair_credit_decision(session: Session, turn: TurnState) -> bool:
    if not credit_confirmation_pending(session) or turn.credit_repair_used:
        return False
    turn.credit_repair_used = True
    log_event(log, "response_recovery", session, agent=session.active_agent,
              operation="credit_decision", result="retrying", fallback_reason="confirmed_request_not_executed")
    return True


def _plain_reply(services: PipelineServices, session: Session, agent_name: str, content: str, turn: TurnState) -> str | None:
    if credit_confirmation_pending(session):
        # A operação continua dependendo da decisão do agente; uma promessa
        # no texto não inicia processamento em segundo plano.
        if _repair_credit_decision(session, turn):
            return None
        session.last_failure_category = "progress_stalled"
        session.last_failure_reason = "A IA não concluiu o pedido confirmado dentro do limite de recuperação."
        return services.replies.say(session, unfinished_credit_confirmation_reply(session), track=False)
    if not turn.invalid_input_tool_called:
        session.flow.unclear_count = 0
    if not content:
        session.last_failure_category = "empty_response"
        session.last_failure_reason = "O LLM retornou uma resposta vazia, sem texto nem chamada de ferramenta."
        log.warning("Resposta vazia do LLM (agente=%s)", agent_name)
        if credit_clarification_pending(session):
            return services.replies.say(session, pending_summary(session), track=False)
        if agent_name == "credito" and session.automatic_reanalysis_response_pending:
            return _automatic_reanalysis_reply(services, session)
        return services.replies.say(session, EMPTY_REPLY)
    if agent_name == "credito" and session.automatic_reanalysis_response_pending:
        session.automatic_reanalysis_response_pending = False
    if "instabilidade" in content.casefold():
        session.last_failure_reason = "A mensagem de instabilidade veio no texto do LLM; a chamada à API foi concluída."
    if (agent_name == "entrevista" and session.flow.interview_question_pending
            and not session.flow.pending_interview_answers
            and missing_interview_fields(session.flow.interview_answers) and "?" not in content):
        # Um agradecimento sozinho não deixa o cliente esperando a próxima pergunta.
        # A pergunta usual vem da IA; esta recuperação cobre a omissão sem nova
        # chamada ao provedor ou alteração de dados.
        content += "\n\n" + next_interview_prompt(session.flow.interview_answers, {})
        log_event(log, "response_recovery", session, agent=agent_name,
                  operation="interview_question", result="completed", fallback_reason="missing_interview_question")
    session.flow.interview_question_pending = False
    checked = grounded_reply(session, content)
    if checked != content:
        log_event(log, "response_validation", session, agent=agent_name,
                  operation="financial_facts", result="replaced", error_category="ungrounded_response")
    elif (pending_summary(session) and pending_summary(session) not in checked
          and not credit_clarification_pending(session)):
        # A pergunta continua sendo da IA. Os dados que autorizam a operação
        # são apresentados diretamente do estado validado, sem valores inventados.
        checked += "\n\n" + pending_summary(session)
    quote_fact = session.tool_facts.get("consultar_cotacao", {})
    quote_unavailable = (agent_name == "cambio" and quote_fact.get("turn") == session.turn
                         and quote_fact.get("result", {}).get("codigo") in {
                             "moeda_nao_identificada", "cotacao_indisponivel"})
    # Falhas repetidas do provedor não provam que o cliente ficou preso no fluxo.
    # Mensagens iguais de indisponibilidade não encerram a sessão.
    return services.replies.say(session, checked,
                                track=not (quote_unavailable or credit_clarification_pending(session)))


def _automatic_reanalysis_reply(services: PipelineServices, session: Session) -> str:
    """Comunica uma reanálise já concluída sem repetir a operação no fallback."""
    session.automatic_reanalysis_response_pending = False
    result = session.automatic_reanalysis_result or {}
    if "erro" in result:
        return services.replies.say(
            session,
            "Seu score foi atualizado, mas não consegui concluir a reanálise automaticamente. "
            "Por segurança, não repeti a solicitação. Você pode consultar seu limite ou fazer um novo pedido quando quiser.",
            track=False,
        )
    from .messages import limit_request_reply
    return services.replies.say(session, limit_request_reply(result), track=False)
