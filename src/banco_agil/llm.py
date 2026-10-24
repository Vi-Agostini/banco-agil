"""Adaptação do cliente OpenAI e atualização dos diagnósticos da sessão."""
from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass
from email.utils import parsedate_to_datetime

from .session import Session
from .provider_diagnostics import generation_report
from .rate_limit import CapacityError, GovernedClient, daily_token_limit
from .tool_contracts import ToolContractError, credit_confirmation_pending, schemas_for, validate_batch
from .observability import log_event
from . import structured_output
from .agent_context import select_history

log = logging.getLogger("banco_agil")
MAX_CONTEXT_USER_TURNS = 2
MIN_RATE_LIMIT_WAIT_SECONDS = 20.0
MAX_RATE_LIMIT_WAIT_SECONDS = 60.0
MAX_CONFIGURED_RATE_LIMIT_WAIT_SECONDS = 120.0
SAFE_TURN_TOKEN_BUDGET = 5_000
RESPONSE_TOKEN_RESERVE = 750
CONTINUATION_TOKEN_RESERVE = 750
MAX_INPUT_TOKENS = SAFE_TURN_TOKEN_BUDGET - RESPONSE_TOKEN_RESERVE - CONTINUATION_TOKEN_RESERVE
ESTIMATED_CHARS_PER_TOKEN = 3
_SKIP_LLM_CONTEXT = "_skip_llm_context"


class InvalidToolResponse(ValueError):
    """Resposta rejeitada antes de registrar histórico ou executar ferramentas."""


@dataclass
class LlmRetryState:
    """Orçamento de retentativas consumido durante uma mensagem do cliente."""

    rate_limit_retry_used: bool = False
    generation_retry_used: bool = False
    admission_deadline: float | None = None


def configured_rate_limit_wait_seconds(value: object) -> float:
    """Limita a configuração de espera ao intervalo seguro."""
    try:
        seconds = float(value) if value is not None else MAX_RATE_LIMIT_WAIT_SECONDS
    except (TypeError, ValueError):
        return MAX_RATE_LIMIT_WAIT_SECONDS
    if not math.isfinite(seconds):
        return MAX_RATE_LIMIT_WAIT_SECONDS
    return min(MAX_CONFIGURED_RATE_LIMIT_WAIT_SECONDS, max(MIN_RATE_LIMIT_WAIT_SECONDS, seconds))


def call_llm(client, model: str, agent, session: Session, max_retries: int = 0, *,
             max_rate_limit_wait_seconds: float = MAX_RATE_LIMIT_WAIT_SECONDS,
             retry_state: LlmRetryState | None = None, llm_mode: str = "native",
             decision_instruction: str | None = None):
    """Consulta o modelo configurado e atualiza os diagnósticos da sessão."""
    structured_output.validate_mode(llm_mode)
    if session.llm_mode is not None and session.llm_mode != llm_mode:
        raise ValueError("Não altere o modo LLM de uma sessão existente.")
    session.llm_mode = llm_mode
    session.last_provider_error_code = None
    session.last_generation_json_valid = None
    session.last_provider_diagnostic = {}
    session.last_failure_cause = None
    tools = _tool_schemas(agent, session)
    presentation_only = (llm_mode == "structured" and not tools
                         and agent.name == "entrevista" and session.flow.interview_started)
    payload = _build_payload(agent, session, presentation_only=presentation_only)
    if decision_instruction:
        # A correção é temporária e vem da aplicação; saídas rejeitadas não entram
        # no histórico e texto do cliente não vira instrução de sistema.
        payload[1] = {**payload[1], "content": payload[1]["content"] + "\n" + decision_instruction}
    base_payload = payload
    format_spec = None
    transport_request = None
    # Mantém o estado das sessões em cache que ainda não têm esse campo.
    structured_tool_models = getattr(session, "structured_tool_models", set())
    if llm_mode == "structured":
        try:
            format_spec = structured_output.response_format(tools)
            tool_transport = model in structured_tool_models
            payload = structured_output.build_payload(base_payload, tools, tool_transport=tool_transport)
            if tool_transport:
                transport_request = structured_output.recovery_request(format_spec)
        except structured_output.StructuredOutputError as error:
            _record_failure(error, session, "invalid_request", None)
            session.last_failure_cause = "invalid_request"
            log_event(log, "llm_call", session, agent=agent.name, operation="chat_completion",
                      attempt=0, result="blocked", error_category="invalid_request",
                      failure_cause="invalid_request", model=_diagnostic_model(model))
            return None
    estimated_tokens = _estimate_request_tokens(
        payload, tools if format_spec is None else [],
        format_spec if transport_request is None else None, request_options=transport_request,
    )
    request_metadata = {} if format_spec is None else {
        "structured_strict": format_spec["json_schema"]["strict"],
        "generation_phase": "message_only" if not tools else "decision",
        "structured_transport": "json_schema" if transport_request is None else "strict_tool",
    }
    if estimated_tokens > MAX_INPUT_TOKENS:
        _record_input_too_long(session, estimated_tokens)
        _exclude_largest_user_message_in_context(session)
        return None
    session.last_llm_payload = json.dumps(payload, ensure_ascii=False, indent=2)
    # Retentativas genéricas são opcionais; o 429 tem orçamento próprio por turno.
    retry_budget = min(1, max(0, max_retries)) if llm_mode == "native" else 0
    max_rate_limit_wait_seconds = configured_rate_limit_wait_seconds(max_rate_limit_wait_seconds)
    retry_state = retry_state or LlmRetryState()
    if isinstance(client, GovernedClient) and retry_state.admission_deadline is None:
        retry_state.admission_deadline = client.limiter.clock() + min(client.max_wait, max_rate_limit_wait_seconds)
    recovered_cause = None
    attempt = 0
    while True:
        attempt += 1
        started = time.perf_counter()
        timings = {}
        try:
            session.last_llm_payload = json.dumps(payload, ensure_ascii=False, indent=2)
            request = dict(
                model=model,
                messages=payload,
                temperature=0.2,
            )
            if transport_request is not None:
                request.update(transport_request)
            elif format_spec is not None:
                request["response_format"] = format_spec
            elif tools:
                request["tools"] = tools
            if isinstance(client, GovernedClient):
                response = client.create(request, estimated_tokens, retry_state.admission_deadline, timings=timings)
            else:
                response = client.chat.completions.create(**request)
            if getattr(response.choices[0], "finish_reason", None) == "length":
                # Mesmo com HTTP 200, uma geração pode estar incompleta. Nunca executa
                # ou reaproveita chamadas parciais de ferramentas.
                usage = getattr(response, "usage", None)
                tokens = {}
                for source, target in (("prompt_tokens", "input_tokens"),
                                       ("completion_tokens", "output_tokens"),
                                       ("total_tokens", "reported_tokens")):
                    value = usage.get(source) if isinstance(usage, Mapping) else getattr(usage, source, None)
                    if type(value) is int and value >= 0:
                        tokens[target] = value
                session.last_provider_diagnostic = {}
                session.last_provider_error_code = None
                session.last_generation_json_valid = None
                session.last_failure_cause = "truncated_response"
                session.last_failure_category = "truncated_response"
                session.last_failure_reason = "A geração do provedor foi interrompida pelo limite de saída."
                session.last_tool_error_code = None
                session.last_llm_response = json.dumps({"status_http": 200, "finish_reason": "length", **tokens})
                log_event(log, "llm_call", session, agent=agent.name, operation="chat_completion",
                          attempt=attempt, duration_ms=round((time.perf_counter() - started) * 1000),
                          result="incomplete", status_http=200, finish_reason="length",
                          error_category="truncated_response", model=_diagnostic_model(model), **request_metadata, **timings, **tokens)
                return None  # O truncamento é bloqueante, mesmo com retentativas genéricas habilitadas.
            message = response.choices[0].message
            if format_spec is not None:
                decoder = structured_output.decode if transport_request is None else structured_output.decode_transport
                message = decoder(message, tools, getattr(response.choices[0], "finish_reason", None))
            _validate_tool_response(message, agent, session, tools)
            if transport_request is not None:
                structured_tool_models.add(model)
                session.structured_tool_models = structured_tool_models
            message = _record_success(message, agent.name, session)
            log_event(log, "llm_call", session, agent=agent.name, operation="chat_completion",
                      attempt=attempt, duration_ms=round((time.perf_counter() - started) * 1000),
                      result="success", status_http=None, model=_diagnostic_model(model),
                      recovery_cause=recovered_cause, **request_metadata, **timings, **_usage_metrics(response))
            return message
        except Exception as error:  # Trata falhas do provedor, de conexão, HTTP e de formato da resposta.
            category, status = _classify_error(error)
            daily_limit = daily_token_limit(error)
            if status == 429 and isinstance(client, GovernedClient):
                client.limiter.defer(_retry_after_seconds(error, float("inf")))
            cause = "daily_token_limit" if daily_limit else _recovery_cause(error, category, status)
            retry_generation = (not daily_limit
                                and cause in {"tool_use_failed", "json_validate_failed"}
                                and not retry_state.generation_retry_used)
            if daily_limit or retry_generation:
                retry_delay = None
            elif status == 429:
                retry_delay = _rate_limit_retry_delay(
                    error, 1, retry_state, max_rate_limit_wait_seconds,
                )
            # A recuperação de geração não amplia as retentativas genéricas.
            # Um 429 continua com seu orçamento independente e limitado.
            elif retry_state.generation_retry_used:
                retry_delay = None
            else:
                retry_delay = _retry_delay(category, status, attempt, retry_budget)
            retry = retry_generation or retry_delay is not None
            _record_failure(error, session, category, status)
            session.last_failure_cause = cause
            if status == 400:
                session.last_provider_diagnostic = generation_report(error, format_spec)
            log_event(log, "llm_call", session, agent=agent.name, operation="chat_completion",
                      attempt=attempt, duration_ms=round((time.perf_counter() - started) * 1000),
                      result="retrying" if retry else "error", status_http=status,
                      exception_type=type(error).__name__, error_category=category,
                      failure_cause=cause, model=_diagnostic_model(model),
                      **session.last_provider_diagnostic, **request_metadata, **timings,
                      capacity_reason=str(error) if isinstance(error, CapacityError) and str(error) in {"request_exceeds_tpm", "admission_deadline"} else None,
                      provider_error_code=session.last_provider_error_code,
                      generation_json_valid=session.last_generation_json_valid,
                      output_mode=llm_mode,
                      retry_block_reason=("daily_token_limit" if daily_limit else "rate_limit_retry_exhausted" if retry_state.rate_limit_retry_used
                                          else "retry_after_exceeds_wait_limit") if status == 429 and not retry else None,
                      max_completion_tokens=client.output_tokens if isinstance(client, GovernedClient) else None)
            if not retry:
                return None
            if retry_generation:
                # Nenhuma saída rejeitada chega ao histórico ou ao executor financeiro.
                # Regenera uma vez com instrução fixa, sem repetir a saída rejeitada.
                retry_state.generation_retry_used = True
                recovered_cause = cause
                instruction = (
                    "\nGere novamente somente a mensagem completa no formato exigido. "
                    "Use tipo=mensagem e acao=null; redija a pergunta ou o resumo a partir do estado atual."
                    if format_spec is not None and not tools else
                    "\nGere novamente uma resposta completa conforme o schema oferecido. "
                    "Use somente ações oferecidas nesta geração, sem repetir resultados locais ou ações antigas. "
                    "Para perguntar, use tipo=mensagem e acao=null. "
                    "Inclua todas as chaves required dos argumentos; campos opcionais sem resposta nova são null. "
                    "Não chame ferramentas nativas."
                    if format_spec is not None else
                    "\nNenhuma ferramenta está disponível nesta etapa. Gere somente o texto da mensagem, sem chamadas."
                    if not tools else
                    "\nGere novamente usando somente ferramentas oferecidas e argumentos válidos."
                )
                if (format_spec is not None and (transport_request is not None or
                        cause == "tool_use_failed" and model in {"openai/gpt-oss-20b", "openai/gpt-oss-120b"})):
                    # A função transporta o mesmo envelope estrito; ela própria não executa
                    # uma ferramenta financeira.
                    transport_request = structured_output.recovery_request(format_spec)
                    payload = structured_output.build_payload(base_payload, tools, tool_transport=True)
                    payload[0] = {**payload[0], "content": payload[0]["content"] +
                        "\nGere novamente o envelope completo conforme o schema, preservando o estado atual e os resultados locais."}
                    request_metadata["structured_transport"] = "strict_tool"
                else:
                    payload = [{**payload[0], "content": payload[0]["content"] + instruction}, *payload[1:]]
                estimated_tokens = _estimate_request_tokens(
                    payload, tools if format_spec is None else [],
                    format_spec if transport_request is None else None, request_options=transport_request,
                )
                if estimated_tokens > MAX_INPUT_TOKENS:
                    _record_input_too_long(session, estimated_tokens)
                    log_event(log, "llm_recovery", session, agent=agent.name, attempt=attempt,
                              result="blocked", failure_cause="input_too_long", model=_diagnostic_model(model))
                    return None
                continue
            if retry_delay is not None and not (status == 429 and isinstance(client, GovernedClient)):
                time.sleep(retry_delay)


def _validate_tool_response(message, agent, session, tools) -> None:
    """Valida o lote inteiro antes de permitir qualquer operação financeira.

    Não corrige nem corta nomes: um sufixo de protocolo não é uma ferramenta
    autorizada e não deve contaminar o histórico reenviado ao provedor.
    """
    try:
        calls = list(getattr(message, "tool_calls", None) or [])
        validate_batch(agent, session, calls)
        offered = {tool["function"]["name"] for tool in tools}
        if any(call.function.name not in offered for call in calls):
            raise ToolContractError("ferramenta_nao_oferecida")
    except ToolContractError as error:
        raise InvalidToolResponse(error.code) from None


def _usage_metrics(response) -> dict:
    """Registra só contagens, sem raciocínio gerado ou texto do cliente."""
    usage = getattr(response, "usage", None)
    def value(obj, key):
        return obj.get(key) if isinstance(obj, Mapping) else getattr(obj, key, None)
    metrics = {}
    for key, label in (("prompt_tokens", "input_tokens"), ("completion_tokens", "output_tokens"),
                       ("total_tokens", "reported_tokens")):
        count = value(usage, key)
        if type(count) is int and count >= 0:
            metrics[label] = count
    cached = value(value(usage, "prompt_tokens_details"), "cached_tokens")
    if type(cached) is int and cached >= 0:
        metrics["cached_input_tokens"] = cached
    return metrics


def _build_payload(agent, session: Session, *, presentation_only: bool = False) -> list[dict]:
    instructions = agent.instructions(presentation_only=True) if presentation_only else agent.instructions()
    return [{"role": "system", "content": instructions},
            {"role": "system", "content": agent.session_context(session)},
            *_compact_local_results(select_history(agent.name, session, _context_history(session.history)),
                                    session, agent.name)]


def _compact_local_results(history: list[dict], session: Session, agent_name: str) -> list[dict]:
    """Omite cópias exatas do estado apenas no contexto enviado ao provedor.

    Preserva a ação, as instruções e as confirmações. Recibos financeiros
    e pares nativos de chamada e resultado não são compactados aqui.
    """
    snapshots = {
        "registrar_intencao_credito": {"proposta": session.credit_proposal},
    }
    if agent_name in {"credito", "entrevista"}:
        snapshots.update({
            "confirmacao_entrevista": {"respostas_confirmadas": session.flow.interview_answers},
            "registrar_respostas_entrevista": {"respostas_propostas": session.flow.pending_interview_answers,
                                              "respostas_ja_confirmadas": session.flow.interview_answers},
        })
    result = []
    for item in history:
        if item.get("role") == "system":
            try:
                wire = json.loads(item["content"])
                local = wire["resultado_local"]
                values = local["resultado"]
                removed = False
                for key, current in snapshots.get(local["acao"], {}).items():
                    if key in values and current is not None and values[key] == current:
                        del values[key]
                        removed = True
                if removed:
                    item = {**item, "content": json.dumps(wire, ensure_ascii=False, separators=(",", ":"))}
            except (ValueError, KeyError, TypeError):
                pass
        result.append(item)
    return result


def _context_history(history: list[dict]) -> list[dict]:
    """Mantém os dois últimos turnos do cliente sem separar pares assistant/tool.

    O estado que precisa sobreviver a turnos mais antigos vive em ``Session`` e
    é reapresentado no prompt do agente. O histórico completo continua na
    sessão para a UI e diagnóstico; somente a cópia enviada ao provedor é
    reduzida.
    """
    user_indexes = [
        index for index, message in enumerate(history)
        if message.get("role") == "user" and not message.get(_SKIP_LLM_CONTEXT)
    ]
    if len(user_indexes) <= MAX_CONTEXT_USER_TURNS:
        window = history
    else:
        window = history[user_indexes[-MAX_CONTEXT_USER_TURNS]:]
    return [
        {key: value for key, value in message.items() if key != _SKIP_LLM_CONTEXT}
        for message in window
        if not message.get(_SKIP_LLM_CONTEXT)
    ]


def _estimate_request_tokens(payload: list[dict], tools: list[dict], format_spec: dict | None = None, *,
                             request_options: dict | None = None) -> int:
    """Estimativa conservadora para limitar texto excessivo antes da chamada."""
    request = {"messages": payload, "tools": tools}
    if format_spec is not None:
        request["response_format"] = format_spec
    if request_options is not None:
        request.update(request_options)
    serialized = json.dumps(
        request,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return max(1, (len(serialized) + ESTIMATED_CHARS_PER_TOKEN - 1) // ESTIMATED_CHARS_PER_TOKEN)


def _record_input_too_long(session: Session, estimated_tokens: int) -> None:
    session.last_failure_cause = "input_too_long"
    session.last_failure_category = "input_too_long"
    session.last_failure_reason = "A mensagem excede o orçamento seguro de processamento."
    session.last_llm_payload = json.dumps(
        {"bloqueado": "orcamento_de_entrada", "tokens_estimados": estimated_tokens},
        ensure_ascii=False,
    )
    session.last_llm_response = "Entrada excessiva bloqueada antes da chamada ao provedor."


def _exclude_largest_user_message_in_context(session: Session) -> None:
    """Evita reenviar a parte que excedeu o orçamento, sem apagá-la da sessão."""
    candidates = [
        message for message in session.history
        if message.get("role") == "user" and not message.get(_SKIP_LLM_CONTEXT)
    ][-MAX_CONTEXT_USER_TURNS:]
    if candidates:
        max(candidates, key=lambda message: len(str(message.get("content", ""))))[_SKIP_LLM_CONTEXT] = True


def _tool_schemas(agent, session: Session) -> list[dict]:
    """Envia ao modelo apenas as ferramentas possíveis no estado atual."""
    return schemas_for(agent, session)


def _record_success(message, agent_name: str, session: Session):
    recorded = {
        "agente": agent_name,
        "content": getattr(message, "content", None),
        "tool_calls": [
            {
                "id": call.id,
                "name": getattr(getattr(call, "function", None), "name", None),
                "arguments": getattr(getattr(call, "function", None), "arguments", None),
            }
            for call in list(getattr(message, "tool_calls", None) or [])
        ],
    }
    if isinstance(message, structured_output.StructuredMessage):
        recorded = {"agente": agent_name, "modo": "structured", "saida": message.envelope}
    session.last_llm_response = json.dumps(recorded, ensure_ascii=False, indent=2)
    session.last_failure_reason = None
    session.last_provider_error_code = None
    session.last_generation_json_valid = None
    session.last_provider_diagnostic = {}
    session.last_failure_cause = None
    session.last_failure_category = None
    session.last_tool_error_code = None
    return message


def _diagnostic_model(model):
    return model if model in {"openai/gpt-oss-20b", "openai/gpt-oss-120b", "fake"} else "other_model"


def _recovery_cause(error, category, status):
    if isinstance(error, structured_output.StructuredOutputError):
        return "invalid_structured_response"
    if isinstance(error, InvalidToolResponse):
        return "tool_contract_violation"
    if status != 400:
        return category
    body = getattr(error, "body", None)
    detail = body.get("error", body) if isinstance(body, Mapping) else None
    if not isinstance(detail, Mapping):
        return "invalid_request"
    # Um parâmetro inválido indica problema na requisição, mesmo quando
    # o provedor também informa um código de erro de geração.
    if detail.get("param") not in (None, ""):
        return "invalid_request"
    text = detail.get("message")
    if isinstance(text, str) and text.lstrip().lower().startswith(("invalid json schema", "invalid schema", "invalid response_format")):
        return "invalid_request"
    code = detail.get("code")
    return code if isinstance(code, str) and code in {"tool_use_failed", "json_validate_failed"} else "invalid_request"


def _provider_diagnostics(error):
    """Retém apenas códigos conhecidos e a validade do JSON, sem conteúdo do provedor."""
    body = getattr(error, "body", None)
    if not isinstance(body, Mapping):
        return None, None
    detail = body.get("error", body)
    if not isinstance(detail, Mapping):
        return None, None
    code = detail.get("code")
    if not isinstance(code, str) or code not in {
            "json_validate_failed", "tool_use_failed", "context_length_exceeded",
            "invalid_api_key", "rate_limit_exceeded", "model_not_found"}:
        code = "invalid_response_schema" if detail.get("param") == "response_format" else "other_provider_error"
    generation = detail.get("failed_generation")
    valid = None
    if isinstance(generation, str) and len(generation) <= 100_000:
        try:
            json.loads(generation)
            valid = True
        except (ValueError, RecursionError):
            valid = False
    return code, valid


def _record_failure(error: Exception, session: Session, category: str, status: int | None):
    session.last_provider_diagnostic = {}
    session.last_failure_cause = _recovery_cause(error, category, status)
    session.last_provider_error_code, session.last_generation_json_valid = _provider_diagnostics(error)
    session.last_failure_category = category
    session.last_failure_reason = _failure_reason(category, status)
    session.last_tool_error_code = str(error) if isinstance(error, (InvalidToolResponse, structured_output.StructuredOutputError)) else None
    if (isinstance(error, structured_output.StructuredOutputError) and error.blocked_action
            and credit_confirmation_pending(session)):
        session.last_tool_error_code = "pedido_confirmado_aguarda_analise"
    session.last_llm_response = f"Erro: {type(error).__name__} | status={status} | codigo={session.last_provider_error_code}"


def _classify_error(error: Exception) -> tuple[str, int | None]:
    if isinstance(error, CapacityError):
        return "local_capacity", None
    if isinstance(error, structured_output.StructuredOutputError):
        return "invalid_structured_response", None
    if isinstance(error, InvalidToolResponse):
        return "invalid_tool_response", None
    status = getattr(error, "status_code", None)
    if isinstance(status, int):
        return f"http_{status}", status
    name = type(error).__name__.casefold()
    if isinstance(error, TimeoutError) or "timeout" in name:
        return "timeout", None
    if isinstance(error, ConnectionError) or "connection" in name:
        return "connection", None
    return "provider_error", None


def _is_retryable(category: str, status: int | None) -> bool:
    return category in {"timeout", "connection"} or status == 429 or (status is not None and status >= 500)


def _retry_delay(category: str, status: int | None, attempt: int, max_retries: int) -> float | None:
    """Retorna uma espera segura para uma única nova tentativa, se houver."""
    if attempt > max(0, max_retries) or not _is_retryable(category, status):
        return None
    if status == 429:
        return None
    return min(0.25 * (2 ** (attempt - 1)), 1.0)


def _rate_limit_retry_delay(error: Exception, retry_budget: int, retry_state: LlmRetryState,
                            max_rate_limit_wait_seconds: float) -> float | None:
    """Consome a única recuperação de 429 permitida no turno."""
    if retry_budget < 1 or retry_state.rate_limit_retry_used:
        return None
    delay = _retry_after_seconds(error, max_rate_limit_wait_seconds)
    if delay is None:
        return None
    retry_state.rate_limit_retry_used = True
    return delay


def _retry_after_seconds(error: Exception, max_rate_limit_wait_seconds: float) -> float | None:
    """Lê Retry-After sem assumir detalhes de um SDK ou provedor específico."""
    candidates = (getattr(error, "headers", None), getattr(getattr(error, "response", None), "headers", None))
    for headers in candidates:
        if not headers or not hasattr(headers, "get"):
            continue
        value = headers.get("retry-after") or headers.get("Retry-After")
        try:
            seconds = float(value)
        except (TypeError, ValueError):
            try:
                seconds = max(0.0, parsedate_to_datetime(value).timestamp() - time.time())
            except (TypeError, ValueError, OverflowError):
                continue
        if math.isfinite(seconds) and seconds >= 0:
            delay = max(MIN_RATE_LIMIT_WAIT_SECONDS, seconds)
            return delay if delay <= max_rate_limit_wait_seconds else None
    return MIN_RATE_LIMIT_WAIT_SECONDS if max_rate_limit_wait_seconds >= MIN_RATE_LIMIT_WAIT_SECONDS else None


def _failure_reason(category: str, status: int | None) -> str:
    if status == 429:
        return "O provedor está temporariamente com alta demanda."
    if status is not None:
        return f"Indisponibilidade do provedor (HTTP {status})."
    return f"Indisponibilidade do provedor ({category})."
