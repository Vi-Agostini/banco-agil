"""Contrato comum do protocolo; regras financeiras continuam nos serviços."""
from __future__ import annotations

import json
import math
import re

from .interview import canonical_interview_answers, canonical_interview_updates, missing_interview_fields, has_interview_request
from .scoring import ValidationError, parse_money


class ToolContractError(ValueError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)  # Nunca inclua argumentos, credenciais ou resposta bruta.


def credit_clarification_pending(session) -> bool:
    """Uma proposta ambígua não pode ser apresentada como autorização financeira."""
    proposal = session.credit_proposal or {}
    return bool(session.active_agent == "credito" and session.authenticated and not session.ended
                and proposal.get("modalidade") == "indefinida" and not proposal.get("executada")
                and not has_interview_request(session))


def credit_confirmation_pending(session) -> bool:
    """O aceite deste turno ainda depende da decisão de ferramenta pelo agente.

    Consentimento antigo não força um pedido se o cliente mudar de assunto.
    Recibos incertos são resolvidos pela reconciliação, sem nova operação.
    """
    proposal = session.credit_proposal or {}
    return bool(session.credit_confirmation_turn == session.turn
                and session.active_agent == "credito" and session.authenticated and session.cpf
                and not session.ended and proposal.get("confirmada") is True
                and not proposal.get("executada") and "novo_limite" in proposal
                and proposal.get("modalidade") in {"limite_total", "acrescimo"}
                and not session.credit_operation and not has_interview_request(session)
                and not session.flow.pending_interview_answers
                and not session.automatic_reanalysis_response_pending)


def validate_name(name) -> None:
    if not isinstance(name, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", name):
        raise ToolContractError("nome_ferramenta_invalido")


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ToolContractError("argumentos_duplicados")
        result[key] = value
    return result


def parse_arguments(tool, raw: str) -> dict:
    try:
        args = json.loads(raw, object_pairs_hook=_object)
        schema = tool.parameters
        if not isinstance(args, dict) or set(args) - set(schema.get("properties", {})):
            raise ValueError
        if set(schema.get("required", ())) - set(args) or len(args) < schema.get("minProperties", 0):
            raise ValueError
        for name, value in args.items():
            spec = schema["properties"][name]
            kind = spec["type"]
            valid = {
                "string": lambda: isinstance(value, str),
                "boolean": lambda: type(value) is bool,
                "number": lambda: type(value) in (int, float) and math.isfinite(value),
                "integer": lambda: type(value) in (int, float) and math.isfinite(value) and int(value) == value,
            }[kind]()
            if not valid or ("enum" in spec and value not in spec["enum"]):
                raise ValueError
        return args
    except (ValueError, TypeError, OverflowError, KeyError):
        raise ToolContractError("argumentos_invalidos") from None


def precondition(agent, session, name: str, args: dict | None = None) -> None:
    """A mesma política orienta a oferta de schemas e o instante da execução."""
    if session.ended:
        raise ToolContractError("sessao_encerrada")
    if agent.name != session.active_agent or agent.get_tool(name) is None:
        raise ToolContractError("ferramenta_fora_escopo")
    if name == "autenticar_cliente":
        raise ToolContractError("autenticacao_local_obrigatoria")
    if credit_clarification_pending(session) and session.credit_clarification_turn == session.turn:
        raise ToolContractError("esclarecimento_credito_necessario")
    if credit_confirmation_pending(session) and name in {"transferir_para_agente", "registrar_intencao_credito"}:
        raise ToolContractError("pedido_confirmado_aguarda_analise")
    if name == "registrar_intencao_credito" and agent.name == "triagem" and session.authenticated:
        # Após autenticar, o Crédito coleta o pedido. Antes disso, esta ferramenta
        # preserva valores espontâneos sem registrar uma solicitação financeira.
        raise ToolContractError("encaminhamento_credito_obrigatorio")
    if name == "registrar_intencao_credito" and session.flow.interview_question_pending:
        raise ToolContractError("coleta_sem_dados_novos")
    if name not in {"encerrar_atendimento", "registrar_entrada_invalida", "registrar_intencao_credito"}:
        if not (session.authenticated and session.cpf):
            raise ToolContractError("nao_autenticado")
    if name == "registrar_respostas_entrevista":
        if not session.flow.interview_started or session.flow.interview_question_pending:
            raise ToolContractError("entrevista_sem_respostas")
        if args is not None:
            try:
                canonical_interview_updates(args)
            except ValueError:
                raise ToolContractError("argumentos_invalidos") from None
    if name == "confirmar_dados_entrevista":
        if (not session.flow.interview_started or not session.flow.pending_interview_answers
                or not session.flow.interview_revision_requested
                or session.tool_facts.get("registrar_respostas_entrevista", {}).get("turn") == session.turn):
            raise ToolContractError("confirmacao_entrevista_indisponivel")
    if name == "registrar_entrevista_e_calcular_score":
        if session.flow.pending_interview_answers:
            raise ToolContractError("entrevista_aguardando_confirmacao")
        if missing_interview_fields(session.flow.interview_answers):
            raise ToolContractError("entrevista_incompleta")
        try:
            canonical_interview_answers(session.flow.interview_answers)
        except ValueError:
            raise ToolContractError("argumentos_invalidos") from None
        if args is not None:
            try:
                supplied = canonical_interview_answers(args)
            except ValueError:
                raise ToolContractError("argumentos_invalidos") from None
            if supplied != canonical_interview_answers(session.flow.interview_answers):
                raise ToolContractError("dados_entrevista_nao_conferem")
    if name == "solicitar_aumento_limite":
        from .tools import credit_wording, credit_literal_authorized
        if has_interview_request(session):
            raise ToolContractError("entrevista_em_andamento")
        if args is not None:
            try:
                parse_money(args["novo_limite"], "novo limite", positive=True)
            except ValidationError as error:
                raise ToolContractError(error.code) from None
        if session.automatic_reanalysis_response_pending:
            raise ToolContractError("reanalise_ja_executada")
        if not session.credit_proposal and credit_wording(session) in {"acrescimo", "indefinida"}:
            raise ToolContractError("valor_ambiguo")
        if session.credit_proposal and session.credit_proposal["modalidade"] == "indefinida":
            raise ToolContractError("valor_ambiguo")
        if session.credit_proposal and "novo_limite" not in session.credit_proposal:
            raise ToolContractError("proposta_incompleta")
        proposal = session.credit_proposal
        staged = proposal and not proposal.get("executada")
        if staged and not proposal.get("confirmada"):
            raise ToolContractError("confirmacao_pedido_necessaria")
        if not staged and not credit_literal_authorized(session, args.get("novo_limite") if args else None):
            raise ToolContractError("confirmacao_pedido_necessaria")
        if args is not None:
            proposal = session.credit_proposal if not (session.credit_proposal or {}).get("executada") else None
            if proposal and proposal.get("novo_limite") != args["novo_limite"]:
                raise ToolContractError("dados_pedido_nao_conferem")
    if name == "registrar_entrada_invalida" and session.flow.interview_question_pending:
        raise ToolContractError("entrada_valida")


def schemas_for(agent, session) -> list[dict]:
    if credit_clarification_pending(session) and session.credit_clarification_turn == session.turn:
        # Um aceite curto não resolve a ambiguidade entre limite total e acréscimo.
        # O esclarecimento continua com a IA, sem ação que deduza uma autorização.
        return []
    if (session.llm_mode == "structured" and agent.name == "entrevista"
            and session.flow.interview_started):
        # Uma etapa já tratada localmente precisa de texto, não de outra decisão.
        # Novas informações podem revisar a proposta antes da próxima chamada;
        # uma entrevista completa continua permitindo o cálculo de score.
        asking_next = (session.flow.interview_question_pending and not session.flow.pending_interview_answers
                       and missing_interview_fields(session.flow.interview_answers))
        summarizing = session.flow.pending_interview_answers and not session.flow.interview_revision_requested
        if asking_next or summarizing:
            return []
    tools = []
    for tool in agent.tools:
        try:
            precondition(agent, session, tool.name)
        except ToolContractError:
            continue
        if tool.name == "registrar_respostas_entrevista":
            if session.flow.pending_interview_answers and not session.flow.interview_revision_requested:
                continue
            if not missing_interview_fields(session.flow.interview_answers) and not session.flow.interview_revision_requested:
                continue
        tools.append(tool.schema())
    return tools


def validate_batch(agent, session, calls) -> None:
    """Valida sem efeitos e rejeita misturas de transferência com outras operações."""
    names = []
    for call in calls:
        function = getattr(call, "function", None)
        name = getattr(function, "name", None)
        validate_name(name)
        tool = agent.get_tool(name)
        if tool is None:
            raise ToolContractError("ferramenta_fora_escopo")
        args = parse_arguments(tool, getattr(function, "arguments", None))
        precondition(agent, session, name, args)
        names.append(name)
    effects = {"solicitar_aumento_limite", "registrar_entrevista_e_calcular_score"}
    if "confirmar_dados_entrevista" in names and len(calls) != 1:
        raise ToolContractError("confirmacao_isolada_obrigatoria")
    if (len(calls) > 1 and "transferir_para_agente" in names) or sum(n in effects for n in names) > 1:
        raise ToolContractError("lote_com_efeitos_incompativeis")
    if "encerrar_atendimento" in names and any(n in effects for n in names):
        raise ToolContractError("lote_com_efeitos_incompativeis")
    if any(n in effects for n in names) and any(n in {"registrar_intencao_credito", "registrar_respostas_entrevista"} for n in names):
        raise ToolContractError("coleta_e_operacao_no_mesmo_lote")
