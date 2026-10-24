"""Ferramentas (tools) expostas aos agentes.

As ferramentas aplicam os contratos e delegam cálculos e persistência aos serviços.
A IA decide quando propor uma ação; não pode dispensar as validações.
Operações financeiras usam a identidade autenticada da sessão. A autenticação
recebe credenciais apenas pelo coletor local, sem argumentos fornecidos pelo modelo.
"""
from __future__ import annotations

import functools
import logging
import re
from decimal import Decimal
from dataclasses import dataclass
from typing import Callable

from .credit_service import CreditError, CreditService
from .exchange import ExchangeError, ExchangeService
from .repository import BankRepository, DataError, normalize_cpf, normalize_date
from .interview import (
    canonical_interview_answers,
    canonical_interview_updates,
    confirm_interview_proposal,
    has_interview_request,
    merge_interview_proposal,
    missing_interview_fields,
)
from .scoring import ValidationError, calculate_score, parse_money
from .session import Session
from .intents import normalize_text
from .statuses import STATUS_REPROVADO

log = logging.getLogger("banco_agil")

MAX_AUTH_FAILURES = 3
MAX_INVALID_INPUTS = 3
MSG_AUTH_RESTART = (
    "Não foi possível autenticar com os dados informados. Vamos recomeçar o login; por favor, informe seu CPF."
)
MSG_AUTH_FAILED = (
    "Não foi possível confirmar os seus dados e, por segurança, vou encerrar o atendimento por aqui. "
    "Quando quiser, é só voltar e tentar novamente. Obrigado por escolher o Banco Ágil!"
)
MSG_FAREWELL = "Atendimento encerrado. Foi um prazer ajudar! Obrigado por escolher o Banco Ágil."
MSG_INVALID_INPUT_LIMIT = (
    "Não consegui identificar uma solicitação válida após 3 tentativas. Vou encerrar este atendimento; "
    "você pode iniciar um novo quando quiser."
)

TRANSFER_TARGETS = {
    "triagem": {"credito", "cambio"},
    "credito": {"cambio", "entrevista", "triagem"},
    "entrevista": {"credito"},
    "cambio": {"credito", "triagem"},
}


@dataclass
class ToolContext:
    session: Session
    repo: BankRepository
    fx: ExchangeService
    credit: CreditService | None = None

    def __post_init__(self):
        if self.credit is None:
            self.credit = CreditService(self.repo)


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict
    handler: Callable[..., dict]

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def _obj(props: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": props, "required": required, "additionalProperties": False}


def require_auth(fn):
    """Bloqueia a ferramenta para sessões encerradas ou não autenticadas."""

    @functools.wraps(fn)
    def wrapper(ctx: ToolContext, *args, **kwargs):
        s = ctx.session
        if s.ended:
            return {"erro": "Atendimento encerrado.", "codigo": "sessao_encerrada"}
        if not (s.authenticated and s.cpf):
            return {
                "erro": "Acesso negado: cliente não autenticado.",
                "codigo": "nao_autenticado",
                "instrucao": "Conduza a autenticação (CPF e data de nascimento) antes de qualquer outro assunto.",
            }
        return fn(ctx, *args, **kwargs)

    return wrapper


# Triagem.
def autenticar_cliente(ctx: ToolContext, cpf: str = "", data_nascimento: str = "") -> dict:
    s = ctx.session
    if s.ended:
        return {"erro": "Atendimento encerrado.", "codigo": "sessao_encerrada"}
    if s.authenticated:
        return {"sucesso": True, "nome": s.nome, "observacao": "Cliente já autenticado."}
    cpf_raw = cpf if isinstance(cpf, str) else (str(cpf) if cpf is not None else "")
    data_raw = data_nascimento if isinstance(data_nascimento, str) else (str(data_nascimento) if data_nascimento is not None else "")

    if not cpf_raw.strip() or not data_raw.strip():
        return {
            "sucesso": False,
            "motivo": "dados_incompletos",
            "instrucao": "Peça ao cliente somente o dado de autenticação que estiver faltando.",
        }

    cpf_n = normalize_cpf(cpf_raw)
    data_n = normalize_date(data_raw)

    if not cpf_n or not data_n:
        return _authentication_failure(s)
    client = ctx.repo.authenticate(cpf_n, data_n)
    if client:
        s.authenticated, s.cpf, s.nome, s.auth_failures = True, cpf_n, client["nome"], 0
        return {"sucesso": True, "nome": client["nome"]}

    return _authentication_failure(s)


def _authentication_failure(s: Session) -> dict:
    s.auth_failures += 1
    if s.auth_failures >= MAX_AUTH_FAILURES:
        s.ended = True
        return {"sucesso": False, "encerrado": True, "mensagem_final": MSG_AUTH_FAILED}
    return {
        "sucesso": False,
        "motivo": "dados_nao_conferem",
        "tentativas_restantes": MAX_AUTH_FAILURES - s.auth_failures,
        "instrucao": "Não identifique qual dado falhou. Informe que o login não foi validado e recomece pedindo o CPF.",
    }


@require_auth
def transferir_para_agente(ctx: ToolContext, destino: str) -> dict:
    s = ctx.session
    if destino not in TRANSFER_TARGETS.get(s.active_agent, set()):
        return {"erro": f"Destino '{destino}' indisponível a partir daqui.", "codigo": "destino_indisponivel"}
    s.active_agent = destino
    instruction = "Continue o atendimento normalmente, sem mencionar transição."
    if destino == "credito" and has_interview_request(s):
        instruction += " Retome o pedido anterior pelo valor preservado; a entrevista ainda precisa ser concluída. Não peça o valor novamente nem registre outro pedido."
    return {"ok": True, "instrucao": instruction}


def encerrar_atendimento(ctx: ToolContext) -> dict:
    ctx.session.ended = True
    return {"ok": True, "mensagem_final": MSG_FAREWELL}


def registrar_entrada_invalida(ctx: ToolContext) -> dict:
    s = ctx.session
    attempts = s.record_invalid_input()
    if attempts >= MAX_INVALID_INPUTS:
        s.ended = True
        return {
            "encerrado": True,
            "tentativas": attempts,
            "mensagem_final": MSG_INVALID_INPUT_LIMIT,
        }
    return {
        "encerrado": False,
        "tentativas_restantes": MAX_INVALID_INPUTS - attempts,
        "instrucao": "Diga brevemente que não entendeu e peça ao cliente para reformular a solicitação.",
    }


# Crédito.
def credit_wording(session: Session) -> str | None:
    text = next((m.get("content", "") for m in reversed(session.history) if m.get("role") == "user"), "")
    text = normalize_text(text)
    if re.search(r"\b(?:aument\w*|acrescimo)(?:\s+\w+){0,4}\s+em\b", text):
        return "acrescimo"
    if re.search(r"\b(?:aument\w*|limite)\s+(?:\w+\s+){0,3}para\b", text):
        return "limite_total"
    if re.search(r"\baumento de\s+(?:r\s+)?(?:\d|\w+ mil)", text):
        return "indefinida"
    return None


def credit_literal_authorized(session: Session, value: float | None = None) -> bool:
    """Recibo conservador de autorização, não um interpretador de linguagem.

    A IA entende texto por extenso. Quando não há um único valor literal ligado
    ao pedido, sua proposta precisa de confirmação explícita do cliente.
    """
    text = next((m.get("content", "") for m in reversed(session.history) if m.get("role") == "user"), "")
    norm = normalize_text(text)
    if re.search(r"\b(?:nao|nunca|nem)\b", norm) or re.search(r"\b(?:mil|milhao|milhoes)\b", norm):
        return False
    explicit_change = (re.search(r"\baument\w*\b", norm) or
                       re.search(r"\bquero\s+(?:r\s*)?\d", norm) or
                       ("acrescimo" in norm and re.search(r"\b(?:quero|solicito)\b", norm)))
    if '?' in text or re.match(r"(?:se|caso|como|talvez)\b", norm):
        return False
    if not (explicit_change or session.flow.awaiting.value == "limit_amount"):
        return False
    numbers = set()
    for match in re.finditer(r"(?<![\w*/-])\d+(?:[.,]\d+)*(?![\w])", text):
        try:
            numbers.add(parse_money(match.group(0)))
        except ValidationError:
            return False
    return len(numbers) == 1 and (value is None or value in numbers)


def registrar_intencao_credito(ctx: ToolContext, valor: float, modalidade: str) -> dict:
    """Memória de coleta, sem solicitação, score ou escrita financeira."""
    s = ctx.session
    if s.ended:
        return {"erro": "Atendimento encerrado.", "codigo": "sessao_encerrada"}
    try:
        amount = parse_money(valor, "valor do pedido", positive=True)
    except ValidationError as error:
        return {"erro": str(error), "codigo": error.code}
    if modalidade not in {"limite_total", "acrescimo", "indefinida"}:
        return {"erro": "Esclareça o significado do valor.", "codigo": "valor_ambiguo"}
    if has_interview_request(s):
        if modalidade == "limite_total" and amount == s.desired_limit:
            return {"ok": True, "pedido_anterior_preservado": True, "valor_desejado": s.desired_limit,
                    "instrucao": "Esse é o pedido anterior. Ofereça retomar a entrevista pendente, sem pedir o valor de novo ou registrar outra solicitação."}
        return {"erro": "Conclua ou cancele a entrevista do pedido anterior antes de preparar outro pedido.",
                "codigo": "entrevista_em_andamento"}
    modalidade = credit_wording(s) or modalidade
    previous = s.credit_proposal or {}
    confirmed = (previous.get("valor") == amount and previous.get("modalidade") == modalidade
                 and previous.get("confirmada") and not previous.get("executada"))
    proposal = {"valor": amount, "modalidade": modalidade, "turno": s.turn,
                "confirmada": bool(modalidade != "indefinida" and (confirmed or credit_literal_authorized(s, amount)))}
    if modalidade == "limite_total":
        proposal["novo_limite"] = amount
    elif modalidade == "acrescimo" and s.authenticated and s.cpf:
        client = ctx.repo.get_client(s.cpf)
        if not client:
            return {"erro": "Cliente não encontrado.", "codigo": "cliente_nao_encontrado"}
        proposal["limite_base"] = client["limite_credito"]
        try:
            proposal["novo_limite"] = parse_money(
                float(Decimal(str(client["limite_credito"])) + Decimal(str(amount))),
                "novo limite", positive=True,
            )
        except ValidationError as error:
            return {"erro": str(error), "codigo": error.code}
    s.credit_proposal = proposal
    return {"ok": True, "proposta": proposal, "solicitacao_registrada": False,
            "instrucao": "Se modalidade=indefinida, esclareça total ou acréscimo sem pedir confirmação. Somente com modalidade definida e novo_limite calculado, resuma e peça confirmação quando confirmada=false. Autentique antes da operação. Use o novo limite calculado pelo código."}


@require_auth
def consultar_limite(ctx: ToolContext) -> dict:
    s = ctx.session
    try:
        ctx.credit.reconcile(s.cpf)
    except DataError:
        log.warning("Reconciliação falhou durante consulta de limite")
    client = ctx.repo.get_client(s.cpf)
    if not client:
        return {"erro": "Cliente não encontrado.", "codigo": "cliente_nao_encontrado"}
    return {"limite_credito_atual": client["limite_credito"]}


@require_auth
def solicitar_aumento_limite(ctx: ToolContext, novo_limite: float) -> dict:
    s = ctx.session
    if has_interview_request(s):
        return {"erro": "Conclua ou cancele a entrevista do pedido anterior antes de registrar outro pedido.",
                "codigo": "entrevista_em_andamento"}
    try:
        novo = parse_money(novo_limite, "novo limite", positive=True)
    except ValidationError as error:
        return {"erro": str(error), "codigo": error.code}
    proposal = s.credit_proposal if not (s.credit_proposal or {}).get("executada") else None
    if credit_wording(s) in {"acrescimo", "indefinida"} and not proposal:
        return {"erro": "Esclareça e registre se o valor é total ou acréscimo.", "codigo": "valor_ambiguo"}
    if proposal:
        if proposal["modalidade"] == "indefinida":
            return {"erro": "Esclareça se o valor é total ou acréscimo.", "codigo": "valor_ambiguo"}
        if proposal["modalidade"] == "acrescimo" and not s.credit_operation:
            client = ctx.repo.get_client(s.cpf)
            if not client or proposal.get("limite_base") != client["limite_credito"]:
                return {"erro": "O limite mudou; atualize a proposta antes de solicitar.", "codigo": "limite_base_alterado"}
        if proposal.get("novo_limite") != novo:
            return {"erro": "O valor não corresponde à proposta coletada.", "codigo": "dados_pedido_nao_conferem"}
    try:
        if s.credit_operation and s.credit_operation["amount"] != novo:
            return {"erro": "Confirme o status do pedido anterior antes de solicitar outro valor.",
                    "codigo": "pedido_estado_incerto"}
        if not s.credit_operation or s.credit_operation["amount"] != novo:
            s.credit_operation = {"amount": novo, "receipt": {}}
        result = ctx.credit.request_increase(s.cpf, novo, receipt=s.credit_operation["receipt"])
    except CreditError as error:
        if s.credit_operation and not s.credit_operation["receipt"]:
            s.credit_operation = None
        return {"erro": str(error), "codigo": error.code}
    s.last_request_status, s.desired_limit, s.last_request_id = result["status_pedido"], novo, result["id"]
    s.credit_operation = None  # Após uma resposta concluída, só operações ainda incertas mantêm recibo.
    s.credit_proposal = {"valor": novo, "modalidade": "limite_total", "novo_limite": novo,
                         "executada": True, "confirmada": True}
    s.automatic_reanalysis_result = None
    s.automatic_reanalysis_response_pending = False
    return result


# Entrevista.
@require_auth
def confirmar_dados_entrevista(ctx: ToolContext) -> dict:
    """Aplica o aceite interpretado pela IA a uma proposta pendente anterior."""
    s = ctx.session
    if (not s.flow.interview_revision_requested
            or s.tool_facts.get("registrar_respostas_entrevista", {}).get("turn") == s.turn):
        return {"erro": "Aguarde uma confirmação do cliente após apresentar a proposta.",
                "codigo": "confirmacao_entrevista_indisponivel"}
    try:
        return confirm_interview_proposal(s)
    except ValidationError as error:
        return {"erro": str(error), "codigo": error.code}


@require_auth
def registrar_respostas_entrevista(
    ctx: ToolContext,
    renda_mensal=None,
    tipo_emprego=None,
    despesas_fixas=None,
    num_dependentes=None,
    tem_dividas=None,
) -> dict:
    """Valida uma proposta do LLM, sem registrá-la antes da confirmação humana."""
    s = ctx.session
    if not s.flow.interview_started or s.active_agent != "entrevista":
        return {"erro": "A entrevista não está ativa.", "codigo": "entrevista_nao_iniciada"}
    try:
        proposed = canonical_interview_updates({
            "renda_mensal": renda_mensal,
            "tipo_emprego": tipo_emprego,
            "despesas_fixas": despesas_fixas,
            "num_dependentes": num_dependentes,
            "tem_dividas": tem_dividas,
        })
        merged, conflicts = merge_interview_proposal(
            s.flow.interview_answers,
            s.flow.pending_interview_answers,
            proposed,
            allow_pending_revisions=s.flow.interview_revision_requested,
            allow_confirmed_revisions=s.flow.interview_revision_requested,
        )
    except ValidationError as error:
        return {
            "erro": str(error),
            "codigo": error.code,
            "instrucao": "Não registre valores presumidos. Peça esclarecimento sobre o dado ambíguo ou inválido.",
        }
    if conflicts:
        return {
            "erro": "Há valores conflitantes para respostas já informadas na entrevista.",
            "codigo": "dados_entrevista_conflitantes",
            "campos_conflitantes": list(conflicts),
            "instrucao": "Peça esclarecimento somente sobre os dados conflitantes; não escolha um valor.",
        }
    s.flow.pending_interview_answers = merged
    s.flow.interview_revision_requested = False
    if not merged:
        return {
            "ok": True,
            "confirmacao_necessaria": False,
            "respostas_propostas": {},
            "respostas_ja_confirmadas": s.flow.interview_answers,
            "instrucao": "Essas respostas já estão confirmadas. Solicite somente os dados ainda ausentes.",
        }
    return {
        "ok": True,
        "confirmacao_necessaria": True,
        "respostas_propostas": merged,
        "respostas_ja_confirmadas": s.flow.interview_answers,
        "instrucao": (
            "Resuma exatamente os dados propostos e peça confirmação explícita do cliente. "
            "Eles ainda não foram registrados e o score não pode ser calculado."
        ),
    }


@require_auth
def registrar_entrevista_e_calcular_score(
    ctx: ToolContext,
    renda_mensal,
    tipo_emprego,
    despesas_fixas,
    num_dependentes,
    tem_dividas,
) -> dict:
    s = ctx.session
    if s.flow.pending_interview_answers:
        return {
            "erro": "Há respostas da entrevista aguardando confirmação do cliente.",
            "codigo": "entrevista_aguardando_confirmacao",
            "instrucao": "Peça confirmação explícita dos dados resumidos antes de calcular o score.",
        }
    missing = missing_interview_fields(s.flow.interview_answers)
    if missing:
        return {
            "erro": "A entrevista ainda não contém todas as respostas necessárias.",
            "codigo": "entrevista_incompleta",
            "campos_faltantes": list(missing),
            "instrucao": "Solicite somente o próximo dado que ainda falta, sem calcular o score.",
        }
    try:
        collected = canonical_interview_answers(s.flow.interview_answers)
    except ValidationError as error:
        return {
            "erro": str(error),
            "codigo": error.code,
            "instrucao": (
                "Peça para corrigir somente a informação inválida, sem reiniciar a entrevista nem repetir "
                "os valores já informados. Se a mesma combinação falhar novamente, ofereça desistir da entrevista."
            ),
        }
    try:
        supplied = canonical_interview_answers({
            "renda_mensal": renda_mensal,
            "tipo_emprego": tipo_emprego,
            "despesas_fixas": despesas_fixas,
            "num_dependentes": num_dependentes,
            "tem_dividas": tem_dividas,
        })
    except ValidationError as error:
        return {
            "erro": str(error),
            "codigo": error.code,
            "instrucao": "Peça para corrigir somente a informação inválida, sem reiniciar a entrevista.",
        }
    if supplied != collected:
        return {
            "erro": "Os dados enviados não correspondem às respostas coletadas do cliente.",
            "codigo": "dados_entrevista_nao_conferem",
            "instrucao": "Use somente as respostas já coletadas do cliente; não presuma valores.",
        }
    novo = calculate_score(**collected)
    client = ctx.repo.get_client(s.cpf)
    if not client:
        return {"erro": "Cliente não encontrado.", "codigo": "cliente_nao_encontrado"}
    ctx.repo.update_score(s.cpf, novo)
    s.active_agent = "credito"
    reanalysis = _automatic_credit_reanalysis(ctx)
    return {
        "score_anterior": client["score"],
        "novo_score": novo,
        "valor_desejado_anteriormente": s.desired_limit,
        "reanalise_automatica": reanalysis,
        "instrucao": _reanalysis_instruction(reanalysis),
    }


def _automatic_credit_reanalysis(ctx: ToolContext) -> dict:
    """Executa uma única reanálise segura depois de persistir o score."""
    s = ctx.session
    if s.last_request_status != STATUS_REPROVADO or s.last_request_id is None or s.desired_limit is None:
        return {"executada": False, "codigo": "reanalise_sem_origem_valida"}
    try:
        requested = parse_money(s.desired_limit, "novo limite", positive=True)
    except ValidationError:
        return {"executada": False, "codigo": "reanalise_sem_origem_valida"}
    if s.automatic_reanalysis_source_id == s.last_request_id:
        return {
            "executada": False,
            "codigo": "reanalise_ja_executada",
            "solicitacao_origem": s.automatic_reanalysis_source_id,
        }

    # Marca antes da escrita: se a persistência falhar em estado incerto, este
    # atendimento não repete automaticamente uma operação possivelmente gravada.
    s.automatic_reanalysis_source_id = s.last_request_id
    receipt = {}
    try:
        result = ctx.credit.reanalyze_rejected_request(s.cpf, s.last_request_id, requested, receipt=receipt)
    except (CreditError, DataError, TimeoutError):
        try:
            # Resolve apenas um recibo existente. Uma reanálise de resultado incerto
            # nunca cria outro pedido financeiro.
            result = ctx.credit.request_increase(s.cpf, requested, receipt=receipt, resume_only=True)
        except (CreditError, DataError, TimeoutError):
            result = {"erro": "Não foi possível concluir a reanálise automática com segurança.",
                      "codigo": "reanalise_indisponivel"}
    if "erro" not in result:
        s.last_request_status = result["status_pedido"]
        s.last_request_id = result["id"]

    s.automatic_reanalysis_result = result
    s.automatic_reanalysis_response_pending = True
    return {"executada": True, **result}


def _reanalysis_instruction(reanalysis: dict) -> str:
    if reanalysis.get("executada") and "erro" not in reanalysis:
        return (
            "A reanálise do pedido anterior já foi executada pelo serviço de crédito. "
            "Informe o resultado ao cliente com base neste retorno e não registre outro pedido."
        )
    if reanalysis.get("executada"):
        return (
            "O score foi atualizado, mas a reanálise automática não foi concluída com segurança. "
            "Informe o ocorrido sem tentar repetir o pedido neste atendimento."
        )
    return "Informe que o score foi atualizado. Não há pedido anterior válido para reanálise automática."


# Câmbio.
@require_auth
def consultar_cotacao(ctx: ToolContext, moeda: str = "USD") -> dict:
    try:
        return ctx.fx.get_rate(moeda)
    except ExchangeError as e:
        clarification = e.code == "moeda_nao_identificada"
        return {
            "erro": str(e), "codigo": e.code,
            "instrucao": (
                "Esclareça qual moeda o cliente deseja pelo nome ou país, sem exigir um código técnico. "
                "Não classifique essa dúvida como entrada inválida."
                if clarification else
                "Informe a indisponibilidade desta cotação e ofereça consultar outra moeda ou tentar mais tarde. "
                "Não atribua a falha ao cliente, não registre entrada inválida e não invente valores."
            ),
        }


def build_transfer_tool(agent_name: str) -> Tool:
    return Tool(
        "transferir_para_agente",
        "Muda o especialista que atende o cliente. Chame sem escrever texto antes.",
        _obj(
            {"destino": {"type": "string", "enum": sorted(TRANSFER_TARGETS[agent_name])}},
            ["destino"],
        ),
        transferir_para_agente,
    )


END_TOOL = Tool(
    "encerrar_atendimento",
    "Encerra o atendimento. Use quando o cliente pedir para encerrar/sair ou não precisar de mais nada.",
    _obj({}, []),
    encerrar_atendimento,
)

INVALID_INPUT_TOOL = Tool(
    "registrar_entrada_invalida",
    "Registra uma mensagem claramente incompreensível ou aleatória. Use somente quando não for possível identificar uma intenção; não use para erros de digitação ou respostas curtas que façam sentido no fluxo.",
    _obj({}, []),
    registrar_entrada_invalida,
)
