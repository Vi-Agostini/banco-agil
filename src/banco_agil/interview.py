"""Validação determinística das respostas confirmadas da entrevista.

O agente de IA conduz a conversa e propõe os dados que entendeu da última
mensagem. Esta camada só aceita valores nos domínios previstos; a proposta
precisa ser confirmada pelo cliente antes de entrar em ``interview_answers``.
"""
from __future__ import annotations

from typing import Any

from .scoring import ValidationError, normalize_employment, parse_bool, parse_dependents, parse_money


INTERVIEW_FIELDS = (
    "renda_mensal",
    "tipo_emprego",
    "despesas_fixas",
    "num_dependentes",
    "tem_dividas",
)


def has_interview_request(session) -> bool:
    """Vincula a entrevista em andamento ao pedido reprovado existente."""
    return bool(session.flow.interview_started and session.last_request_status == "reprovado"
                and session.last_request_id is not None and session.desired_limit is not None)


def confirm_interview_proposal(session) -> dict:
    """Confirma só os dados pendentes da entrevista, sem autorizar operações financeiras."""
    flow = session.flow
    if not (session.authenticated and session.cpf and session.active_agent == "entrevista"
            and flow.interview_started and flow.pending_interview_answers):
        raise ValidationError("Não há dados da entrevista para confirmar.", "confirmacao_entrevista_indisponivel")
    flow.interview_answers.update(flow.pending_interview_answers)
    flow.pending_interview_answers.clear()
    flow.interview_revision_requested = False
    flow.input_kind = "interview"
    flow.input_prompt = None
    flow.attempts = 0
    flow.interview_question_pending = True
    return {"ok": True, "confirmacao_pendente": False,
            "respostas_confirmadas": dict(flow.interview_answers),
            "instrucao": "Dados confirmados pelo cliente. Não peça nova confirmação desses valores. "
                         "Formule já a próxima pergunta sobre campos ausentes, em ordem livre; calcule apenas quando completos."}


def next_interview_prompt(answers: dict[str, Any], pending: dict[str, Any]) -> str:
    """Orienta a coleta sem inventar valores nem confirmar uma proposta."""
    if pending:
        return "Confirme ou corrija os dados do resumo anterior antes de continuar."
    prompts = {
        "renda_mensal": "Qual é a sua renda mensal?",
        "tipo_emprego": "Seu emprego é formal, autônomo ou está desempregado?",
        "despesas_fixas": "Qual é o valor das suas despesas fixas mensais?",
        "num_dependentes": "Quantos dependentes você tem?",
        "tem_dividas": "Você tem dívidas ativas?",
    }
    missing = missing_interview_fields(answers)
    return prompts[missing[0]] if missing else "Seus dados confirmados foram preservados; não repeti o cálculo do score."


def missing_interview_fields(answers: dict[str, Any]) -> tuple[str, ...]:
    return tuple(field for field in INTERVIEW_FIELDS if field not in answers)


def canonical_interview_answers(values: dict[str, Any]) -> dict[str, Any]:
    """Normaliza os cinco valores na forma usada pelo cálculo e pela comparação."""
    return {
        "renda_mensal": parse_money(values["renda_mensal"], "renda mensal"),
        "tipo_emprego": normalize_employment(values["tipo_emprego"]),
        "despesas_fixas": parse_money(values["despesas_fixas"], "despesas fixas"),
        "num_dependentes": parse_dependents(values["num_dependentes"]),
        "tem_dividas": parse_bool(values["tem_dividas"], "dívidas ativas"),
    }


def canonical_interview_updates(values: dict[str, Any]) -> dict[str, Any]:
    """Valida um subconjunto de respostas proposto pelo agente.

    Ausências continuam ausências: somente os campos enviados pela ferramenta
    entram na proposta pendente de confirmação.
    """
    updates = {field: values[field] for field in INTERVIEW_FIELDS if values.get(field) is not None}
    if not updates:
        raise ValidationError("Informe ao menos uma resposta da entrevista.", "resposta_ausente")
    defaults = {
        "renda_mensal": 0,
        "tipo_emprego": "desempregado",
        "despesas_fixas": 0,
        "num_dependentes": 0,
        "tem_dividas": False,
    }
    normalized = canonical_interview_answers({**defaults, **updates})
    return {field: normalized[field] for field in updates}


def merge_interview_proposal(
    confirmed: dict[str, Any], pending: dict[str, Any], proposed: dict[str, Any], *,
    allow_pending_revisions: bool = False,
    allow_confirmed_revisions: bool = False,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Acumula uma proposta sem alterar respostas já confirmadas.

    Cada valor já passou pela validação de domínio. Uma nova chamada pode trazer
    campos complementares da mesma mensagem do cliente. Uma revisão solicitada
    pelo cliente pode propor outro valor confirmado ou pendente; o valor
    confirmado só muda após nova confirmação da proposta corrigida.
    Repetições equivalentes são idempotentes.
    """
    normalized_confirmed = canonical_interview_updates(confirmed) if confirmed else {}
    normalized_pending = canonical_interview_updates(pending) if pending else {}
    normalized_proposed = canonical_interview_updates(proposed)
    conflicts = []

    for field, value in normalized_proposed.items():
        if field in normalized_confirmed:
            if normalized_confirmed[field] != value:
                if allow_confirmed_revisions:
                    normalized_pending[field] = value
                else:
                    conflicts.append(field)
            continue
        if field in normalized_pending:
            if normalized_pending[field] != value:
                if allow_pending_revisions:
                    normalized_pending[field] = value
                else:
                    conflicts.append(field)
            continue
        normalized_pending[field] = value

    return normalized_pending, tuple(conflicts)
