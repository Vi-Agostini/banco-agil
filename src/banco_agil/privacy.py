"""Mascaramento e remoção de dados pessoais (CPF e data de nascimento)."""
from __future__ import annotations

import re

_CPF_RE = re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b")
_DATE_RE = re.compile(r"\b\d{1,2}[/.\-]\d{1,2}[/.\-]\d{4}\b|\b\d{4}-\d{1,2}-\d{1,2}\b")
_BIRTH_MARKER_RE = re.compile(
    r"\b(?:nasc(?:i(?:do|da)?|imento)?|data[_\s]+(?:de[_\s]+)?nascimento|anivers[áa]rio)\b",
    re.IGNORECASE,
)
_UNRELATED_DATE_CONTEXT_RE = re.compile(
    r"\b(?:venc(?:e|imento|er)|pag(?:amento|ar)|agend(?:amento|ar|ado)|fatura)\b",
    re.IGNORECASE,
)


def mask_cpf(cpf: str | None) -> str:
    digits = "".join(c for c in (cpf or "") if c.isdigit())
    return f"***.***.***-{digits[-2:]}" if len(digits) >= 2 else "***"


def redact_text(text: str, *, redact_birth_dates: bool = False) -> str:
    """Mascara CPF e apenas datas usadas como credencial de nascimento.

    Datas são protegidas integralmente quando a triagem as aguarda. Nos demais
    casos, só a data mais próxima de um marcador explícito de nascimento é
    redigida, preservando datas de vencimento, pagamento e agendamento.
    """
    source = text if isinstance(text, str) else ""
    dates = list(_DATE_RE.finditer(source))
    if redact_birth_dates:
        birth_dates = dates
    else:
        markers = list(_BIRTH_MARKER_RE.finditer(source))
        birth_dates = []
        for marker in markers:
            if dates:
                following = [date for date in dates if date.start() >= marker.end()]
                candidates = following or [date for date in dates if date.end() <= marker.start()]
                if candidates:
                    birth_dates.append(min(candidates, key=lambda date: _distance(marker, date)))
    birth_spans = {date.span() for date in birth_dates}

    parts: list[str] = []
    cursor = 0
    for date in dates:
        parts.append(_CPF_RE.sub(lambda m: mask_cpf(m.group(0)), source[cursor:date.start()]))
        parts.append("[data de nascimento]" if date.span() in birth_spans else date.group(0))
        cursor = date.end()
    parts.append(_CPF_RE.sub(lambda m: mask_cpf(m.group(0)), source[cursor:]))
    return "".join(parts)


def _distance(first: re.Match, second: re.Match) -> int:
    """Retorna a distância entre dois trechos sem tratar a ordem como relevante."""
    if first.end() < second.start():
        return second.start() - first.end()
    if second.end() < first.start():
        return first.start() - second.end()
    return 0


def has_birth_marker(text: str) -> bool:
    return bool(_BIRTH_MARKER_RE.search(text if isinstance(text, str) else ""))


def has_unrelated_date_context(text: str) -> bool:
    return bool(_UNRELATED_DATE_CONTEXT_RE.search(text if isinstance(text, str) else ""))


def redact_history(history: list[dict]) -> None:
    """Remove credenciais na própria lista do histórico enviado ao LLM."""
    for msg in history:
        if isinstance(msg.get("content"), str):
            msg["content"] = redact_text(msg["content"])
        for call in msg.get("tool_calls") or []:
            fn = call["function"]
            if isinstance(fn.get("arguments"), str):
                fn["arguments"] = redact_text(fn["arguments"])
