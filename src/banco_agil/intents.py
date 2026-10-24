"""Interpretação pura de intenções e credenciais em texto livre."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from .repository import normalize_cpf, normalize_date


@dataclass(frozen=True)
class Credentials:
    cpf: str | None
    birth_date: str | None


_FORMATTED_CPF = re.compile(r"(?<!\d)\d{3}\.\d{3}\.\d{3}-\d{2}(?!\d)")
_PLAIN_CPF = re.compile(r"(?<!\d)\d{11}(?!\d)")
_DATE_CANDIDATE = re.compile(r"(?<!\d)(?:\d{4}-\d{1,2}-\d{1,2}|\d{1,2}[/.-]\d{1,2}[/.-]\d{4})(?!\d)")
_BIRTH_MARKER = re.compile(
    r"\b(?:nasc(?:i(?:do|da)?|imento)?|data[_\s]+(?:de[_\s]+)?nascimento|anivers[áa]rio)\b",
    re.IGNORECASE,
)
_CLOSE_VERBS = {"encerrar", "encerra", "encerre", "finalizar", "finalize", "sair", "saia"}
_CLOSE_FILLERS = {
    "quero", "vou", "pode", "preciso", "por", "favor", "obrigado", "obrigada", "o", "a",
    "de", "do", "da", "aqui", "agora", "hoje", "ai", "ok", "esta",
}
_CLOSE_TARGETS = {"atendimento", "conversa", "chat", "ligacao", "sessao"}
_CLOSE_SCOPE_INTENSIFIERS = {"todo", "toda", "inteiro", "inteira", "tudo"}
_GOODBYES = ("tchau", "adeus", "ate logo", "ate mais")
_LIMIT_AMOUNT = re.compile(r"^\s*(?:R\$\s*)?-?\d+(?:[.,]\d+)*\s*$", re.IGNORECASE)
_LIMIT_INCREASE = re.compile(r"\b(aument\w*|mais limite|limite maior|aumento de limite)\b")
_CLOSE_ACTION = re.compile(r"\b(encerrar|encerre|sair|saia|tchau|finalizar)\b")
# Sinais de interrupção que, junto com a palavra "entrevista", indicam cancelamento
# somente da etapa de entrevista (e não do atendimento inteiro).
_INTERVIEW_STOP = {
    "encerrar", "encerra", "encerre", "finalizar", "finalize",
    "abortar", "aborta", "aborte", "cancelar", "cancela", "cancele",
    "desistir", "desista", "desisto", "desisti", "sair", "saia",
    "parar", "pare", "recuso", "recusar", "recusei", "nao", "nunca",
}


def normalize_text(text: str) -> str:
    """Normaliza texto para comparações de intenção independentes de acento/pontuação."""
    source = text if isinstance(text, str) else ""
    decomposed = unicodedata.normalize("NFD", source.casefold())
    without_accents = "".join(char for char in decomposed if unicodedata.category(char) != "Mn")
    return " ".join(re.sub(r"[^\w\s]", " ", without_accents).split())


def is_isolated_greeting(text: str) -> bool:
    """Reconhece saudações completas sem descartar outros pedidos na mesma mensagem."""
    return normalize_text(text) in {"oi", "ola", "bom dia", "boa tarde", "boa noite"}


def is_close_request(text: str) -> bool:
    """Reconhece pedidos claros de encerramento sem capturar assuntos financeiros."""
    normalized = normalize_text(text)
    if not normalized:
        return False
    words = normalized.split()
    if len(words) <= 6 and _is_goodbye(words):
        return True
    if len(words) <= 8 and _is_explicit_decline_to_continue(words):
        return True

    remaining = [word for word in words if word not in _CLOSE_FILLERS]
    verbs = [word for word in remaining if word in _CLOSE_VERBS]
    if len(words) > 8 and not (verbs and any(word in _CLOSE_TARGETS for word in remaining)):
        return False
    allowed = _CLOSE_TARGETS | _CLOSE_VERBS | _CLOSE_SCOPE_INTENSIFIERS
    return bool(verbs) and all(word in allowed for word in remaining)


def _is_goodbye(words: list[str]) -> bool:
    """Aceita despedida curta, com eventuais palavras de cortesia."""
    return any(
        words[:len(phrase.split())] == phrase.split()
        and all(word in _CLOSE_FILLERS for word in words[len(phrase.split()):])
        for phrase in _GOODBYES
    )


def _is_explicit_decline_to_continue(words: list[str]) -> bool:
    """Reconhece apenas recusas de continuidade com objeto inequívoco."""
    if words[:3] == ["nao", "quero", "mais"]:
        remainder = [word for word in words[3:] if word not in _CLOSE_FILLERS]
        return len(remainder) == 1 and remainder[0] in {"nada", "atendimento", "conversa"}
    if words[:5] == ["nao", "preciso", "de", "mais", "nada"]:
        return all(word in _CLOSE_FILLERS for word in words[5:])
    return False


def is_affirmative(text: str) -> bool:
    """Reconhece respostas afirmativas curtas à confirmação pendente."""
    normalized = normalize_text(text)
    accepted = {
        "sim", "claro", "claro que sim", "claro aceito", "aceito", "pode", "pode ser", "vamos",
        "vamos la", "com certeza", "ok", "okay",
    }
    return normalized in accepted or normalized.startswith(("sim ", "aceito ", "pode ser "))


def is_concise_affirmative(text: str) -> bool:
    """Reconhece um aceite sem conteúdo adicional a interpretar."""
    return normalize_text(text) in {
        "sim", "claro", "claro que sim", "claro aceito", "aceito", "pode", "pode ser", "vamos",
        "vamos la", "com certeza", "ok", "okay",
    }


def is_explicit_interview_confirmation(text: str) -> bool:
    """Reconhece confirmações completas e isoladas dos dados da entrevista."""
    return normalize_text(text) in {
        "sim", "claro", "claro que sim", "claro aceito", "aceito", "pode", "pode ser", "vamos",
        "vamos la", "com certeza", "ok", "okay", "positivo", "confirmo", "confirmado",
        "correto", "esta correto", "pode prosseguir",
    }


def is_interview_data_confirmation(text: str) -> bool:
    """Aplica o aceite curto somente aos dados pendentes da entrevista."""
    normalized = normalize_text(text)
    acknowledgements = {
        "isso", "isso mesmo", "esta certo", "estao certos", "esta tudo certo",
        "esses dados estao corretos", "ja confirmei", "ja confirmei esses dados",
        "ja confirmei mil vezes", "perfeito isso mesmo",
    }
    if is_explicit_interview_confirmation(normalized) or normalized in acknowledgements:
        return True
    # Aceita confirmação completa seguida apenas de uma frase curta de continuidade.
    # Correções, condições e novos dados financeiros seguem para a IA.
    # Esse aceite de entrevista não substitui o consentimento para crédito.
    match = re.fullmatch(r"(.+?)(?: e)? (?:podemos|pode|vamos) (?:continuar|prosseguir|seguir)(?: por favor)?", normalized)
    return bool(match and (is_explicit_interview_confirmation(match[1]) or match[1] in acknowledgements))


def is_negative(text: str) -> bool:
    """Reconhece respostas negativas curtas à confirmação pendente."""
    normalized = normalize_text(text)
    accepted = {"nao", "negativo", "prefiro nao", "agora nao", "depois", "recuso"}
    return normalized in accepted or normalized.startswith(("nao ", "prefiro nao", "agora nao"))


def is_concise_negative(text: str) -> bool:
    """Reconhece uma recusa sem conteúdo adicional a interpretar."""
    return normalize_text(text) in {"nao", "negativo", "prefiro nao", "agora nao", "depois", "recuso"}


def wants_limit_increase_without_amount(text: str) -> bool:
    """Reconhece um pedido de aumento sem valor e sem pedido de encerramento."""
    source = text if isinstance(text, str) else ""
    if re.search(r"\d", source):
        return False
    normalized = normalize_text(source)
    if _CLOSE_ACTION.search(normalized):
        return False
    return bool(_LIMIT_INCREASE.search(normalized))


def wants_to_end_interview_only(text: str) -> bool:
    """Reconhece quando o cliente quer encerrar/apenas desistir da etapa da entrevista,
    não todo o atendimento. A frase precisa mencionar "entrevista" e carregar um sinal de
    interrupção. Se também nomear o atendimento/conversa, o pedido é global (retorna False).

    Exemplos positivos: "encerrar a entrevista", "não quero mais a entrevista",
    "pode encerrar a entrevista", "quero finalizar a entrevista", "desistir da entrevista".
    Exemplos negativos (pedido global): "encerrar o atendimento", "encerrar a conversa".
    """
    source = text if isinstance(text, str) else ""
    normalized = normalize_text(source)
    if "entrevista" not in normalized:
        return False
    words = normalized.split()
    # Se também nomeia o atendimento/conversa/sessão, o pedido é global, não da entrevista.
    if any(word in _CLOSE_TARGETS for word in words):
        return False
    return any(word in _INTERVIEW_STOP for word in words)


def is_global_close_request(text: str) -> bool:
    """True quando o pedido de encerramento nomeia explicitamente o atendimento/conversa
    (ou é uma despedida/recusa de continuidade). Usado para não perguntar confirmação em
    pedidos inequívocos mesmo dentro da entrevista."""
    normalized = normalize_text(text)
    if not normalized:
        return False
    words = normalized.split()
    if len(words) <= 6 and _is_goodbye(words):
        return True
    if len(words) <= 8 and _is_explicit_decline_to_continue(words):
        return True
    remaining = [word for word in words if word not in _CLOSE_FILLERS]
    has_close_verb = any(word in _CLOSE_VERBS for word in remaining)
    has_global_target = any(word in _CLOSE_TARGETS for word in remaining)
    has_global_scope = "tudo" in remaining
    return has_close_verb and (has_global_target or has_global_scope)


# Respostas à confirmação de encerramento durante a entrevista.
# O escopo global só é aceito quando nomeado explicitamente; "sim" confirma a
# primeira opção da pergunta (encerrar somente a entrevista), nunca o global.
_CLARIFY_PHRASES = (
    "o que", "oque", "como assim", "nao entendi", "entendi nao", "explique",
    "quer dizer", "pode repetir", "repetir", "pode dizer", "que opcoes",
    "quais as opcoes", "quais opcoes", "qual a diferenca", "nao sei",
)
_CLOSE_SCOPE_GLOBAL_PHRASES = (
    "encerrar tudo", "finalizar tudo", "encerre tudo", "encerra tudo",
    "todo o atendimento", "o atendimento todo", "a conversa toda",
    "a conversa inteira", "o atendimento inteiro",
)
_CLOSE_SCOPE_INTERVIEW_PHRASES = (
    "so a entrevista", "somente a entrevista", "apenas a entrevista",
    "a entrevista so", "encerrar a entrevista", "finalizar a entrevista",
    "encerre a entrevista", "encerra a entrevista", "cancelar a entrevista",
    "abortar a entrevista", "desistir da entrevista",
)


def close_scope_reply(text: str) -> str | None:
    """Classifica a resposta à confirmação pendente de encerramento.

    Retorna:
      - "global": pedido explícito de encerrar todo o atendimento;
      - "interview": encerra somente a entrevista (também "sim", que confirma a
        primeira — e mais segura — opção da pergunta);
      - "no": recusa de encerrar; a entrevista continua;
      - "clarify": pedido de esclarecimento da própria pergunta;
      - None: resposta não relacionada à confirmação (o chamador abandona o
        estado pendente e segue o fluxo normal, sem encerrar nada).
    """
    source = text if isinstance(text, str) else ""
    normalized = normalize_text(source)
    if not normalized:
        return None
    # Encerramento global exige escopo explícito; nunca é inferido de "sim".
    if any(phrase in normalized for phrase in _CLOSE_SCOPE_GLOBAL_PHRASES):
        return "global"
    if is_global_close_request(source):
        return "global"
    # "Sim" confirma apenas o encerramento da entrevista.
    if any(phrase in normalized for phrase in _CLOSE_SCOPE_INTERVIEW_PHRASES):
        return "interview"
    if wants_to_end_interview_only(source):
        return "interview"
    # "Não entendi" pede esclarecimento; não é recusa.
    if any(phrase in normalized for phrase in _CLARIFY_PHRASES):
        return "clarify"
    # Respostas curtas confirmam ou recusam a primeira opção: encerrar só a entrevista.
    if is_affirmative(source):
        return "interview"
    # A recusa mantém a entrevista em andamento.
    if is_negative(source):
        return "no"
    return None


def parse_limit_amount(text: str) -> str | None:
    """Retorna o valor somente quando toda a entrada é claramente numérica."""
    if isinstance(text, str) and _LIMIT_AMOUNT.fullmatch(text):
        return text.strip()
    return None


def extract_credentials(text: str, *, awaiting_birth_date: bool = False) -> Credentials:
    """Extrai CPF e data sem permitir que os dígitos do CPF virem data."""
    source = text if isinstance(text, str) else ""
    cpf_match = _FORMATTED_CPF.search(source) or _PLAIN_CPF.search(source)
    if cpf_match is None:
        labeled = re.search(r"(?i)\bcpf\b\s*[:=]?\s*([^\s,;]+)", source)
        if labeled:
            candidate = labeled.group(1).strip("()")
            cpf_value = normalize_cpf(candidate)
            if cpf_value:
                cpf_match = labeled
    cpf_value = normalize_cpf(cpf_match.group(0)) if cpf_match else None
    remainder = source[:cpf_match.start()] + " " + source[cpf_match.end():] if cpf_match else source

    date_matches = list(_DATE_CANDIDATE.finditer(remainder))
    marker = _BIRTH_MARKER.search(remainder)
    if marker:
        following = [date for date in date_matches if date.start() >= marker.end()]
        date_match = following[0] if following else next(
            (date for date in reversed(date_matches) if date.end() <= marker.start()), None
        )
    else:
        date_match = date_matches[0] if date_matches else None
    date_value = normalize_date(date_match.group(0)) if date_match else None
    if date_value is None and awaiting_birth_date and re.fullmatch(r"\s*\d{8}\s*", remainder):
        date_value = normalize_date(remainder.strip())
    return Credentials(cpf=cpf_value, birth_date=date_value)


def has_birth_date_candidate(text: str) -> bool:
    """Indica que há uma data explícita, mesmo quando ela não passa na validação."""
    source = text if isinstance(text, str) else ""
    return _DATE_CANDIDATE.search(source) is not None
