"""Formata listas, confirmações, perguntas e moeda para exibição no chat."""
import re

from .response_guard import CONFIRM_INSTRUCTION, interview_list
from .interview import has_interview_request


# Preserva o Markdown de código e expressões matemáticas.
# O cifrão de R$ não inicia uma expressão matemática.
_CODE_OR_MATH = re.compile(
    r"(`+)[\s\S]*?\1|(~{3,})[\s\S]*?\2"
    r"|(?<![\w\\$])\$\$[\s\S]*?\$\$"
    r"|(?<![\w\\$])\$(?!\$)(?:\\.|[^\\$\n])+\$"
)


def chat_markdown(text: str) -> str:
    """Preserva R$ fora de código e fórmulas, sem alterar o texto guardado na sessão."""
    pieces = []
    previous = 0
    for match in _CODE_OR_MATH.finditer(text):
        pieces.extend((text[previous:match.start()].replace("R$", r"R\$"), match.group()))
        previous = match.end()
    pieces.append(text[previous:].replace("R$", r"R\$"))
    return "".join(pieces)


_INTERVIEW_SUMMARY = re.compile(
    r"(?im)(?:resumo das informa[çc][õo]es fornecidas|resumo dos dados|resumo da entrevista):"
    r"(?:[ \t]*\n(?:[ \t]*\n)*(?:- [^\n]*(?:\n|$))+"
    r"|(?=[ \t]*[^\s])[^\n]*?(?:\.(?=\s+[A-ZÁÉÍÓÚÂÊÔÃÕÇ])|(?=\n|$))[ \t]*)"
)


def customer_reply(session, text: str) -> str:
    """Formata a mensagem sem confirmar dados ou autorizar operações.

    A conversa continua sendo escrita pela IA. O resumo dos dados pendentes
    e a instrução de confirmação usam o estado validado da sessão.
    """
    if session.authenticated and session.active_agent == "entrevista":
        values = {**session.flow.interview_answers, **session.flow.pending_interview_answers}
        lines = interview_list(values)
        if lines and _INTERVIEW_SUMMARY.search(text):
            from .response_guard import pending_summary
            pending = pending_summary(session)
            if pending:
                # Exibe cada campo uma vez, separando os dados confirmados da nova proposta.
                # A próxima pergunta não deve exigir confirmação da entrevista inteira.
                text = text.replace("\n\n" + pending, "")
                confirmed = {k: v for k, v in session.flow.interview_answers.items()
                             if k not in session.flow.pending_interview_answers}
                known = interview_list(confirmed)
                lines = ("Informações já confirmadas:\n\n" + known + "\n\n" if known else "") + pending
            text = _INTERVIEW_SUMMARY.sub("Resumo das informações fornecidas:\n\n" + lines + "\n\n", text).strip()
    pending_interview = (session.authenticated and session.active_agent == "entrevista"
                         and session.flow.interview_started
                         and any(value is not None for value in session.flow.pending_interview_answers.values()))
    proposal = session.credit_proposal or {}
    pending_credit = (session.authenticated and session.active_agent == "credito"
                      and not has_interview_request(session)
                      and not proposal.get("executada") and not proposal.get("confirmada")
                      and proposal.get("modalidade") in {"limite_total", "acrescimo"}
                      and proposal.get("novo_limite") is not None)
    if not session.ended and (pending_interview or pending_credit):
        # Mantém uma única instrução de confirmação, sem orientações concorrentes.
        text = re.sub(r"(?:Para (?:autorizar este pedido|confirmar esses dados), )?"
                      r"responda [“\"']?confirmo[”\"']?", CONFIRM_INSTRUCTION, text, flags=re.I)
        if re.search(re.escape(CONFIRM_INSTRUCTION), text, flags=re.I):
            text = re.sub(re.escape(CONFIRM_INSTRUCTION), CONFIRM_INSTRUCTION, text, flags=re.I)
        else:
            text = text.rstrip() + "\n\n" + CONFIRM_INSTRUCTION + ". Para alterar, informe a correção."
    return separate_questions(text)


def separate_questions(text: str) -> str:
    """Separa perguntas com duas quebras de linha, preservando números e código."""
    def plain(value):
        pieces = re.split(r"(?<=[.!?])(\s+)", value)
        for index in range(1, len(pieces), 2):
            if "?" in pieces[index + 1] and pieces[index].strip() == "":
                pieces[index] = "\n\n"
        return "".join(pieces)

    pieces, previous = [], 0
    for match in _CODE_OR_MATH.finditer(text):
        pieces.extend((plain(text[previous:match.start()]), match.group()))
        previous = match.end()
    pieces.append(plain(text[previous:]))
    return "".join(pieces)
