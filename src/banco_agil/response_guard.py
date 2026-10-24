"""Confere fatos financeiros e recupera perguntas incompatíveis com o estado atual."""
from __future__ import annotations

import math
import re
from datetime import datetime

from .intents import normalize_text
from .messages import format_brl, limit_request_reply
from .scoring import parse_money
from .tool_contracts import credit_clarification_pending, credit_confirmation_pending

_MONEY = re.compile(r"R\$\s*([\d]+(?:[.,]\d+)*)(?!\d)|([\d]+(?:[.,]\d+)*)\s*reais\b", re.I)
CONFIRM_INSTRUCTION = "me envie **confirmo** para continuar"


def interview_list(values: dict) -> str:
    """Lista só os campos validados da entrevista, preservando zero e falso."""
    labels = {"renda_mensal": "renda mensal", "tipo_emprego": "tipo de emprego",
              "despesas_fixas": "despesas fixas", "num_dependentes": "dependentes",
              "tem_dividas": "possui dívidas"}
    lines = []
    for field, label in labels.items():
        value = values.get(field)
        if value is None:
            continue
        formatted = format_brl(value) if field in {"renda_mensal", "despesas_fixas"} else (
            ("sim" if value else "não") if type(value) is bool else str(value))
        if field == "tipo_emprego" and value == "autonomo":
            formatted = "autônomo"
        lines.append(f"- {label}: {formatted}")
    return "\n".join(lines)


def pending_summary(session) -> str:
    """Dados estruturados visíveis vinculam a confirmação ao que será usado."""
    if not session.authenticated or session.ended:
        return ""
    pending = session.flow.pending_interview_answers
    if session.active_agent == "entrevista" and session.flow.interview_started and pending:
        lines = interview_list(pending)
        return "Dados para confirmação:\n\n" + lines if lines else ""
    if session.active_agent != "credito":
        return ""
    from .interview import has_interview_request
    if has_interview_request(session):
        return ""
    proposal = session.credit_proposal or {}
    if proposal and not proposal.get("executada"):
        if proposal.get("modalidade") == "indefinida":
            return (f"Dados da proposta: valor informado: {format_brl(proposal['valor'])}.\n\n"
                    "Esse valor é o novo limite total ou um acréscimo ao limite atual?")
        parts = [f"valor solicitado: {format_brl(proposal['valor'])}"]
        if proposal.get("modalidade") == "acrescimo":
            parts = [f"aumento de {format_brl(proposal['valor'])}"]
            if "novo_limite" in proposal:
                parts.append(f"novo limite: {format_brl(proposal['novo_limite'])}")
        summary = "Dados da proposta: " + "; ".join(parts) + "."
        if (session.active_agent == "credito" and not proposal.get("confirmada")
                and proposal.get("modalidade") in {"limite_total", "acrescimo"}
                and "novo_limite" in proposal):
            summary += "\n\nPara autorizar este pedido, " + CONFIRM_INSTRUCTION + ". Para alterar, informe a correção."
        return summary
    return ""


def unfinished_credit_confirmation_reply(session) -> str | None:
    """Uma proposta confirmada ainda pode estar aguardando a execução da análise."""
    if not credit_confirmation_pending(session):
        return None
    return ("Sua confirmação foi recebida, mas a análise deste pedido ainda não foi concluída. "
            "A proposta continua preservada neste atendimento. "
            "Envie uma nova mensagem para retomarmos a solicitação.\n\n"
            + pending_summary(session))


def interview_recovery_reply(session) -> str | None:
    """Explica os dados pendentes preservados sem expor nomes internos."""
    pending = session.flow.pending_interview_answers
    if not (session.authenticated and session.active_agent == "entrevista"
            and session.flow.interview_started and pending):
        return None
    lines = interview_list(pending)
    if not lines:
        return None
    reason = " por alta demanda" if session.last_failure_category == "http_429" else ""
    return (f"Não consegui concluir a resposta agora{reason}. Sua entrevista continua ativa. "
            "Mantive o que você informou.\n\nDados para confirmação:\n\n" + lines +
            "\n\nEssas informações ainda aguardam sua confirmação; " + CONFIRM_INSTRUCTION +
            ". Para alterar, informe a correção.")


def _result(session, name, *, current=False):
    fact = session.tool_facts.get(name, {})
    result = fact.get("result", {})
    if "erro" in result or (current and fact.get("turn") != session.turn):
        return {}
    return result


def _request(session, *, current=False):
    result = _result(session, "solicitar_aumento_limite", current=current)
    score = _result(session, "registrar_entrevista_e_calcular_score", current=current)
    return score.get("reanalise_automatica", {}) if score.get("reanalise_automatica", {}).get("executada") else result


def _safe_facts(session):
    request = _request(session, current=True)
    if "status_pedido" in request:
        return limit_request_reply(request)
    quote = _result(session, "consultar_cotacao", current=True)
    if quote and type(quote.get('valor_em_brl')) in (int, float):
        amount = f"{quote['valor_em_brl']:.4f}".replace('.', ',')
        return (f"A última cotação disponível de {quote['moeda']} é R$ {amount}. "
                f"Fonte: {quote.get('fonte') or 'não informada'}. Atualização da fonte: {quote.get('atualizado_em') or 'não informada'}.")
    limit = _result(session, "consultar_limite", current=True)
    if limit:
        return f"Seu limite de crédito atual é {format_brl(limit['limite_credito_atual'])}."
    score = _result(session, "registrar_entrevista_e_calcular_score", current=True)
    if score:
        return f"Seu score foi atualizado para {score['novo_score']}."
    summary = pending_summary(session)
    if summary:
        if credit_clarification_pending(session):
            return summary
        return summary + " Confira os dados antes de confirmar ou informe a correção."
    return "Não há confirmação dessa operação nos resultados do atendimento. Posso conferir os dados antes de prosseguir."


def confirmed_result_reply(session) -> str | None:
    """Apresenta apenas resultados bem-sucedidos deste turno."""
    request = _request(session, current=True)
    quote = _result(session, "consultar_cotacao", current=True)
    limit = _result(session, "consultar_limite", current=True)
    if ("status_pedido" in request or "limite_credito_atual" in limit
            or quote and type(quote.get("valor_em_brl")) in (int, float)):
        return _safe_facts(session)
    return None


def grounded_reply(session, text: str) -> str:
    """Texto natural é preservado quando os fatos verificáveis conferem.

    Sem fatos não é permitido anunciar sucesso. Divergências usam uma resposta
    de contingência com os dados efetivamente retornados, sem nova operação.
    Não se trata de um avaliador semântico geral de linguagem natural.
    """
    norm = normalize_text(text)
    proposal = session.credit_proposal or {}
    clarification_stage = (session.credit_clarification_turn == session.turn
                           or proposal.get("turno") == session.turn)
    unsafe_authorization = bool(re.search(r"\b(?:confirmo|confirme|confirma|autorize|autorizar)\b", norm)
                                and re.search(r"\b(?:pedido|solicitacao|proposta|limite|aumento)\b", norm))
    if (credit_clarification_pending(session) and (clarification_stage or unsafe_authorization)
            and not credit_clarification_question(text)):
        session.last_failure_category = "ungrounded_response"
        session.last_failure_reason = "A IA não esclareceu total ou acréscimo de uma proposta indefinida."
        confirmed = confirmed_result_reply(session)
        return (confirmed + "\n\n" if confirmed else "") + pending_summary(session)
    if session.authenticated and session.active_agent == "credito":
        from .interview import has_interview_request
        if has_interview_request(session) and _asks_known_credit_amount(text):
            from .messages import interview_credit_resume_message
            session.last_failure_category = "ungrounded_response"
            session.last_failure_reason = "A IA pediu novamente o valor do pedido vinculado à entrevista."
            return interview_credit_resume_message(session.desired_limit)
    if (session.authenticated and session.active_agent == "entrevista"
            and session.flow.interview_started and not session.flow.pending_interview_answers
            and not session.flow.interview_revision_requested
            and _repeated_interview_question(session, text)):
        from .interview import next_interview_prompt
        session.last_failure_category = "ungrounded_response"
        session.last_failure_reason = "A IA perguntou somente sobre dados já confirmados da entrevista."
        return next_interview_prompt(session.flow.interview_answers, {})
    if (session.authenticated and session.active_agent == "entrevista"
            and session.flow.interview_started and _extra_interview_question(text)):
        from .interview import next_interview_prompt
        session.last_failure_category = "ungrounded_response"
        session.last_failure_reason = "A IA pediu detalhes fora dos cinco campos da entrevista."
        if session.flow.pending_interview_answers:
            return pending_summary(session) + "\n\n" + CONFIRM_INSTRUCTION + "."
        return next_interview_prompt(session.flow.interview_answers, {})
    if session.authenticated:
        for sentence in re.split(r"(?<=[.!?;,])\s+", text):
            wording = normalize_text(sentence)
            requests = re.finditer(r"\b(?:(?:informe|informar|forneca|fornecer|digite|envie|enviar|confirme)\b.{0,80}"
                                   r"|qual (?:e )?(?:o |a )?(?:seu |sua )?|(?:preciso|necessito) (?:do |da |de )(?:seu |sua )?)"
                                   r"\b(?:cpf|data de nascimento)\b", wording)
            for request in requests:
                # Uma proibição posterior não justifica um pedido anterior de credenciais.
                if re.search(r"\b(?:nao|nunca|nem)(?:\s+\w+){0,3}\s*$", wording[:request.start()]):
                    continue
                session.last_failure_category = "ungrounded_response"
                session.last_failure_reason = "O LLM solicitou identificação já validada na sessão."
                return "Sua identidade já está confirmada. Podemos continuar com sua solicitação."
    statements = [part for part in re.split(r"(?<=[.!?])\s+", text)
                  if not part.endswith("?") and not re.match(r"\s*(?:se|caso)\b", part, re.I)]
    claims = normalize_text(" ".join(statements))
    request = _request(session, current=True)
    limit = _result(session, "consultar_limite")
    quote = _result(session, "consultar_cotacao")
    score = _result(session, "registrar_entrevista_e_calcular_score", current=True)
    amounts = []
    for match in _MONEY.finditer(text):
        try:
            amounts.append(parse_money(match.group(1) or match.group(2)))
        except ValueError:
            return _reject(session)
    # Também confere números sem "R$" quando ligados explicitamente ao campo.
    if not amounts:
        bare = re.finditer(r"\b(?:limite|cota[cç][aã]o|d[oó]lar|euro|libra)\b[^.!?\n]{0,50}?"
                           r"(?<![\d/:-])(\d+(?:[.,]\d+)*)(?![\d/:-])", text, re.I)
        for match in bare:
            try:
                amounts.append(parse_money(match.group(1)))
            except ValueError:
                return _reject(session)
    positive_claims = " ".join(normalize_text(part) for part in statements
                               if not re.search(r"\b(?:nao|nunca|nem|nenhum|nenhuma)\b", normalize_text(part)))
    pending_labels = {"renda_mensal": "renda", "tipo_emprego": "emprego",
                      "despesas_fixas": "despesas", "num_dependentes": "dependentes", "tem_dividas": "dividas"}
    pending_targets = "|".join(["dados", "informacoes", "respostas"] + [
        pending_labels[k] for k in session.flow.pending_interview_answers])
    if (session.active_agent == "entrevista" and session.flow.pending_interview_answers
            and re.search(r"\b(?:" + pending_targets + r")\b.{0,120}\b(?:confirmad[oa]s?|registrad[oa]s?)\b", positive_claims)):
        session.last_failure_category = "ungrounded_response"
        session.last_failure_reason = "A IA anunciou confirmação de dados ainda pendentes da entrevista."
        return (pending_summary(session) + "\n\nEssas informações ainda aguardam confirmação; "
                + CONFIRM_INSTRUCTION + ". Para alterar, informe a correção.")
    approved = bool(re.search(r"\b(?:aprovad[oa]|aprovei|aprovamos)\b", positive_claims))
    rejected = "reprovad" in claims or ("nao foi aprovado" in claims)
    changed = bool(re.search(r"\b(?:aumentei|atualizei|(?:limite.*(?:foi|esta|ja|agora).*(?:aumentad|atualizad|ativo)))", positive_claims))
    registered = bool(re.search(r"\b(?:registrei|registramos|pedido.*registrad|solicitacao.*registrad)", positive_claims))
    violation = False
    # Não há fila de análise em segundo plano. O recibo registra a gravação,
    # mas não comprova uma promessa de resposta futura.
    background_credit = (session.active_agent == "credito" and (
        re.search(r"\b(?:seu pedido|sua solicitacao|seu aumento|a analise|o pedido|a solicitacao)\b.{0,140}"
                  r"\b(?:esta|segue|continua)\b.{0,30}\b(?:processad[oa]|processamento|analise)\b", positive_claims)
        or re.search(r"\bestamos (?:processando|analisando)\b.{0,80}\b(?:pedido|solicitacao|aumento)\b", positive_claims)
        or re.search(r"\b(?:em breve|voce recebera|retornaremos|avisaremos)\b.{0,100}"
                     r"\b(?:resposta|retorno|resultado|aprovacao|analise)\b", positive_claims)))
    if background_credit:
        violation = True
    if approved and request.get("status_pedido") != "aprovado":
        violation = True
    if rejected and request.get("status_pedido") != "reprovado":
        violation = True
    if changed and not (request.get("status_pedido") == "aprovado" and request.get("limite_aplicado")):
        violation = True
    if registered and not request.get("status_pedido"):
        violation = True
    if "score" in positive_claims and re.search(r"\b(?:atualizad|recalculad|calculei)", positive_claims) and not score:
        violation = True
    score_value = re.search(r"\bscore(?: foi)?(?: atualizado| recalculado)?(?: para| e| de| agora)? (\d{1,4})\b", claims)
    if score_value and score_value.group(1) != str(score.get("novo_score")):
        violation = True
    if re.search(r"\b(?:login falhou|falha no login|falha de autenticacao|nao foi possivel autenticar)\b", norm):
        auth = _result(session, "autenticar_cliente", current=True)
        violation |= auth.get("sucesso") is not False
    quote_statement = bool(re.search(r"\b(?:cotacao|dolar|euro|libra)\b", norm)) and bool(amounts)
    limit_statement = bool(re.search(r"\b(?:seu limite|limite atual|limite de credito|novo limite)\b", norm)) and bool(amounts)
    # Valores de uma pergunta/proposta não são uma confirmação de saldo/limite.
    question = "?" in text and not (approved or rejected or changed or registered)
    if question and amounts and pending_summary(session):
        proposal = session.credit_proposal or {}
        expected = [value for field, value in {**session.flow.interview_answers, **session.flow.pending_interview_answers}.items()
                    if field in {"renda_mensal", "despesas_fixas"}]
        expected.extend(proposal[key] for key in ("valor", "novo_limite", "limite_base") if key in proposal)
        expected.extend(value for value in (limit.get("limite_credito_atual"), quote.get("valor_em_brl")) if value is not None)
        violation |= any(not any(math.isclose(value, actual, rel_tol=0, abs_tol=0.00005) for actual in expected) for value in amounts)
    if not question and not amounts and re.search(r"\b(?:mil|reais)\b", norm) and re.search(r"\b(?:limite|cotacao)\b", norm):
        # Não adivinha números por extenso em uma afirmação financeira do modelo.
        violation = True
    if amounts and not question:
        expected = []
        if quote_statement:
            expected = [quote.get("valor_em_brl")]
        elif limit_statement or approved or rejected or changed:
            previous = _request(session)
            expected = [request.get("novo_limite_solicitado"), request.get("limite_credito_atual"),
                        limit.get("limite_credito_atual"), previous.get("limite_credito_atual")]
        else:
            expected = [v for source in (session.flow.interview_answers, session.flow.pending_interview_answers,
                                        session.credit_proposal or {}) for k, v in source.items()
                        if k in {"renda_mensal", "despesas_fixas", "valor", "novo_limite"}]
        expected = [v for v in expected if type(v) in (int, float)]
        violation |= any(not any(math.isclose(value, actual, rel_tol=0, abs_tol=0.00005) for actual in expected)
                         for value in amounts)
    if quote_statement:
        for code, aliases in {"USD": ("dolar",), "EUR": ("euro",), "GBP": ("libra",)}.items():
            if any(alias in norm.split() for alias in aliases) and quote.get("moeda") != code:
                violation = True
        source = normalize_text(quote.get("fonte", ""))
        if any(provider in norm and provider not in source for provider in ("awesomeapi", "frankfurter", "bce")):
            violation = True
        def dates(value):
            found = set()
            for token in re.findall(r"\b(?:\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4})\b", value):
                try:
                    found.add(datetime.strptime(token, "%Y-%m-%d" if '-' in token else "%d/%m/%Y").date())
                except ValueError:
                    found.add(None)
            return found
        violation |= bool(dates(text) - dates(str(quote.get("atualizado_em", ""))))
        for clock in re.findall(r"\b\d{2}:\d{2}(?::\d{2})?\b", text):
            if clock not in str(quote.get("atualizado_em", "")):
                violation = True
    return _reject(session) if violation else text


def credit_clarification_question(text: str) -> bool:
    """Valida a próxima etapa da resposta sem reinterpretar a entrada do cliente."""
    questions = " ".join(part for part in re.split(r"(?<=[.!?])\s+", text) if "?" in part)
    wording = normalize_text(questions)
    return bool(re.search(r"\b(?:limite total|novo limite|limite final|total do limite)\b", wording)
                and re.search(r"\b(?:acrescimo|acrescentar|adicionar|somar|aumentar em)\b", wording)
                and not re.search(r"\b(?:confirmo|confirmar|confirme|confirma|autorizar|autorize)\b",
                                  normalize_text(text)))


def _reject(session):
    session.last_failure_category = "ungrounded_response"
    session.last_failure_reason = "Afirmação financeira sem correspondência nos resultados estruturados."
    return _safe_facts(session)


def _extra_interview_question(text: str) -> bool:
    """Rejeita os campos excedentes sem descartar exemplos válidos de dívidas."""
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        wording = normalize_text(sentence)
        if "?" not in sentence:
            continue
        if "reserva de emergencia" in wording:
            return True
        if (re.search(r"\b(?:outros?|mais|adicionais) (?:creditos?|emprestimos?|financiamentos?)\b", wording)
                and not re.search(r"\bdividas?\b", wording)):
            return True
        if re.search(r"\b(?:valor|saldo|parcelas?|taxas?)\s+"
                     r"(?:(?:d[aoe]s?|parcelas|seus|suas|dess[ae]s?)\s+){0,4}"
                     r"(?:emprestimos?|financiamentos?|dividas?)\b", wording):
            return True
    return False


def _asks_known_credit_amount(text: str) -> bool:
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        wording = normalize_text(sentence)
        if "?" not in sentence or not re.search(r"\b(?:qual|quanto|informe|informar|diga|fornecer|forneca)\b", wording):
            continue
        if re.search(r"\bnovo limite\b|\blimite(?: total)?\b.{0,45}\b(?:desej|solicitar)"
                     r"|\bvalor\b.{0,30}\b(?:desejado|solicitar|aumento)\b", wording):
            return True
    return False


def _repeated_interview_question(session, text: str) -> bool:
    """Recupera uma pergunta explicitamente redundante sem interpretar texto do cliente.

    Assuntos novos, correções e perguntas sobre campos ausentes continuam
    sob responsabilidade do agente de conversa.
    """
    from .interview import missing_interview_fields
    missing = set(missing_interview_fields(session.flow.interview_answers))
    if not missing:
        return False
    questions = " ".join(sentence for sentence in re.split(r"(?<=[.!?])\s+", text) if "?" in sentence)
    wording = normalize_text(questions)
    if not wording or re.search(r"\b(?:score|limite|cotacao|cambio|dolar|euro|moeda)\b", wording):
        return False
    references = {
        field for field, pattern in {
            "renda_mensal": r"\b(?:renda|salario|rendimentos)\b",
            "tipo_emprego": r"\b(?:emprego|tipo de trabalho)\b",
            "despesas_fixas": r"\b(?:despesas|gastos)\b",
            "num_dependentes": r"\bdependentes?\b",
            "tem_dividas": r"\b(?:dividas?|endividamento)\b",
        }.items() if re.search(pattern, wording)
    }
    return bool(references & session.flow.interview_answers.keys()) and not references & missing
