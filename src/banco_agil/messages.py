"""Textos apresentados ao cliente e seus formatos determinísticos."""
from __future__ import annotations

from .statuses import STATUS_APROVADO, STATUS_REPROVADO

GREETING = "Olá! Bem-vindo ao Banco Ágil. Para começar, poderia me informar o seu CPF?"
FALLBACK = "Desculpe, estou com uma instabilidade momentânea. Pode tentar novamente em instantes?"
LLM_UNAVAILABLE = (
    "Estou com uma indisponibilidade técnica temporária. Sua sessão continua ativa; "
    "por segurança, confirme o status antes de repetir uma operação."
)
LLM_RATE_LIMITED = (
    "Estou com uma indisponibilidade temporária por alta demanda. Sua sessão continua ativa; por segurança, "
    "confirme o status antes de repetir uma operação. Aguarde alguns instantes e envie uma nova mensagem para "
    "continuarmos de onde paramos."
)
LLM_INPUT_TOO_LONG = (
    "Recebi uma mensagem muito extensa para processar com segurança. Envie em partes menores, começando pela "
    "informação mais importante. Seu atendimento continua ativo."
)
EMPTY_REPLY = "Desculpe, não consegui entender. Pode reformular a sua mensagem?"
CLOSED = "Este atendimento foi encerrado. Inicie uma nova conversa quando quiser."
LOOP_GUARD_REPLY = (
    "Não consegui concluir essa etapa com os dados enviados. Revise a informação solicitada "
    "ou escolha outro assunto para eu ajudar."
)
REPEAT_GUARD_REPLY = (
    "Percebi que estamos repetindo a mesma etapa e, para não prolongar o atendimento, vou encerrar por aqui. "
    "Você pode iniciar um novo quando quiser. Obrigado por escolher o Banco Ágil!"
)
INTERVIEW_START = "Vamos começar. Qual é a sua renda mensal?"
INTERVIEW_DECLINED = (
    "Tudo bem. Posso ajudar com consulta de limite, câmbio ou outro assunto. "
    "O que você gostaria de fazer? Se preferir, posso encerrar a conversa."
)
INTERVIEW_CLOSE_AMBIGUOUS = (
    "Você gostaria de encerrar apenas a entrevista e voltar para o atendimento de crédito? "
    "Responda 'sim' para encerrar só a entrevista, 'não' para continuar a entrevista, "
    "ou 'encerrar tudo' para encerrar todo o atendimento."
)
INTERVIEW_CLOSE_CANCELLED = (
    "Tudo bem, não vou encerrar nada. Continuamos a entrevista; é só enviar sua resposta."
)
LIMIT_AMOUNT_PROMPT = "Informe o valor desejado para o novo limite."
LIMIT_AMOUNT_REQUEST = (
    "Posso verificar o aumento. Qual seria o novo limite desejado? "
    "Informe o valor em reais, por exemplo, R$ 6.000,00."
)
TRIAGE_AUTHENTICATED = "Olá, {name}. Sua identidade foi confirmada. Como posso ajudar?"
TRIAGE_BIRTH_DATE = "Recebi seu CPF. Para confirmar sua identidade, qual é a sua data de nascimento?"
TRIAGE_CPF = "Antes de ajudar, preciso identificar você. Qual é o seu CPF?"
TRIAGE_CONTINUE = "Entendi. Pode me dizer novamente como posso ajudar?"
LIMIT_RETRY_PROMPT = "Informe outro valor para o novo limite."
INTERVIEW_RETRY_PROMPT = "Corrija somente a resposta solicitada."
CURRENCY_RETRY_PROMPT = "Informe um código de moeda de três letras."


def format_brl(value: float) -> str:
    """Formata um valor numérico como moeda brasileira."""
    return f"R$ {value:,.2f}".replace(",", "_").replace(".", ",").replace("_", ".")


def auth_restart_message(result: dict) -> str:
    remaining = result.get("tentativas_restantes")
    if remaining == 2:
        allowance = "Você ainda pode fazer mais duas tentativas."
    elif remaining == 1:
        allowance = "Você ainda pode fazer mais uma tentativa."
    else:
        allowance = ""
    return f"Não foi possível autenticar com os dados informados. {allowance} Vamos recomeçar; informe seu CPF."


def agent_fallback_message(agent_name: str) -> str:
    if agent_name == "credito":
        return (
            "Entendi que você precisa de ajuda com seu limite. Informe o valor desejado ou peça para "
            "consultar seu limite atual."
        )
    if agent_name == "entrevista":
        return (
            "Não consegui processar esta etapa agora. Nenhuma alteração foi feita no seu cadastro; "
            "por favor, envie novamente sua última resposta para continuarmos."
        )
    if agent_name == "cambio":
        return "Não consegui consultar a cotação agora. Informe novamente a moeda que deseja consultar, como USD ou EUR."
    return FALLBACK


_ERROR_DETAILS = {
    "valor_nao_positivo": "O valor precisa ser maior que zero.",
    "valor_negativo": "O valor precisa ser maior que zero.",
    "valor_invalido": "Não consegui interpretar esse número como um valor válido em reais.",
    "valor_acima_do_limite": "Não consegui interpretar esse número como um valor válido em reais.",
    "limite_menor_ou_igual_ao_atual": "O limite solicitado precisa ser maior que o limite atual.",
    "emprego_invalido": "Informe um tipo de emprego válido.",
    "dependentes_invalidos": "Informe um número válido de dependentes.",
    "dividas_invalidas": "Informe sim ou não para dívidas ativas.",
    "entrevista_incompleta": "Ainda faltam respostas obrigatórias da entrevista.",
    "dados_entrevista_nao_conferem": "Os dados não correspondem às respostas já informadas.",
    "cliente_nao_encontrado": "Não encontrei o cadastro para continuar.",
    "argumentos_invalidos": "Não consegui interpretar os dados informados.",
    "erro_tecnico": "Ocorreu um problema técnico ao validar os dados.",
}


def credit_tool_error_reply(result: dict) -> str:
    return requested_input_error("limit", 1, result.get("codigo", "erro_tecnico"), None)


def requested_input_error(kind: str, attempt: int, code: str, prompt: str | None) -> str:
    if attempt > 3:
        if kind == "limit":
            return "O valor informado não é válido. Se quiser, posso ajudar com outra dúvida ou encerrar a conversa."
        return "A resposta informada não é válida. Se quiser, posso ajudar com outra dúvida ou encerrar a conversa."
    if kind == "limit":
        detail = _ERROR_DETAILS.get(code, "Não consegui validar esse valor para o pedido.")
        return f"O valor informado é inválido. {detail} Informe outro valor em reais ou escolha outro assunto."
    if kind.startswith("interview"):
        requested = prompt or "Corrija somente a resposta solicitada."
        detail = _ERROR_DETAILS.get(code)
        explanation = f" {detail}" if detail else ""
        return f"A resposta informada é inválida.{explanation} Tentativa {attempt} de 3. {requested}"
    if kind == "currency":
        return (
            f"A moeda informada é inválida. Tentativa {attempt} de 3. "
            "Informe um código de três letras, como USD, EUR ou GBP."
        )
    return f"A resposta informada é inválida. Tentativa {attempt} de 3. Tente novamente."


def limit_request_reply(result: dict) -> str:
    if "erro" in result:
        return credit_tool_error_reply(result)
    if result.get("status_pedido") == STATUS_REPROVADO:
        amount = format_brl(result["novo_limite_solicitado"])
        return (
            f"Seu pedido de aumento para {amount} foi {STATUS_REPROVADO} pelo score atual. "
            "Posso encaminhá-lo para uma entrevista financeira para reavaliar seu score. "
            "Você gostaria de começar?"
        )
    amount = format_brl(result["limite_credito_atual"])
    if not result.get("limite_aplicado"):
        return (
            "Seu pedido foi aprovado, mas não consegui aplicar o novo limite agora. "
            "Nenhuma confirmação de atualização foi feita; tente consultar novamente em instantes."
        )
    return f"Seu pedido foi {STATUS_APROVADO} e o novo limite de {amount} já está ativo."


def rejected_limit_offer_message(amount: float) -> str:
    return (
        f"Seu pedido de aumento para {format_brl(amount)} não foi aprovado pelo score atual. "
        "Posso encaminhá-lo para uma entrevista financeira para reavaliar seu score. "
        "Você gostaria de começar?"
    )


def loop_guard_reply(agent_name: str) -> str:
    if agent_name == "credito":
        return "Esse valor não foi aceito e não vou repetir a mesma solicitação. Informe um limite maior que zero ou escolha outro assunto."
    if agent_name == "entrevista":
        return "Uma ou mais respostas da entrevista não passaram na validação. Envie apenas a informação corrigida ou desista da entrevista."
    if agent_name == "triagem":
        return "Não consegui identificar todos os dados dessa mensagem. Envie seu CPF para continuarmos."
    if agent_name == "cambio":
        return "Não consegui consultar essa moeda. Informe um código de três letras, como USD, EUR ou GBP."
    return LOOP_GUARD_REPLY


def interview_credit_resume_message(amount) -> str:
    return (f"O pedido anterior de {format_brl(amount)} está preservado. "
            "Para reavaliá-lo, precisamos concluir a entrevista financeira.\n\n"
            "Quer retomar a entrevista de onde paramos?")
