"""Seleciona o contexto sem alterar a sessão compartilhada."""
from __future__ import annotations

import json

from .flow import Awaiting
from .interview import missing_interview_fields
from .tool_contracts import credit_clarification_pending, credit_confirmation_pending


_CREDIT = {"consultar_limite", "solicitar_aumento_limite", "registrar_entrevista_e_calcular_score"}
_INTERVIEW = {"registrar_respostas_entrevista", "registrar_entrevista_e_calcular_score", "confirmacao_entrevista",
              "confirmar_dados_entrevista"}
_EXCHANGE = {"consultar_cotacao"}
_CONTROL = {"autenticar_cliente", "transferir_para_agente", "encerrar_atendimento", "registrar_entrada_invalida"}
_RELEVANT = {
    "triagem": _CONTROL | {"registrar_intencao_credito"},
    "credito": _CONTROL | _CREDIT | _INTERVIEW | {"registrar_intencao_credito"},
    "entrevista": _CONTROL | _INTERVIEW | {"registrar_intencao_credito"},
    "cambio": _CONTROL | _EXCHANGE | {"registrar_intencao_credito"},
}
_KNOWN = set().union(*_RELEVANT.values())
_FACTS = {
    "triagem": set(), "credito": _CREDIT, "entrevista": set(), "cambio": _EXCHANGE,
}
_FACT_KEYS = {"limite_credito_atual", "novo_limite_solicitado", "status_pedido", "limite_aplicado",
              "novo_score", "moeda", "valor_em_brl", "fonte", "atualizado_em", "codigo", "erro"}


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _unsettled(result) -> bool:
    if not isinstance(result, dict):
        return False
    reanalysis = result.get("reanalise_automatica")
    if isinstance(reanalysis, dict) and reanalysis.get("executada"):
        return "erro" in reanalysis or _unsettled(reanalysis)
    return (result.get("status_pedido") == "pendente"
            or result.get("status_pedido") == "aprovado" and result.get("limite_aplicado") is False)


def session_context(agent: str, s) -> str:
    """Mantém a autenticação, as pendências e a etapa atual no contexto."""
    ctx = [f"Cliente autenticado: {s.nome}. Identidade já validada; não peça CPF ou data de nascimento novamente."
           if s.authenticated else "Cliente NÃO autenticado."]
    if agent == "triagem" and s.authenticated:
        ctx.append("Etapa atual: identificação do assunto e encaminhamento. Não reinicie a autenticação. "
                   "Consultas e aumentos de limite seguem para credito; valores já coletados permanecem na sessão.")
    elif agent == "triagem":
        if s.flow.awaiting is Awaiting.BIRTH_DATE and s.flow.pending_cpf:
            ctx.append("Etapa atual: CPF recebido localmente, identidade ainda não validada. "
                       "Falta a data de nascimento. Não peça o CPF novamente nem reconstrua o valor redigido.")
        else:
            ctx.append("Etapa atual: coleta de identificação; falta o CPF. "
                       "Conduza a conversa sem presumir autenticação nem propor uma ação de validação.")
    if s.flow.awaiting is Awaiting.INTERVIEW_OFFER:
        ctx.append("Contexto: o cliente foi perguntado se aceita a entrevista financeira.")
    elif s.flow.awaiting is Awaiting.LIMIT_AMOUNT:
        ctx.append("Contexto: aguardando o valor do novo limite desejado.")
    elif s.flow.awaiting is Awaiting.CLOSE_CONFIRM:
        ctx.append("Há confirmação pendente sobre encerrar a entrevista ou todo o atendimento; preserve o escopo.")
    if s.flow.input_prompt and (agent in {"credito", "entrevista"} or not s.authenticated):
        ctx.append("Esclarecimento pendente: " + s.flow.input_prompt)
    proposal = s.credit_proposal or {}
    if proposal and not proposal.get("executada"):
        ctx.append("Intenção de crédito ainda não executada, preservada entre agentes: " + _json(proposal)
                   + ". Confirmação da entrevista não autoriza este pedido.")
    if credit_clarification_pending(s):
        ctx.append("Falta esclarecer se o valor informado é o novo limite total ou um acréscimo ao limite atual. "
                   "Não peça confirmação de uma operação ainda indefinida. Um aceite não escolhe o significado. "
                   "Interprete o esclarecimento do cliente e registre a proposta definida antes de pedir autorização.")
    if credit_confirmation_pending(s):
        ctx.append("O cliente confirmou a proposta neste turno. Use solicitar_aumento_limite com o novo_limite "
                   "preservado antes de finalizar a resposta. Não transfira nem colete novamente. "
                   "Não existe análise agendada: apresente o resultado real da ferramenta, sem prometer retorno futuro.")
    if s.credit_operation:
        ctx.append("Pedido financeiro com resultado incerto, valor " + str(s.credit_operation.get("amount"))
                   + ". Preserve o recibo da sessão; não registre um novo pedido para substituir o anterior.")
    if s.desired_limit is not None and (agent == "credito" or (
            agent == "entrevista" and s.last_request_status == "reprovado")):
        ctx.append("Último pedido formal: " + _json({"id": s.last_request_id, "limite_total": s.desired_limit,
                                                     "status": s.last_request_status})
                   + ". Reanálise usa somente essa origem, sem inventar outro pedido.")
    facts = {}
    for name, fact in s.tool_facts.items():
        # Os resultados deste turno acompanham as mudanças de agente e os pedidos mistos.
        unfinished = _unsettled(fact["result"])
        if name not in _FACTS[agent] and fact.get("turn") != s.turn and not unfinished:
            continue
        result = {key: value for key, value in fact["result"].items() if key in _FACT_KEYS}
        if result:
            facts[name] = {"turno": fact.get("turn"), **result}
        if unfinished:
            ctx.append("Há um resultado financeiro pendente de conclusão. Não anuncie aplicação do limite nem repita a operação; consulte o status ao retomar o crédito.")
    if facts:
        ctx.append("Resultados estruturados disponíveis (resultados antigos não autorizam nova operação): " + _json(facts))
    if (agent == "credito" and s.automatic_reanalysis_source_id is not None
            and s.automatic_reanalysis_result is not None):
        ctx.append("Reanálise automática já tentada: " + _json({
            "origem": s.automatic_reanalysis_source_id, "resultado": s.automatic_reanalysis_result,
            "resposta_pendente": s.automatic_reanalysis_response_pending,
        }) + ". Informe o resultado confirmado; não registre outro pedido.")
    if s.flow.interview_started:
        if agent in {"credito", "entrevista"}:
            ctx.append("Entrevista ativa. Dados confirmados pelo cliente (não perguntar novamente): "
                       + _json(s.flow.interview_answers))
            if s.flow.pending_interview_answers:
                ctx.append("Proposta da entrevista aguardando confirmação explícita: "
                           + _json(s.flow.pending_interview_answers) + ". Não calcule score nem confirme por conta própria.")
            else:
                ctx.append("Não há confirmação pendente: não reconfirme valores preservados. Campos ainda necessários: "
                           + _json(missing_interview_fields(s.flow.interview_answers)))
                if s.flow.interview_question_pending and missing_interview_fields(s.flow.interview_answers):
                    ctx.append("O aceite ou a confirmação desta mensagem já foi processado localmente. "
                               "Formule a próxima pergunta sobre um campo ausente; não proponha registrar "
                               "novamente esses dados nem executar confirmacao_entrevista, que é um resultado local.")
            if s.flow.interview_revision_requested:
                ctx.append("Interprete a última mensagem: aceite sem ressalvas confirma a proposta pela ação; "
                           "correções precisam de nova proposta e confirmação; dúvidas exigem esclarecimento.")
        else:
            ctx.append("Há uma entrevista em andamento preservada, "
                       + ("com dados aguardando confirmação. " if s.flow.pending_interview_answers else "sem confirmação pendente. ")
                       + "Retome pelo agente adequado; não reinicie a coleta e não use um aceite deste assunto para confirmar a entrevista.")
    return "\n" + "\n".join(ctx)


def _question_stage(agent: str, s) -> str:
    base = {"autenticado": s.authenticated}
    if agent == "entrevista":
        base.update(ativa=s.flow.interview_started, confirmados=s.flow.interview_answers,
                    pendentes=s.flow.pending_interview_answers, revisao=s.flow.interview_revision_requested,
                    encerramento=s.flow.awaiting is Awaiting.CLOSE_CONFIRM,
                    pedido=s.last_request_id, valor_pedido=s.desired_limit)
    elif agent == "credito":
        base.update(proposta=s.credit_proposal, pedido=s.last_request_id, status=s.last_request_status,
                    reanalise=s.automatic_reanalysis_source_id)
    elif agent == "triagem":
        base.update(aguardando=s.flow.awaiting.value, falhas=s.auth_failures, cpf_pendente=bool(s.flow.pending_cpf))
    else:
        base.update(cotacao=s.tool_facts.get("consultar_cotacao", {}).get("turn"))
    return _json(base)


def remember_question(s, text: str) -> None:
    """Guarda uma pergunta exibida com a etapa correspondente."""
    if not s.ended and "?" in text:
        s.agent_questions[s.active_agent] = {"text": text, "stage": _question_stage(s.active_agent, s)}


def _successful(result) -> bool:
    return (isinstance(result, dict) and "erro" not in result and result.get("sucesso") is not False
            and result.get("ok") is not False and not _unsettled(result))


def _local_result(item):
    if item.get("role") != "system":
        return None
    try:
        value = json.loads(item["content"])["resultado_local"]
        if isinstance(value, dict) and isinstance(value.get("acao"), str):
            return value
    except (KeyError, ValueError, TypeError):
        pass
    return None


def without_completed_proposals(history: list[dict]) -> list[dict]:
    """Mantém os recibos sem apresentar ações concluídas como exemplos de nova saída.

    Só altera a cópia enviada ao provedor. Operações sem par, com erro ou
    resultado incerto permanecem intactas, assim como as mensagens do cliente.
    """
    selected = []
    for i, item in enumerate(history):
        if item.get("role") == "assistant" and not item.get("tool_calls") and i + 1 < len(history):
            try:
                envelope = json.loads(item["content"])
                response = envelope["resposta"]
                action = response["acao"]
                local = _local_result(history[i + 1])
                if (set(envelope) == {"resposta"} and set(response) == {"tipo", "mensagem", "acao"}
                        and response["tipo"] == "acao" and response["mensagem"] is None
                        and isinstance(action, dict) and set(action) == {"nome", "argumentos"}
                        and isinstance(action["nome"], str) and action["nome"] in _KNOWN
                        and local and local["acao"] == action["nome"] and _successful(local.get("resultado"))):
                    continue
            except (KeyError, ValueError, TypeError):
                pass
        selected.append(item)
    return selected


def select_history(agent: str, s, history: list[dict]) -> list[dict]:
    """Filtra recibos concluídos de outros assuntos apenas nos turnos anteriores.

    Preserva as mensagens naturais e o turno atual inteiro. Chamadas nativas
    e seus resultados são inseparáveis; grupos incompletos ou com erro ficam.
    """
    current = max((i for i, m in enumerate(history) if m.get("role") == "user"), default=-1)
    selected = []
    i = 0
    while i < len(history):
        item = history[i]
        if i >= current:
            selected.extend(history[i:])
            break
        calls = item.get("tool_calls") if item.get("role") == "assistant" else None
        if calls:
            group = history[i:i + 1 + len(calls)]
            names = [c.get("function", {}).get("name") for c in calls]
            paired = len(group) == 1 + len(calls) and all(
                r.get("role") == "tool" and r.get("tool_call_id") == c.get("id") for c, r in zip(calls, group[1:]))
            try:
                success = paired and all(_successful(json.loads(r["content"])) for r in group[1:])
            except (KeyError, ValueError, TypeError):
                success = False
            if not (success and all(isinstance(n, str) and n in _KNOWN and n not in _RELEVANT[agent] for n in names)
                    and i + len(group) <= current):
                selected.extend(group)
            i += len(group)
            continue
        # A proposta estruturada e seu resultado local também são mantidos juntos.
        if item.get("role") == "assistant" and i + 1 < current:
            try:
                name = json.loads(item["content"])["resposta"]["acao"]["nome"]
            except (KeyError, ValueError, TypeError):
                name = None
            local = _local_result(history[i + 1])
            if isinstance(name, str) and local:
                if not (name in _KNOWN and name not in _RELEVANT[agent]
                        and local["acao"] == name and _successful(local.get("resultado"))):
                    selected.extend(history[i:i + 2])
                i += 2
                continue
        local = _local_result(item)
        if not (local and local["acao"] in _KNOWN and local["acao"] not in _RELEVANT[agent]
                and _successful(local.get("resultado"))):
            selected.append(item)
        i += 1
    question = s.agent_questions.get(agent)
    if (question and question["stage"] == _question_stage(agent, s)
            and not any(m.get("role") == "assistant" and m.get("content") == question["text"] for m in selected)):
        selected.insert(0, {"role": "assistant", "content": question["text"]})
    return selected
