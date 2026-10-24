"""Classificação determinística das respostas que podem ser interceptadas localmente."""
from __future__ import annotations

import json
import re
from collections.abc import Callable

from .agents import AGENTS
from .interview import confirm_interview_proposal
from .flow import Awaiting, FlowController, TECHNICAL_FAILURE_CODES
from .intents import (
    close_scope_reply,
    extract_credentials,
    has_birth_date_candidate,
    is_concise_affirmative,
    is_concise_negative,
    is_explicit_interview_confirmation,
    is_interview_data_confirmation,
    is_close_request,
    is_isolated_greeting,
    is_global_close_request,
    is_negative,
    normalize_text,
    parse_limit_amount,
    wants_limit_increase_without_amount,
    wants_to_end_interview_only,
)
from .messages import (
    INTERVIEW_CLOSE_AMBIGUOUS,
    INTERVIEW_CLOSE_CANCELLED,
    INTERVIEW_DECLINED,
    LIMIT_AMOUNT_PROMPT,
    TRIAGE_AUTHENTICATED,
    TRIAGE_BIRTH_DATE,
    TRIAGE_CPF,
    auth_restart_message,
    limit_request_reply,
)
from .privacy import has_birth_marker, has_unrelated_date_context, redact_history
from .session import Session
from . import structured_output
from .reply_recorder import ReplyRecorder
from .tool_executor import ToolExecutor
from .tools import MSG_FAREWELL


def interview_offer_reply(text: str) -> bool | None:
    """Retorna sim/não somente quando a resposta à oferta é inequívoca."""
    if is_concise_affirmative(text) or normalize_text(text) in {
        "sim quero entrevista", "sim quero a entrevista", "quero entrevista", "quero a entrevista",
    }:
        return True
    if is_negative(text):
        return False
    return None


def limit_amount_reply(text: str) -> str | None:
    """Retorna a entrada se for claramente numérica, caso contrário deixa o LLM responder."""
    return parse_limit_amount(text)


class ConversationInterceptors:
    """Executa os tratamentos locais na ordem de prioridade definida."""

    def __init__(self, tools: ToolExecutor, replies: ReplyRecorder, flow: FlowController) -> None:
        self.tools = tools
        self.replies = replies
        self.flow = flow
        self._handlers: tuple[Callable[[Session, str], str | None], ...] = (
            self._close_confirmation,
            self._close_request,
            self._authentication_greeting,
            self._authentication_credentials,
            self._interview_offer,
            self._interview_confirmation,
            self._credit_confirmation,
            self._limit_amount,
            self._limit_increase_without_amount,
            self._authenticated_credit_topic,
        )

    def try_handle(self, session: Session, text: str) -> str | None:
        for handler in self._handlers:
            reply = handler(session, text)
            if reply is not None:
                return reply
        return None

    def _close_confirmation(self, session: Session, text: str) -> str | None:
        """Resolve a confirmação pendente sem encerrar além do escopo autorizado.

        Respostas não relacionadas abandonam a confirmação e seguem para
        o fluxo normal, sem encerrar o atendimento.
        """
        if session.flow.awaiting is not Awaiting.CLOSE_CONFIRM:
            return None
        scope = close_scope_reply(text)
        if scope == "global":
            self.flow.clear_requested_input(session)
            return self._end_session(session)
        if scope == "interview":
            self.flow.clear_requested_input(session)
            return self._transfer_to_credit(session)
        if scope == "no":
            self.flow.clear_requested_input(session)
            return self.replies.say(session, INTERVIEW_CLOSE_CANCELLED)
        if scope == "clarify":
            session.flow.attempts += 1
            if session.flow.attempts >= 3:
                # Muitas repetições: cancela a confirmação e mantém a entrevista.
                self.flow.clear_requested_input(session)
                return self.replies.say(session, INTERVIEW_CLOSE_CANCELLED)
            return self.replies.say(session, INTERVIEW_CLOSE_AMBIGUOUS)
        # Resposta não relacionada: abandona a confirmação sem fechar nada e
        # deixa os demais tratamentos e o pipeline seguirem com a mensagem.
        self.flow.clear_requested_input(session)
        return None

    def _close_request(self, session: Session, text: str) -> str | None:
        in_interview = session.active_agent == "entrevista"
        # 1) Cancelamento explícito da entrevista -> volta para o crédito, sem encerrar tudo.
        if in_interview and wants_to_end_interview_only(text):
            return self._transfer_to_credit(session)
        # 2) Pedido de encerramento (global) reconhecido.
        if not is_close_request(text):
            return None
        # 3) Em entrevista, pedido ambíguo (sem alvo global claro) -> grava a
        #    confirmação pendente na sessão e pergunta antes de encerrar.
        if in_interview and not is_global_close_request(text):
            self.flow.clear_requested_input(session)
            session.flow.awaiting = Awaiting.CLOSE_CONFIRM
            return self.replies.say(session, INTERVIEW_CLOSE_AMBIGUOUS)
        # 4) Encerramento global explícito (ou fora da entrevista) -> encerra o atendimento.
        return self._end_session(session)

    def _transfer_to_credit(self, session: Session) -> str:
        """Cancela somente a entrevista e devolve o cliente ao agente de crédito."""
        session.flow.unclear_count = 0
        call_id = f"transfer-{len(session.history)}"
        if session.llm_mode != "structured":
            session.history.append({"role": "assistant", "content": None, "tool_calls": [{
                "id": call_id, "type": "function",
                "function": {"name": "transferir_para_agente", "arguments": '{"destino": "credito"}'},
            }]})
        result = self.tools.execute(
            AGENTS[session.active_agent], session, "transferir_para_agente",
            json.dumps({"destino": "credito"}, ensure_ascii=False),
        )
        self.flow.finish_interview(session)
        if session.llm_mode == "structured":
            structured_output.record_result(session, "transferir_para_agente", result)
        else:
            session.history.append({
                "role": "tool", "tool_call_id": call_id,
                "content": json.dumps(result, ensure_ascii=False),
            })
        return self.replies.say(session, INTERVIEW_DECLINED)

    def _end_session(self, session: Session) -> str:
        session.flow.unclear_count = 0
        call_id = f"close-{len(session.history)}"
        if session.llm_mode != "structured":
            session.history.append({"role": "assistant", "content": None, "tool_calls": [{
                "id": call_id, "type": "function",
                "function": {"name": "encerrar_atendimento", "arguments": "{}"},
            }]})
        result = self.tools.execute(AGENTS[session.active_agent], session, "encerrar_atendimento", "{}")
        if session.llm_mode == "structured":
            structured_output.record_result(session, "encerrar_atendimento", result)
        else:
            session.history.append({
                "role": "tool", "tool_call_id": call_id,
                "content": json.dumps(result, ensure_ascii=False),
            })
        return self.replies.say(session, result.get("mensagem_final", MSG_FAREWELL))

    def _authentication_greeting(self, session: Session, text: str) -> str | None:
        if session.active_agent != "triagem" or session.authenticated or not is_isolated_greeting(text):
            return None
        prompt = TRIAGE_BIRTH_DATE if session.flow.awaiting is Awaiting.BIRTH_DATE else TRIAGE_CPF
        # Uma saudação válida não conta como falha de autenticação ou entrada inválida.
        # Mantém as credenciais pendentes e os contadores de segurança.
        return self.replies.say(session, f"Olá! {prompt}", track=False)

    def _authentication_credentials(self, session: Session, text: str) -> str | None:
        if session.active_agent != "triagem" or session.authenticated:
            return None
        awaiting_birth_date = session.flow.awaiting is Awaiting.BIRTH_DATE
        credentials = extract_credentials(text, awaiting_birth_date=awaiting_birth_date)
        if awaiting_birth_date:
            cpf = credentials.cpf or session.flow.pending_cpf
            if credentials.birth_date is None:
                if cpf and has_birth_date_candidate(text):
                    result = self.tools.execute(
                        AGENTS["triagem"], session, "autenticar_cliente",
                        json.dumps({"cpf": cpf, "data_nascimento": text}, ensure_ascii=False),
                    )
                    redact_history(session.history)
                    return self._authentication_reply(session, result)
                if credentials.cpf:
                    session.flow.pending_cpf = credentials.cpf
                return None
            if cpf is None:
                session.flow.awaiting = Awaiting.NONE
                return None
            args = {"cpf": cpf, "data_nascimento": credentials.birth_date}
            result = self.tools.execute(
                AGENTS["triagem"], session, "autenticar_cliente",
                json.dumps(args, ensure_ascii=False),
            )
            redact_history(session.history)
            if result.get("sucesso"):
                return self.replies.say(session, TRIAGE_AUTHENTICATED.format(name=session.nome))
            return self._authentication_reply(session, result)
        # As credenciais completas são verificadas localmente, sem pedir CPF ou data à IA.
        # O histórico mascarado preserva o assunto para o pipeline continuar o atendimento.
        if credentials.cpf and credentials.birth_date and (
            has_birth_marker(text) or not has_unrelated_date_context(text)
        ):
            result = self.tools.execute(
                AGENTS["triagem"], session, "autenticar_cliente",
                json.dumps(
                    {"cpf": credentials.cpf, "data_nascimento": credentials.birth_date},
                    ensure_ascii=False,
                ),
            )
            redact_history(session.history)
            if result.get("sucesso"):
                return None
            return self._authentication_reply(session, result)
        if credentials.cpf and has_birth_date_candidate(text) and (
            has_birth_marker(text) or not has_unrelated_date_context(text)
        ):
            result = self.tools.execute(
                AGENTS["triagem"], session, "autenticar_cliente",
                json.dumps({"cpf": credentials.cpf, "data_nascimento": text}, ensure_ascii=False),
            )
            redact_history(session.history)
            return self._authentication_reply(session, result)
        if (
            credentials.cpf
            and credentials.birth_date is None
            and not has_birth_date_candidate(text)
        ):
            session.flow.pending_cpf = credentials.cpf
            session.flow.awaiting = Awaiting.BIRTH_DATE
            return self.replies.say(session, TRIAGE_BIRTH_DATE)
        return None

    def _interview_offer(self, session: Session, text: str) -> str | None:
        if session.flow.awaiting is not Awaiting.INTERVIEW_OFFER:
            return None
        if not (session.authenticated and session.cpf):
            return None
        answer = interview_offer_reply(text)
        if answer is None:
            return None
        self.flow.clear_requested_input(session)
        session.flow.unclear_count = 0
        if answer:
            session.active_agent = "entrevista"
            self.flow.start_interview(session)
            session.flow.interview_question_pending = True
            return None  # A IA formula a pergunta; apenas os schemas dependem do estado.
        session.active_agent = "triagem"
        return self.replies.say(session, INTERVIEW_DECLINED)

    def _interview_confirmation(self, session: Session, text: str) -> str | None:
        """Confirma localmente uma proposta resumida pelo agente de entrevista.

        A mensagem que originou a proposta já foi tratada pelo LLM. Apenas uma
        resposta curta e inequívoca confirma ou descarta esses valores; textos
        diferentes seguem para o agente, que pode pedir esclarecimentos ou
        registrar uma nova proposta.
        """
        if session.active_agent != "entrevista":
            return None
        if not session.flow.pending_interview_answers:
            if session.flow.interview_answers and re.search(
                    r"\b(?:corrig\w*|correcao|na verdade|me enganei|atualiz\w*)\b", normalize_text(text)):
                session.flow.interview_revision_requested = True
            return None
        if is_interview_data_confirmation(text):
            structured_output.record_result(session, "confirmacao_entrevista", confirm_interview_proposal(session))
        elif is_concise_negative(text):
            session.flow.pending_interview_answers.clear()
            session.flow.interview_revision_requested = False
            session.flow.input_kind = "interview"
            session.flow.input_prompt = None
            session.flow.attempts = 0
        else:
            # Há texto além de uma confirmação curta. Ele pode corrigir apenas
            # parte da proposta; mantenha os demais valores pendentes e deixe
            # o agente interpretar a mensagem antes de pedir novo resumo.
            session.flow.interview_revision_requested = True
            session.flow.input_kind = "interview"
            session.flow.input_prompt = None
            session.flow.attempts = 0
        return None

    def _limit_amount(self, session: Session, text: str) -> str | None:
        amount = limit_amount_reply(text)
        if session.active_agent != "credito" or session.flow.awaiting is not Awaiting.LIMIT_AMOUNT or amount is None:
            return None
        if session.credit_proposal and not session.credit_proposal.get("executada"):
            return None  # Preserve a distinção entre total, acréscimo e ambiguidade.
        session.flow.unclear_count = 0
        result = self.tools.execute(
            AGENTS["credito"], session, "solicitar_aumento_limite",
            json.dumps({"novo_limite": amount}, ensure_ascii=False),
        )
        if "erro" in result:
            reply = self.flow.requested_input_error(session, "limit", result)
        else:
            reply = limit_request_reply(result)
        return self.replies.say(session, reply, track=result.get("codigo") not in TECHNICAL_FAILURE_CODES)

    def _credit_confirmation(self, session: Session, text: str) -> str | None:
        proposal = session.credit_proposal
        if (session.active_agent != "credito" or not proposal or proposal.get("executada")
                or session.flow.awaiting is Awaiting.INTERVIEW_OFFER
                or session.flow.pending_interview_answers):
            return None
        if is_explicit_interview_confirmation(text):
            if proposal["modalidade"] == "indefinida":
                # Um aceite não esclarece se o valor é limite total ou acréscimo.
                # A IA deve fazer essa pergunta sem reiniciar o encaminhamento.
                session.credit_clarification_turn = session.turn
            else:
                proposal["confirmada"] = True
                session.credit_confirmation_turn = session.turn
        elif is_concise_negative(text):
            session.credit_proposal = None
        return None  # A IA formula a próxima resposta, usando o estado confirmado.

    def _authentication_reply(self, session: Session, result: dict) -> str:
        message = result.get("mensagem_final") or auth_restart_message(result)
        return self.replies.say(session, message)

    def _limit_increase_without_amount(self, session: Session, text: str) -> str | None:
        if (
            session.active_agent != "credito"
            or session.flow.awaiting in (Awaiting.INTERVIEW_OFFER, Awaiting.LIMIT_AMOUNT)
            or not wants_limit_increase_without_amount(text)
        ):
            return None
        session.flow.unclear_count = 0
        session.flow.awaiting = Awaiting.LIMIT_AMOUNT
        session.flow.input_kind = "limit"
        session.flow.input_prompt = LIMIT_AMOUNT_PROMPT
        session.flow.attempts = 0
        return None  # Inclusive valores por extenso seguem para interpretação da IA.

    def _authenticated_credit_topic(self, session: Session, text: str) -> None:
        """Encaminha apenas um assunto explícito; a IA continua a conversa.

        Não interpreta valores, cria propostas ou autoriza operações. Mensagens
        ambíguas/mistas permanecem com o agente para esclarecimento natural.
        """
        if (session.active_agent != "triagem" or not session.authenticated or not session.cpf
                or session.ended or session.flow.awaiting is not Awaiting.NONE):
            return None
        topic = normalize_text(text)
        if re.search(r"\b(?:nao|nem|cambio|cotacao|dolar|euro|libra|entrevista)\b", topic):
            return None
        if not re.search(r"\b(?:meu limite|limite de credito|limite do cartao|"
                         r"(?:consultar|aumentar) (?:o )?(?:meu )?limite)\b", topic):
            return None
        result = self.tools.execute(AGENTS["triagem"], session, "transferir_para_agente",
                                    json.dumps({"destino": "credito"}))
        # Resultado local real, sem fabricar uma proposta do modelo ou pares
        # de chamadas nativas. Os dois modos aceitam contexto de sistema.
        structured_output.record_result(session, "transferir_para_agente", result)
        return None
