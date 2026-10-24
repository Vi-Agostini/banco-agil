"""Execução validada de ferramentas dentro do escopo do agente ativo."""
from __future__ import annotations

import inspect
import json
import logging

from .credit_service import CreditService
from .exchange import ExchangeService
from .flow import FlowController
from .tool_contracts import ToolContractError, parse_arguments, precondition
from .repository import BankRepository, DataError
from .session import Session
from .tools import ToolContext

log = logging.getLogger("banco_agil")


class ToolExecutor:
    """Valida escopo e argumentos antes de executar a ferramenta."""

    def __init__(
        self, repo: BankRepository, fx: ExchangeService, credit: CreditService,
        flow: FlowController | None = None,
    ) -> None:
        self.repo = repo
        self.fx = fx
        self.credit = credit
        self.flow = flow or FlowController()

    def execute_from_llm(self, agent, session: Session, name: str, raw_args: str) -> dict:
        """Rejeita credenciais de autenticação fornecidas pelo modelo."""
        if name == "autenticar_cliente":
            # Apenas o coletor local recebe as credenciais originais. A IA vê
            # o histórico mascarado e não pode fornecer credenciais confiáveis.
            return {"erro": "Autenticação disponível somente pela coleta local de credenciais.",
                    "codigo": "autenticacao_local_obrigatoria"}
        try:
            tool = agent.get_tool(name)
            if tool is None:
                raise ToolContractError("ferramenta_fora_escopo")
            args = parse_arguments(tool, raw_args)
            precondition(agent, session, name, args)
        except ToolContractError as error:
            return {"erro": "Chamada não autorizada ou dados incompletos. Esclareça a solicitação.", "codigo": error.code}
        return self.execute(agent, session, name, raw_args)

    def execute(self, agent, session: Session, name: str, raw_args: str) -> dict:
        if agent.name != session.active_agent:
            return {"erro": "O escopo do atendimento mudou.", "codigo": "ferramenta_fora_escopo"}
        tool = agent.get_tool(name)
        if tool is None:
            return self._apply_effects(session, name, {
                "erro": f"Ferramenta '{name}' fora do escopo deste atendimento.",
                "codigo": "ferramenta_fora_escopo",
            })
        context = ToolContext(session, self.repo, self.fx, self.credit)
        try:
            args = json.loads(raw_args)
            if not isinstance(args, dict):
                raise ValueError("argumentos não são um objeto")
            inspect.signature(tool.handler).bind(context, **args)
        except (ValueError, TypeError):
            log.warning("Argumentos inválidos para a tool %s", name)
            return self._apply_effects(session, name, {
                "erro": "Argumentos inválidos. Colete os dados necessários e tente de novo.",
                "codigo": "argumentos_invalidos",
            })
        try:
            result = tool.handler(context, **args)
        except DataError as error:
            log.error("Erro de dados na tool %s: %s", name, error)
            result = {
                "erro": "Sistema de dados indisponível no momento.",
                "codigo": "dados_indisponiveis",
                "instrucao": "Peça desculpas, explique que houve um problema técnico e sugira tentar novamente.",
            }
        except Exception as error:
            log.error("Erro inesperado na tool %s (%s)", name, type(error).__name__, exc_info=True)
            result = {
                "erro": "Falha técnica inesperada. Peça desculpas e ofereça tentar novamente.",
                "codigo": "erro_tecnico",
            }
        return self._apply_effects(session, name, result)

    def _apply_effects(self, session: Session, name: str, result: dict) -> dict:
        if "erro" in result:
            result.setdefault("codigo", "erro_tecnico")
        session.tool_facts[name] = {"result": result, "turn": session.turn}
        self.flow.apply_tool_effects(session, name, result)
        return result
