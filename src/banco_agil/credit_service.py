"""Regras de crédito e retomada das gravações entre os CSVs.

Não há transação entre arquivos. Primeiro registra o pedido pendente, depois
grava a decisão e, se aprovado, aplica o limite ao cliente. A retomada usa o
recibo da operação ou a reconciliação para evitar repetir seus efeitos; criar
um pedido sem recibo não é uma operação idempotente. Rejeições não alteram o limite.
"""
from __future__ import annotations

import logging
import math

from .privacy import mask_cpf
from .repository import BankRepository, DataError
from .scoring import ValidationError
from .statuses import STATUS_APROVADO, STATUS_PENDENTE, STATUS_REPROVADO

log = logging.getLogger("banco_agil")


class CreditError(ValidationError):
    """Pedido inválido (mensagem segura para o cliente)."""

    def __init__(self, message: str, code: str = "credito_invalido") -> None:
        super().__init__(message, code)


class CreditService:
    def __init__(self, repo: BankRepository):
        self.repo = repo

    def decide(self, score: int, requested: float) -> str:
        return STATUS_APROVADO if requested <= self.repo.max_limit_for_score(score) else STATUS_REPROVADO

    def request_increase(self, cpf: str, requested: float, *, receipt: dict | None = None,
                         resume_only: bool = False) -> dict:
        with self.repo.lock:
            self.reconcile(cpf)
            client = self.repo.get_client(cpf)
            if client is None:
                raise CreditError("Cliente não encontrado.", "cliente_nao_encontrado")
            current = client["limite_credito"]
            if receipt and "request_id" in receipt:
                rows = self.repo.list_requests()
                index = receipt["request_id"]
                if (receipt.get("cpf") != cpf or receipt.get("requested") != requested
                        or type(index) is not int or index < 0 or index > len(rows)):
                    raise CreditError("Não foi possível retomar o pedido com segurança.", "pedido_estado_incerto")
                if index < len(rows):
                    row = rows[index]
                    if (row["cpf_cliente"] != cpf or float(row["novo_limite_solicitado"]) != requested
                            or float(row["limite_atual"]) != receipt["current"]
                            or row["data_hora_solicitacao"] != receipt.get("timestamp")):
                        raise CreditError("Não foi possível retomar o pedido com segurança.", "pedido_estado_incerto")
                    return {
                        "id": index, "status_pedido": row["status_pedido"],
                        "novo_limite_solicitado": requested, "limite_credito_atual": current,
                        "limite_aplicado": row["status_pedido"] == STATUS_APROVADO and math.isclose(current, requested, abs_tol=.005),
                    }
                # Sem linha gravada, só tenta de novo se a posição continuar livre
                # e os dados usados como referência não tiverem mudado.
                if current != receipt["current"]:
                    raise CreditError("Não foi possível retomar o pedido com segurança.", "pedido_estado_incerto")
            if resume_only:
                raise CreditError("Não foi possível confirmar a gravação do pedido.", "pedido_estado_incerto")
            if requested <= current:
                raise CreditError(
                    "O novo limite precisa ser maior que o limite atual.",
                    "limite_menor_ou_igual_ao_atual",
                )
            # Valida a tabela de score antes de gravar; uma falha não altera o pedido ou o limite.
            status = self.decide(client["score"], requested)
            request = self.repo.create_request(cpf, current, requested, receipt=receipt)
            self.repo.set_request_status(request["id"], cpf, status)
            applied = False
            if status == STATUS_APROVADO:
                try:
                    self.repo.update_limit(cpf, requested)
                    applied = True
                except DataError:
                    log.error("Limite não aplicado (solicitação %s, %s); reconciliação pendente",
                              request["id"], mask_cpf(cpf))
            return {
                "id": request["id"],
                "status_pedido": status,
                "novo_limite_solicitado": requested,
                "limite_credito_atual": requested if status == STATUS_APROVADO and applied else current,
                "limite_aplicado": applied,
            }

    def reanalyze_rejected_request(self, cpf: str, source_request_id: int, requested: float,
                                  *, receipt: dict | None = None) -> dict:
        """Reavalia uma reprovação identificada da sessão com as regras usuais."""
        with self.repo.lock:
            requests = self.repo.list_requests()
            if not isinstance(source_request_id, int) or not 0 <= source_request_id < len(requests):
                raise CreditError("Não foi possível localizar o pedido anterior.", "reanalise_origem_invalida")
            source = requests[source_request_id]
            try:
                source_requested = float(source["novo_limite_solicitado"])
            except (KeyError, TypeError, ValueError) as error:
                raise CreditError("Não foi possível validar o pedido anterior.", "reanalise_origem_invalida") from error
            if (
                source["cpf_cliente"] != cpf
                or source["status_pedido"] != STATUS_REPROVADO
                or not math.isclose(source_requested, requested, abs_tol=0.005)
            ):
                raise CreditError("O pedido anterior não está disponível para reanálise.", "reanalise_origem_invalida")
            return self.request_increase(cpf, requested, receipt=receipt)

    def reconcile(self, cpf: str | None = None) -> int:
        """Completa operações interrompidas. Retorna quantas correções fez."""
        fixed = 0
        with self.repo.lock:
            try:
                for req in self.repo.list_requests():
                    if req["status_pedido"] != STATUS_PENDENTE or (cpf and req["cpf_cliente"] != cpf):
                        continue
                    client = self.repo.get_client(req["cpf_cliente"])
                    requested = float(req["novo_limite_solicitado"])
                    if client is None or requested <= client["limite_credito"]:
                        status = STATUS_REPROVADO
                    else:
                        status = self.decide(client["score"], requested)
                    self.repo.set_request_status(req["id"], req["cpf_cliente"], status)
                    log.warning("Solicitação %s pendente resolvida como %s", req["id"], status)
                    fixed += 1
                latest: dict[str, dict] = {}
                for req in self.repo.list_requests():
                    if req["status_pedido"] == STATUS_APROVADO:
                        latest[req["cpf_cliente"]] = req
                for owner, req in latest.items():
                    if cpf and owner != cpf:
                        continue
                    client = self.repo.get_client(owner)
                    requested = float(req["novo_limite_solicitado"])
                    # só aplica se o limite ainda é o de antes do pedido (não sobrescreve ajustes externos)
                    if (
                        client
                        and math.isclose(client["limite_credito"], float(req["limite_atual"]), abs_tol=0.005)
                        and requested > client["limite_credito"]
                    ):
                        self.repo.update_limit(owner, requested)
                        log.warning("Aprovação %s aplicada na reconciliação (%s)", req["id"], mask_cpf(owner))
                        fixed += 1
            except (KeyError, ValueError, TypeError) as e:
                raise DataError("Arquivo de solicitações inválido.") from e
        return fixed
