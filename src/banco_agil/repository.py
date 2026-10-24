"""Acesso aos arquivos CSV (clientes, tabela de score e solicitações)."""
from __future__ import annotations

import csv
import math
import os
import re
import tempfile
import threading
from datetime import datetime
from pathlib import Path

from .statuses import FINAL_STATUSES, STATUS_PENDENTE

CLIENT_FIELDS = ["cpf", "nome", "data_nascimento", "limite_credito", "score"]
REQUEST_FIELDS = [
    "cpf_cliente",
    "data_hora_solicitacao",
    "limite_atual",
    "novo_limite_solicitado",
    "status_pedido",
]
_DATE_FORMATS = ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y")
_LOCK = threading.RLock()


class DataError(Exception):
    """Falha de leitura/escrita nos arquivos de dados."""


def normalize_cpf(value: str) -> str | None:
    if not isinstance(value, str):
        return None
    digits = re.sub(r"[.\s-]", "", value.strip())
    return digits if re.fullmatch(r"\d{11}", digits) else None


def normalize_date(value: str) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    formats = _DATE_FORMATS + (("%d%m%Y",) if re.fullmatch(r"\d{8}", text) else ())
    for fmt in formats:
        try:
            parsed = datetime.strptime(text, fmt).date()
        except ValueError:
            continue
        if 1900 <= parsed.year <= datetime.now().year:
            return parsed.isoformat()
    return None


def _parse_client(row: dict) -> dict:
    try:
        raw_score = float(row["score"])
        if not math.isfinite(raw_score) or not raw_score.is_integer() or not 0 <= raw_score <= 1000:
            raise ValueError("score fora do intervalo")
        return {
            "cpf": normalize_cpf(row["cpf"]),
            "nome": row["nome"],
            "data_nascimento": normalize_date(row["data_nascimento"]) or row["data_nascimento"],
            "limite_credito": float(row["limite_credito"]),
            "score": int(raw_score),
        }
    except (KeyError, ValueError, TypeError) as e:
        raise DataError("Registro de cliente inválido em clientes.csv.") from e


class BankRepository:
    def __init__(self, data_dir: str | Path):
        self.dir = Path(data_dir)
        self.clients_path = self.dir / "clientes.csv"
        self.requests_path = self.dir / "solicitacoes_aumento_limite.csv"
        self.score_path = self.dir / "score_limite.csv"
        self.lock = _LOCK  # reentrante: serviços podem agrupar várias operações

    # Leitura e gravação dos arquivos.
    def _read(self, path: Path, optional: bool = False) -> list[dict]:
        try:
            with open(path, newline="", encoding="utf-8") as f:
                return list(csv.DictReader(f))
        except FileNotFoundError as e:
            if optional:
                return []
            raise DataError(f"Arquivo não encontrado: {path.name}") from e
        except (OSError, csv.Error, UnicodeDecodeError) as e:
            raise DataError(f"Falha ao ler {path.name}: {type(e).__name__}") from e

    def _write(self, path: Path, fields: list[str], rows: list[dict]) -> None:
        tmp = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
            with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            os.replace(tmp, path)
        except OSError as e:
            if tmp and os.path.exists(tmp):
                os.unlink(tmp)
            raise DataError(f"Falha ao gravar {path.name}: {type(e).__name__}") from e

    # Clientes.
    def get_client(self, cpf: str) -> dict | None:
        """Busca por CPF normalizado (o CSV pode armazenar CPF com ou sem pontuação)."""
        if not cpf:
            return None
        with self.lock:
            for row in self._read(self.clients_path):
                if normalize_cpf(row.get("cpf", "")) == cpf:
                    return _parse_client(row)
        return None

    def authenticate(self, cpf: str, birth_iso: str) -> dict | None:
        client = self.get_client(cpf)
        if client and client["data_nascimento"] == birth_iso:
            return client
        return None

    def _update_client(self, cpf: str, **changes) -> None:
        with self.lock:
            rows = self._read(self.clients_path)
            found = False
            for row in rows:
                if normalize_cpf(row.get("cpf", "")) == cpf:
                    row.update({k: str(v) for k, v in changes.items()})
                    found = True
            if not found:
                raise DataError("Cliente não encontrado para atualização.")
            self._write(self.clients_path, CLIENT_FIELDS, rows)

    def update_score(self, cpf: str, score: int) -> None:
        self._update_client(cpf, score=int(score))

    def update_limit(self, cpf: str, limit: float) -> None:
        self._update_client(cpf, limite_credito=f"{limit:.2f}")

    # Faixas de score.
    def max_limit_for_score(self, score: int) -> float:
        if isinstance(score, bool) or not isinstance(score, int) or not 0 <= score <= 1000:
            raise DataError("Score de cliente inválido.")
        try:
            rows = list(self._read(self.score_path))
        except (KeyError, ValueError) as e:
            raise DataError("Tabela score_limite.csv inválida.") from e
        if not rows:
            raise DataError("Tabela score_limite.csv não contém faixas válidas.")
        for row in rows:
            if int(row["score_minimo"]) <= score <= int(row["score_maximo"]):
                return float(row["limite_maximo"])
        raise DataError("Tabela score_limite.csv não contém faixa para o score informado.")

    # Solicitações de aumento.
    # A posição da linha identifica o pedido; novas solicitações mantêm essa ordem.
    # O status pode mudar uma vez, sem alterar o contrato de cinco colunas.
    def list_requests(self) -> list[dict]:
        with self.lock:
            rows = self._read(self.requests_path, optional=True)
        return [{"id": i, **row} for i, row in enumerate(rows)]

    def create_request(self, cpf: str, current: float, requested: float, *, receipt: dict | None = None) -> dict:
        row = {
            "cpf_cliente": cpf,
            "data_hora_solicitacao": datetime.now().isoformat(timespec="microseconds"),
            "limite_atual": f"{current:.2f}",
            "novo_limite_solicitado": f"{requested:.2f}",
            "status_pedido": STATUS_PENDENTE,
        }
        with self.lock:
            rows = self._read(self.requests_path, optional=True)
            if receipt is not None:
                # Prepara a identidade antes da gravação atômica para reconhecer a mesma
                # linha na retomada, mesmo após um timeout na substituição do arquivo.
                receipt.update(request_id=len(rows), cpf=cpf, requested=requested,
                               current=current, timestamp=row["data_hora_solicitacao"])
            rows.append(row)
            self._write(self.requests_path, REQUEST_FIELDS, rows)
            return {"id": len(rows) - 1, **row}

    def set_request_status(self, request_id: int, cpf: str, status: str) -> None:
        """Decide uma solicitação 'pendente'. Decisões já tomadas são imutáveis."""
        if status not in FINAL_STATUSES:
            raise ValueError("Status final inválido.")
        with self.lock:
            rows = self._read(self.requests_path, optional=True)
            if not 0 <= request_id < len(rows) or rows[request_id]["cpf_cliente"] != cpf:
                raise DataError("Solicitação não localizada.")
            if rows[request_id]["status_pedido"] != STATUS_PENDENTE:
                raise DataError("Solicitação já decidida.")
            rows[request_id]["status_pedido"] = status
            self._write(self.requests_path, REQUEST_FIELDS, rows)
