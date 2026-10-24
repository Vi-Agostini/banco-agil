"""Cotação de moedas em tempo real (APIs públicas, sem chave)."""
from __future__ import annotations

import logging
import math
import re
from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

import requests

log = logging.getLogger("banco_agil")
TIMEOUT = 8
SAO_PAULO = ZoneInfo("America/Sao_Paulo")


class ExchangeError(Exception):
    """Erro com mensagem segura para o cliente (detalhes técnicos vão só para o log)."""

    def __init__(self, message: str, *, code: str = "cotacao_indisponivel"):
        self.code = code
        super().__init__(message)


def _valid_rate(value) -> float:
    rate = float(value)
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError("cotação inválida")
    rounded = round(rate, 4)
    if rounded <= 0:
        raise ValueError("cotação inválida após arredondamento")
    return rounded


def _awesomeapi_updated_at(data: dict) -> str | None:
    timestamp = data.get("timestamp")
    if timestamp is not None:
        try:
            epoch = float(timestamp)
            if epoch > 1_000_000_000_000:
                epoch /= 1000
            updated = datetime.fromtimestamp(epoch, timezone.utc).astimezone(SAO_PAULO)
            return updated.strftime("%d/%m/%Y às %H:%M:%S (horário de São Paulo)")
        except (OverflowError, OSError, TypeError, ValueError):
            pass

    create_date = data.get("create_date")
    if create_date:
        try:
            updated = datetime.strptime(create_date, "%Y-%m-%d %H:%M:%S").replace(tzinfo=SAO_PAULO)
            return updated.strftime("%d/%m/%Y às %H:%M:%S (horário de São Paulo)")
        except (TypeError, ValueError):
            pass
    return None


class ExchangeService:
    def __init__(self, http_get=None):
        self._get = http_get or (lambda *a, **k: requests.get(*a, **k))

    def get_rate(self, currency: str) -> dict:
        code = (currency or "").strip().upper() if isinstance(currency, str) else ""
        if not re.fullmatch(r"[A-Z]{3}", code) or code == "BRL":
            raise ExchangeError(
                "Qual moeda você deseja consultar? Pode informar o nome ou o país.",
                code="moeda_nao_identificada",
            )
        for name, fetch in (("AwesomeAPI", self._awesomeapi), ("Frankfurter", self._frankfurter)):
            try:
                return fetch(code)
            except (requests.RequestException, KeyError, ValueError, TypeError, AttributeError) as e:
                log.warning("Cotação %s indisponível em %s (%s)", code, name, type(e).__name__)
        raise ExchangeError(
            "Não consegui obter a cotação dessa moeda nas fontes disponíveis neste momento."
        )

    def _awesomeapi(self, code: str) -> dict:
        r = self._get(f"https://economia.awesomeapi.com.br/json/last/{code}-BRL", timeout=TIMEOUT)
        r.raise_for_status()
        data = r.json()[f"{code}BRL"]
        return {
            "moeda": code,
            "valor_em_brl": _valid_rate(data["bid"]),
            "atualizado_em": _awesomeapi_updated_at(data),
            "fonte": "AwesomeAPI",
        }

    def _frankfurter(self, code: str) -> dict:
        r = self._get(
            "https://api.frankfurter.app/latest", params={"from": code, "to": "BRL"}, timeout=TIMEOUT
        )
        r.raise_for_status()
        data = r.json()
        updated = data.get("date")
        if not isinstance(updated, str) or not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", updated):
            raise ValueError("data BCE inválida")
        updated = date.fromisoformat(updated).isoformat()
        return {
            "moeda": code,
            "valor_em_brl": _valid_rate(data["rates"]["BRL"]),
            "atualizado_em": updated,
            "fonte": "Frankfurter (BCE)",
        }
