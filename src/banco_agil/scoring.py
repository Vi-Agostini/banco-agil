"""Validação financeira e fórmula de score adotada pelo projeto.

    score = (renda / (despesas + 1)) * 30 + peso_emprego + peso_dependentes + peso_dividas

O resultado é arredondado e limitado entre 0 e 1000. O termo renda/despesa
é sensível à proporção dos valores e pode saturar o score quando as despesas
são baixas; não leva em conta a renda absoluta. O acréscimo de 1 ao denominador
evita divisão por zero. A fórmula e seus pesos permanecem os mesmos.
"""
from __future__ import annotations

import math
import re
from typing import Any

from .intents import normalize_text

PESO_RENDA = 30
PESO_EMPREGO = {"formal": 300, "autonomo": 200, "desempregado": 0}
PESO_DEPENDENTES = {0: 100, 1: 80, 2: 60, 3: 30}  # 3 = "3 ou mais"
PESO_DIVIDAS = {True: -100, False: 100}

MAX_MONEY = 1e12
MAX_DEPENDENTS = 50

_EMPLOYMENT_ALIASES = {
    "formal": "formal",
    "clt": "formal",
    "carteira assinada": "formal",
    "autonomo": "autonomo",
    "freelancer": "autonomo",
    "desempregado": "desempregado",
}
_TRUE = {"sim", "s", "yes", "y", "true", "verdadeiro"}
_FALSE = {"nao", "n", "no", "false", "falso"}
_BR_THOUSANDS = re.compile(r"^-?\d{1,3}(\.\d{3})+(,\d+)?$")
_COMMA_DECIMAL = re.compile(r"^-?\d+(,\d+)?$")
_PLAIN = re.compile(r"^-?\d+(\.\d+)?$")


class ValidationError(ValueError):
    """Falha de validação com código estável para tratamento pela aplicação."""

    def __init__(self, message: str, code: str) -> None:
        super().__init__(message)
        self.code = code


def _strip(text: str) -> str:
    return normalize_text(text)


def normalize_employment(value: Any) -> str:
    key = _EMPLOYMENT_ALIASES.get(_strip(value)) if isinstance(value, str) else None
    if not key:
        raise ValidationError("Tipo de emprego deve ser: formal, autônomo ou desempregado.", "emprego_invalido")
    return key


def parse_bool(value: Any, field: str = "valor") -> bool:
    """Aceita bool real ou 'sim/não/true/false'. Qualquer outro valor é erro."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        key = _strip(value)
        if key in _TRUE:
            return True
        if key in _FALSE:
            return False
    code = "dividas_invalidas" if "divid" in _strip(field) else "valor_invalido"
    raise ValidationError(f"Informe 'sim' ou 'não' para {field}.", code)


def parse_money(value: Any, field: str = "valor", positive: bool = False) -> float:
    """Converte para float finito e não negativo (ou > 0 se `positive`)."""
    if isinstance(value, bool) or value is None:
        raise ValidationError(f"Valor inválido para {field}.", "valor_invalido")
    if isinstance(value, str):
        s = value.replace("R$", "").strip()
        if _strip(s) in {"zero", "zero real", "zero reais"}:
            s = "0"
        if _BR_THOUSANDS.match(s):
            s = s.replace(".", "").replace(",", ".")
        elif _COMMA_DECIMAL.match(s):
            s = s.replace(",", ".")
        elif not _PLAIN.match(s):
            raise ValidationError(f"Valor inválido para {field}.", "valor_invalido")
        number = float(s)
    elif isinstance(value, (int, float)):
        number = float(value)
    else:
        raise ValidationError(f"Valor inválido para {field}.", "valor_invalido")
    if not math.isfinite(number):
        raise ValidationError(f"Valor inválido para {field}.", "valor_invalido")
    if number < 0:
        comparison = "maior que" if positive else "maior ou igual a"
        raise ValidationError(f"O valor de {field} deve ser {comparison} zero.", "valor_negativo")
    if positive and number == 0:
        raise ValidationError(f"O valor de {field} deve ser maior que zero.", "valor_nao_positivo")
    if number > MAX_MONEY:
        raise ValidationError(f"Valor de {field} acima do limite aceito.", "valor_acima_do_limite")
    return number


def parse_dependents(value: Any) -> int:
    if isinstance(value, bool) or value is None:
        raise ValidationError("Número de dependentes inválido.", "dependentes_invalidos")
    if isinstance(value, str):
        text = value.strip()
        value = 3 if text == "3+" else (int(text) if text.isdigit() else None)
    if isinstance(value, float) and math.isfinite(value) and value == int(value):
        value = int(value)
    if not isinstance(value, int) or value < 0 or value > MAX_DEPENDENTS:
        raise ValidationError("Número de dependentes inválido.", "dependentes_invalidos")
    return value


def calculate_score(
    renda_mensal: Any,
    despesas_fixas: Any,
    tipo_emprego: Any,
    num_dependentes: Any,
    tem_dividas: Any,
) -> int:
    renda = parse_money(renda_mensal, "renda mensal")
    despesas = parse_money(despesas_fixas, "despesas fixas")
    emprego = normalize_employment(tipo_emprego)
    dependentes = PESO_DEPENDENTES[min(parse_dependents(num_dependentes), 3)]
    dividas = parse_bool(tem_dividas, "dívidas ativas")
    score = (
        (renda / (despesas + 1)) * PESO_RENDA
        + PESO_EMPREGO[emprego]
        + dependentes
        + PESO_DIVIDAS[dividas]
    )
    return max(0, min(1000, round(score)))
