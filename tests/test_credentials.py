import random

import pytest

from banco_agil.intents import extract_credentials
from banco_agil.repository import normalize_cpf, normalize_date
from banco_agil.tools import autenticar_cliente


def _valid_cpf(rng):
    digits = [rng.randrange(10) for _ in range(9)]
    for weight in (10, 11):
        total = sum(digit * factor for digit, factor in zip(digits, range(weight, 1, -1)))
        remainder = total % 11
        digits.append(0 if remainder < 2 else 11 - remainder)
    return "".join(map(str, digits))


@pytest.mark.parametrize("value", ["11144477735", "111.444.777-35"])
def test_extracts_same_cpf_from_plain_or_formatted_input(value):
    assert extract_credentials(value).cpf == "11144477735"


@pytest.mark.parametrize("text", [
    "meu cpf é 111.444.777-35 e nasci em 12/05/1990",
    "11144477735 12/05/1990",
    "11144477735 (12/05/1990)",
])
def test_extracts_cpf_and_birth_date_without_cross_contamination(text):
    assert extract_credentials(text) == extract_credentials(
        "meu cpf é 111.444.777-35 e nasci em 12/05/1990"
    )
    credentials = extract_credentials(text)
    assert credentials.cpf == "11144477735"
    assert credentials.birth_date == "1990-05-12"


def test_compact_birth_date_requires_waiting_for_birth_date():
    assert extract_credentials("12051990").birth_date is None
    assert extract_credentials("12051990", awaiting_birth_date=True).birth_date == "1990-05-12"


@pytest.mark.parametrize("value", ["31/02/1990", "12/05/1850", "abc", ""])
def test_invalid_or_implausible_dates_are_rejected(value):
    assert normalize_date(value) is None


def test_normalizers_only_accept_whole_values():
    assert normalize_cpf("Meu CPF é 111.444.777-35") is None
    assert normalize_date("CPF 11144477735") is None
    assert normalize_cpf("111 444 777 35") == "11144477735"
    assert normalize_date("(12/05/1990)") is None


def test_twenty_thousand_cpfs_never_parse_as_dates():
    rng = random.Random(314159)
    for _ in range(20_000):
        cpf = _valid_cpf(rng)
        formatted = f"{cpf[:3]}.{cpf[3:6]}.{cpf[6:9]}-{cpf[9:]}"
        assert normalize_date(cpf) is None
        assert normalize_date(formatted) is None
        for candidate in (cpf, formatted):
            credentials = extract_credentials(candidate)
            assert credentials.cpf == cpf
            assert credentials.birth_date is None


@pytest.mark.parametrize("birth_date", ["12/05/1990", "1990-05-12"])
def test_auth_accepts_unformatted_cpf_and_supported_dates(ctx, birth_date):
    result = autenticar_cliente(ctx, "11144477735", birth_date)
    assert result["sucesso"] and ctx.session.cpf == "11144477735"