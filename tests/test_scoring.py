
import pytest

from banco_agil.scoring import (
    ValidationError,
    calculate_score,
    normalize_employment,
    parse_bool,
    parse_dependents,
    parse_money,
)
from banco_agil.messages import requested_input_error


@pytest.mark.parametrize("operation,code", [
    (lambda: parse_money(0, positive=True), "valor_nao_positivo"),
    (lambda: parse_money(-1), "valor_negativo"),
    (lambda: parse_money("inválido"), "valor_invalido"),
    (lambda: parse_money(1e13), "valor_acima_do_limite"),
    (lambda: normalize_employment("astronauta"), "emprego_invalido"),
    (lambda: parse_dependents(-1), "dependentes_invalidos"),
    (lambda: parse_bool("talvez", "dívidas ativas"), "dividas_invalidas"),
])
def test_validation_errors_expose_stable_codes(operation, code):
    with pytest.raises(ValidationError) as caught:
        operation()

    assert caught.value.code == code


@pytest.mark.parametrize("code,detail", [
    ("valor_nao_positivo", "maior que zero"),
    ("valor_negativo", "maior que zero"),
    ("valor_invalido", "interpretar esse número"),
    ("valor_acima_do_limite", "interpretar esse número"),
    ("limite_menor_ou_igual_ao_atual", "maior que o limite atual"),
    ("emprego_invalido", "tipo de emprego válido"),
    ("dependentes_invalidos", "número válido de dependentes"),
    ("dividas_invalidas", "sim ou não para dívidas ativas"),
    ("cliente_nao_encontrado", "Não encontrei o cadastro"),
    ("argumentos_invalidos", "Não consegui interpretar os dados informados"),
    ("erro_tecnico", "problema técnico ao validar os dados"),
])
def test_error_codes_have_customer_facing_details(code, detail):
    assert detail in requested_input_error("interview_income", 1, code, "Corrija a resposta.")


@pytest.mark.parametrize("value,expected", [
    ("sim", True), ("Sim", True), ("não", False), ("NAO", False), ("true", True), ("false", False),
    (True, True), (False, False),
])
def test_parse_bool_accepts(value, expected):
    assert parse_bool(value) is expected


@pytest.mark.parametrize("value", ["talvez", "", None, 1, 0, [], "nao sei"])
def test_parse_bool_rejects(value):
    with pytest.raises(ValueError):
        parse_bool(value)


@pytest.mark.parametrize("value,expected", [
    (1500, 1500.0), (1500.5, 1500.5), ("1500", 1500.0), ("1.500,50", 1500.5), ("1500,5", 1500.5),
    ("R$ 2.000", 2000.0), (0, 0.0), ("R$ 0,00", 0.0), ("zero", 0.0), ("zero reais", 0.0),
])
def test_parse_money_accepts(value, expected):
    assert parse_money(value) == expected


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), -1, "-5", "abc", "nan", "inf",
                                   "1e5", None, True, [], 1e13, ""])
def test_parse_money_rejects(value):
    with pytest.raises(ValueError):
        parse_money(value)


def test_parse_money_positive_rejects_zero():
    for value in (0, "R$ 0,00", "zero"):
        with pytest.raises(ValueError, match="maior que zero"):
            parse_money(value, positive=True)


@pytest.mark.parametrize("value", ["-5", "-1.000,00", "R$ -10,50"])
def test_parse_money_positive_reports_negative_values(value):
    with pytest.raises(ValueError, match="maior que zero"):
        parse_money(value, positive=True)


@pytest.mark.parametrize("value,expected", [(0, 0), (2, 2), ("3+", 3), ("4", 4), (2.0, 2)])
def test_parse_dependents_accepts(value, expected):
    assert parse_dependents(value) == expected


@pytest.mark.parametrize("value", [-1, 1.5, "x", None, True, 999, float("nan")])
def test_parse_dependents_rejects(value):
    with pytest.raises(ValueError):
        parse_dependents(value)


def test_formula_matches_challenge():
    assert calculate_score(5000, 2000, "formal", 0, False) == 575  # 74.96 + 300 + 100 + 100
    assert calculate_score(3000, 1000, "autônomo", 2, True) == round(3000 / 1001 * 30 + 200 + 60 - 100)


def test_debts_yes_vs_no_use_correct_weight():
    com = calculate_score(5000, 2000, "formal", 0, "sim")
    sem = calculate_score(5000, 2000, "formal", 0, "não")
    assert sem - com == 200 and sem == 575


def test_dependents_3_plus_weight():
    assert calculate_score(0, 0, "formal", 3, False) == 300 + 30 + 100
    assert calculate_score(0, 0, "formal", 9, False) == 300 + 30 + 100


def test_zero_expenses_no_division_error_and_clamped():
    assert calculate_score(10000, 0, "formal", 0, False) == 1000


def test_extreme_values_stay_in_range():
    assert calculate_score(1e12, 0, "formal", 0, False) == 1000
    assert calculate_score(0, 1e12, "desempregado", 50, True) == 0  # -70 -> piso 0
    assert calculate_score(0, 0, "desempregado", 3, True) == 0


@pytest.mark.parametrize("kwargs", [
    dict(renda_mensal=-1, despesas_fixas=0, tipo_emprego="formal", num_dependentes=0, tem_dividas="não"),
    dict(renda_mensal=1, despesas_fixas=-1, tipo_emprego="formal", num_dependentes=0, tem_dividas="não"),
    dict(renda_mensal=float("nan"), despesas_fixas=0, tipo_emprego="formal", num_dependentes=0, tem_dividas="não"),
    dict(renda_mensal=1, despesas_fixas=float("inf"), tipo_emprego="formal", num_dependentes=0, tem_dividas="não"),
    dict(renda_mensal=1, despesas_fixas=0, tipo_emprego="astronauta", num_dependentes=0, tem_dividas="não"),
    dict(renda_mensal=1, despesas_fixas=0, tipo_emprego="formal", num_dependentes=-2, tem_dividas="não"),
    dict(renda_mensal=1, despesas_fixas=0, tipo_emprego="formal", num_dependentes=0, tem_dividas="talvez"),
])
def test_calculate_score_rejects_invalid(kwargs):
    with pytest.raises(ValueError):
        calculate_score(**kwargs)
