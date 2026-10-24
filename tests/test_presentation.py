import pytest

from banco_agil.presentation import chat_markdown


@pytest.mark.parametrize("message", [
    "Seu limite é R$ 5.000,00.",
    "Limite atual R$ 5.000,00 e solicitado R$ 8.000,00.",
    "**Atual:** R$ 5.000,00; **pedido:** R$ 8.000,00; diferença R$ 3.000,00.",
    "Cotação R$\u202f5,1234; total R$\u00a010,2468.",
])
def test_only_presentation_escapes_currency(message):
    original = message
    rendered = chat_markdown(message)
    assert rendered.replace(r"R\$", "R$") == original
    assert rendered.count(r"R\$") == original.count("R$")
    assert message == original
    assert chat_markdown(rendered) == rendered


@pytest.mark.parametrize("protected", [
    "`R$ 8.000,00`", "``R$ 8.000,00 e `exemplo` ``",
    "```text\nR$ 5.000,00 e R$ 8.000,00\n```",
    "~~~text\nR$ 5.000,00\n~~~",
    "$x^2 + 1$", "$R$ 5", "$$\nx^2 + 1\n$$",
    r"$\text{R$} + x$", r"R\$ 5.000,00",
])
def test_code_math_and_already_escaped_currency_remain_intact(protected):
    assert chat_markdown(protected) == protected
    mixed = "**Valor:** R$ 8.000,00. " + protected
    assert chat_markdown(mixed) == r"**Valor:** R\$ 8.000,00. " + protected
