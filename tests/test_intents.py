import pytest

from banco_agil.intents import (
    close_scope_reply,
    is_affirmative,
    is_explicit_interview_confirmation,
    is_interview_data_confirmation,
    is_close_request,
    is_global_close_request,
    is_negative,
    parse_limit_amount,
    wants_limit_increase_without_amount,
    wants_to_end_interview_only,
)


@pytest.mark.parametrize("text", [
    "encerrar",
    "encerra",
    "quero encerrar",
    "quero encerrar o atendimento",
    "pode encerrar a conversa",
    "sair",
    "quero sair",
    "vou sair agora",
    "tchau",
    "tchau, obrigado",
    "até logo",
    "adeus",
    "finalizar atendimento",
    "por favor, encerre",
    "não quero mais nada",
    "não preciso de mais nada",
    "pode encerrar por favor",
    "ENCERRAR!!!",
    "por favor, pode encerrar esta conversa agora",
    "por favor pode encerrar esta conversa agora hoje por favor",
    "encerra o chat",
    "pode finalizar a conversa",
    "quero encerrar todo o atendimento",
    "pode finalizar todo o atendimento",
    "quero encerrar tudo",
])
def test_close_request_recognizes_explicit_end_intents(text):
    assert is_close_request(text)


@pytest.mark.parametrize("text", [
    "quero sair do cheque especial",
    "preciso finalizar meu pedido de aumento",
    "como faço para sair do vermelho com meu limite?",
    "quero sair do rotativo do cartão",
    "preciso sair do negativado",
    "sair da lista de inadimplentes",
    "não quero mais esse limite",
    "não quero mais",
    "pode encerrar a entrevista",
    "quero finalizar a entrevista",
    "como faço para finalizar meu pedido",
    "obrigado",
    "ok",
    "5000",
    "sim",
    "não",
    "não quero mais nada por favor pode repetir isso hoje obrigado",
    "quero cancelar só a entrevista",
    "quero cancelar a operação",
    "não quero encerrar todo o atendimento",
    "se puder encerrar tudo",
])
def test_close_request_does_not_match_other_intents(text):
    assert not is_close_request(text)


@pytest.mark.parametrize("text", ["sim", "Claro, aceito", "pode ser", "vamos lá"])
def test_affirmative_replies_are_recognized(text):
    assert is_affirmative(text)


@pytest.mark.parametrize("text", [
    "sim", "SIM!", "ok", "Okay.", "positivo", "CONFIRMO", "confirmado!",
    "correto", "Está correto.", "com certeza", "Pode prosseguir!",
])
def test_explicit_interview_confirmations_accept_complete_positive_replies(text):
    assert is_explicit_interview_confirmation(text)


@pytest.mark.parametrize("text", [
    "sim, mas minha renda é 3000", "ok, só muda meus gastos", "confirmo, porém tenho dívidas",
    "não", "negativo", "talvez", "acho que sim", "pode prosseguir se estiver certo", "sim mesmo",
])
def test_explicit_interview_confirmations_reject_ambiguous_negative_or_uncertain_replies(text):
    assert not is_explicit_interview_confirmation(text)


@pytest.mark.parametrize("text", ["confirmo, podemos continuar", "CONFIRMO! Podemos continuar.",
    "sim, pode continuar", "isso mesmo, vamos seguir", "está correto e podemos prosseguir",
    "confirmado, pode seguir por favor"])
def test_interview_acknowledgement_with_continuation_is_data_confirmation_only(text):
    assert is_interview_data_confirmation(text)
    assert not is_explicit_interview_confirmation(text)  # Mantém a política de consentimento financeiro.


@pytest.mark.parametrize("text", ["não confirmo, podemos continuar", "acho que sim, podemos continuar",
    "confirmo se estiver certo, podemos continuar", "confirmo, mas meu emprego é autônomo",
    "sim, pode continuar, minha renda é 3000", "confirmo, podemos continuar e aumentar meu limite",
    "confirmo, podemos continuar se corrigir minha renda", "confirmo, porém não tenho dívidas"])
def test_confirmation_with_correction_condition_or_new_intent_stays_with_ai(text):
    assert not is_interview_data_confirmation(text)


@pytest.mark.parametrize("text", ["não", "prefiro não", "agora não", "recuso"])
def test_negative_replies_are_recognized(text):
    assert is_negative(text)


@pytest.mark.parametrize(("text", "expected"), [
    ("8000", "8000"),
    ("R$ 8.000,00", "R$ 8.000,00"),
    ("-10", "-10"),
    ("oito mil", None),
    ("meu limite é 8000", None),
    ("", None),
])
def test_limit_amount_requires_a_whole_numeric_input(text, expected):
    assert parse_limit_amount(text) == expected


@pytest.mark.parametrize("text", ["quero mais limite", "gostaria de aumentar", "aumento de limite"])
def test_limit_increase_without_amount_is_recognized(text):
    assert wants_limit_increase_without_amount(text)


@pytest.mark.parametrize("text", ["quero mais limite para 8000", "quero encerrar", "oi"])
def test_limit_increase_without_amount_rejects_amount_or_other_intents(text):
    assert not wants_limit_increase_without_amount(text)


# Cancelamento da entrevista e encerramento global.
@pytest.mark.parametrize("text", [
    "encerrar a entrevista",
    "pode encerrar a entrevista",
    "quero finalizar a entrevista",
    "não quero mais a entrevista",
    "desistir da entrevista",
    "quero cancelar a entrevista",
    "quero cancelar só a entrevista",
    "ENCERRAR A ENTREVISTA",
    "encerrar entrevista",
])
def test_wants_to_end_interview_only_recognizes_step_cancel(text):
    assert wants_to_end_interview_only(text)


@pytest.mark.parametrize("text", [
    "encerrar o atendimento",
    "encerrar a conversa",
    "quero sair",
    "tchau",
    "oi",
    "aumentar meu limite",
    "fazer a entrevista",   # aceite, não cancelamento
    "quanto tempo demora a entrevista",  # pergunta, não cancelamento
])
def test_wants_to_end_interview_only_rejects_other_intents(text):
    assert not wants_to_end_interview_only(text)


@pytest.mark.parametrize("text", [
    "encerrar o atendimento",
    "encerrar a conversa",
    "tchau",
    "não quero mais nada",
    "quero encerrar todo o atendimento",
    "pode finalizar todo o atendimento",
    "quero encerrar tudo",
])
def test_is_global_close_request_detects_explicit_global_end(text):
    assert is_global_close_request(text)


@pytest.mark.parametrize("text", [
    "encerrar",              # ambíguo: sem alvo global explícito
    "sair",
    "pode finalizar",
    "quero encerrar",
    "não sei",
])
def test_is_global_close_request_rejects_ambiguous_end(text):
    assert not is_global_close_request(text)


# Resposta à confirmação pendente de encerramento.
@pytest.mark.parametrize("text", [
    "sim",                       # confirma a primeira opção da pergunta (só a entrevista)
    "só a entrevista",
    "apenas a entrevista",
    "quero encerrar a entrevista",
    "pode encerrar só a entrevista",
])
def test_close_scope_reply_interview_scope(text):
    assert close_scope_reply(text) == "interview"


@pytest.mark.parametrize("text", [
    "encerrar tudo",
    "encerrar o atendimento",
    "tchau",
    "não quero mais nada",
    "finalizar tudo",
])
def test_close_scope_reply_global_scope_requires_explicit_target(text):
    assert close_scope_reply(text) == "global"


@pytest.mark.parametrize("text", ["não", "não obrigado", "prefiro não", "agora não"])
def test_close_scope_reply_no_keeps_conversation(text):
    assert close_scope_reply(text) == "no"


@pytest.mark.parametrize("text", [
    "como assim?",
    "não entendi",
    "o que você quis dizer?",
    "pode repetir?",
])
def test_close_scope_reply_clarification(text):
    assert close_scope_reply(text) == "clarify"


@pytest.mark.parametrize("text", [
    "quero aumentar meu limite",
    "quanto é o meu limite?",
    "5000",
    "",
])
def test_close_scope_reply_unrelated_returns_none(text):
    assert close_scope_reply(text) is None
