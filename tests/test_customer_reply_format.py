"""A formatação usa o estado validado sem alterar dados ou consentimentos."""
import copy

import pytest

from banco_agil.presentation import chat_markdown, customer_reply, separate_questions
from banco_agil.reply_recorder import ReplyRecorder
from banco_agil.response_guard import grounded_reply, interview_recovery_reply, pending_summary
from banco_agil.session import Session


def interview():
    s = Session(authenticated=True, active_agent="entrevista", cpf="local-fixture")
    s.flow.interview_started = True
    s.flow.interview_answers = {"renda_mensal": 123000}
    s.flow.pending_interview_answers = {"tipo_emprego": "autonomo", "despesas_fixas": 500,
                                         "num_dependentes": 9, "tem_dividas": True}
    s.credit_proposal = dict(valor=31000, novo_limite=31000, modalidade="limite_total", confirmada=False)
    return s


def test_pending_summary_uses_list_and_does_not_reconfirm_confirmed_income():
    s = interview()
    text = pending_summary(s)
    assert text.startswith("Dados para confirmação:\n\n- tipo de emprego: autônomo")
    assert "\n- despesas fixas: R$ 500,00" in text
    assert "\n- dependentes: 9" in text and "\n- possui dívidas: sim" in text
    assert "123.000" not in text and "31.000" not in text and ";" not in text


@pytest.mark.parametrize("active", ["cambio", "triagem", "credito"])
def test_interview_pending_survives_handoff_but_is_not_shown_in_another_subject(active):
    s = interview()
    s.active_agent = active
    state = copy.deepcopy(s.flow)
    summary = pending_summary(s)
    assert "dívidas" not in summary and "autônomo" not in summary
    if active == "credito":
        assert "31.000" in summary and "**confirmo**" in summary
    else:
        assert summary == ""
        assert customer_reply(s, "A cotação está disponível.") == "A cotação está disponível."
    assert s.flow == state


@pytest.mark.parametrize("track", [True, False])
def test_normal_and_recovery_replies_share_confirmation_instruction_without_effects(track):
    s = interview()
    flow, proposal = copy.deepcopy(s.flow), copy.deepcopy(s.credit_proposal)
    text = ReplyRecorder().say(s, "Confira os dados.\n\n" + pending_summary(s), track=track)
    assert text.count("me envie **confirmo** para continuar") == 1
    assert s.flow.interview_answers == flow.interview_answers
    assert s.flow.pending_interview_answers == flow.pending_interview_answers
    assert s.credit_proposal == proposal and not s.credit_proposal["confirmada"]
    assert s.history[-1]["content"] == text


def test_confirmation_instruction_is_present_only_while_pending_in_current_subject():
    s = interview()
    s.flow.pending_interview_answers.clear()
    assert "confirmo" not in customer_reply(s, "Obrigado pela confirmação. Qual é seu emprego?")
    assert s.credit_proposal["confirmada"] is False


def test_no_instruction_for_null_only_state_or_closed_or_unauthenticated_session():
    for kind in ("null", "ended", "unauthenticated"):
        s = interview()
        if kind == "null": s.flow.pending_interview_answers = {"tipo_emprego": None}
        elif kind == "ended": s.ended = True
        else: s.authenticated = False
        assert "confirmo" not in customer_reply(s, "Podemos conversar.")
        assert pending_summary(s) == ""


def test_zero_expenses_dependents_and_false_debts_are_not_dropped():
    s = interview()
    s.flow.pending_interview_answers = dict(despesas_fixas=0, num_dependentes=0, tem_dividas=False)
    text = pending_summary(s)
    assert "R$ 0,00" in text and "dependentes: 0" in text and "possui dívidas: não" in text


def test_natural_full_summary_becomes_state_backed_list_with_separate_question():
    s = interview()
    original = ("Resumo das informações fornecidas: despesas fixas de R$ 500,00; 9 dependentes; "
                "renda mensal de R$ 123.000,00; tipo de emprego: autônomo; possui dívidas: sim. Confirma?")
    text = customer_reply(s, original)
    assert "Resumo das informações fornecidas:\n\nInformações já confirmadas:\n\n- renda mensal: R$ 123.000,00" in text
    assert "\n- tipo de emprego: autônomo" in text and "\n- possui dívidas: sim\n\nConfirma?" in text
    assert ";" not in text and "me envie **confirmo** para continuar" in text
    assert original.startswith("Resumo das informações fornecidas: despesas")


def test_natural_summary_and_pipeline_pending_block_do_not_duplicate_values_or_request_old_data_again():
    s = interview()
    text = customer_reply(s, "Resumo das informações fornecidas: os dados da entrevista. Confirma?\n\n" + pending_summary(s))
    assert text.count("R$ 123.000,00") == 1 and text.count("R$ 500,00") == 1
    assert text.count("Dados para confirmação:") == 1
    pending = text.split("Dados para confirmação:", 1)[1]
    assert "123.000" not in pending and "**confirmo**" in pending
    assert customer_reply(s, text) == text


def test_ai_list_summary_is_not_expanded_or_duplicated_again():
    s = interview()
    text = "Resumo das informações fornecidas:\n\n- renda mensal: R$ 123.000,00\n- tipo de emprego: autônomo"
    formatted = customer_reply(s, text + "\n\n" + pending_summary(s))
    assert formatted.count("R$ 123.000,00") == 1 and formatted.count("autônomo") == 1
    assert formatted.count("R$ 500,00") == 1
    assert customer_reply(s, formatted) == formatted


@pytest.mark.parametrize("original,expected", [
    ("Seu limite é R$ 5.000,00. Precisa de mais alguma coisa?", "Seu limite é R$ 5.000,00.\n\nPrecisa de mais alguma coisa?"),
    ("Obrigado! Qual é sua renda?", "Obrigado!\n\nQual é sua renda?"),
    ("Tudo certo.\nQual é sua renda?", "Tudo certo.\n\nQual é sua renda?"),
    ("Seu limite é R$ 5.000,00.\n\nPrecisa de mais alguma coisa?", "Seu limite é R$ 5.000,00.\n\nPrecisa de mais alguma coisa?"),
    ("Qual é sua renda?", "Qual é sua renda?"),
    ("Seu limite é R$ 5.000,00.", "Seu limite é R$ 5.000,00."),
    ("`A. Pergunta?`", "`A. Pergunta?`"),
])
def test_question_separation_is_idempotent_and_preserves_numbers_and_markdown(original, expected):
    assert separate_questions(original) == expected
    assert separate_questions(expected) == expected
    assert chat_markdown(expected).replace(r"R\$", "R$") == expected


@pytest.mark.parametrize("old", ["Para autorizar este pedido, responda “confirmo”.", "Para confirmar esses dados, responda “confirmo" + "”."])
def test_legacy_confirmation_wording_is_replaced_once(old):
    s = interview()
    text = customer_reply(s, old)
    assert text.count("me envie **confirmo** para continuar") == 1 and "responda" not in text


def test_recovery_preserves_failure_notice_pending_data_and_list():
    s = interview()
    s.last_failure_category = "http_429"
    text = interview_recovery_reply(s)
    assert "alta demanda" in text and "entrevista continua ativa" in text
    assert "\n- tipo de emprego: autônomo" in text and "**confirmo**" in text
    assert "123.000" not in text and "31.000" not in text


@pytest.mark.parametrize("question", ["Você possui reserva de emergência?",
    "Você tem outros créditos ativos?", "Qual o valor das parcelas de seus empréstimos?"])
def test_extra_interview_fields_are_reoriented_without_inventing_data(question):
    s = interview()
    s.flow.pending_interview_answers.clear()
    state = copy.deepcopy(s.flow)
    checked = grounded_reply(s, question)
    assert "emprego" in checked and "?" in checked
    assert s.last_failure_category == "ungrounded_response" and s.flow == state


@pytest.mark.parametrize("question", ["Você tem dívidas ativas, como empréstimos ou financiamentos?",
    "Você possui dívidas?", "Quantos dependentes você tem?", "Qual é seu tipo de emprego?",
    "Qual é o valor de suas despesas fixas e você tem dívidas?"])
def test_legitimate_interview_questions_stay_ai_written(question):
    assert grounded_reply(interview(), question) == question


def test_extra_question_while_pending_requests_only_existing_proposal_confirmation():
    s = interview()
    state = copy.deepcopy(s.flow)
    text = grounded_reply(s, "Você possui reserva de emergência?")
    assert "Dados para confirmação:\n\n- " in text and "**confirmo**" in text
    assert "reserva" not in text and "31.000" not in text and s.flow == state


def test_credit_must_not_reask_existing_request_amount_when_interview_is_unfinished():
    s = interview()
    s.active_agent = "credito"
    s.last_request_id, s.last_request_status, s.desired_limit = 7, "reprovado", 31000
    flow, proposal = copy.deepcopy(s.flow), copy.deepcopy(s.credit_proposal)
    text = grounded_reply(s, "Qual o novo limite total que deseja solicitar?")
    assert "31.000" in text and "retomar a entrevista" in text and "Qual" not in text
    assert s.flow == flow and s.credit_proposal == proposal
    assert not s.credit_proposal["confirmada"]


def test_unfinished_interview_request_does_not_display_another_credit_proposal_as_authorizable():
    s = interview()
    s.active_agent = "credito"
    s.last_request_id, s.last_request_status, s.desired_limit = 7, "reprovado", 31000
    proposal = copy.deepcopy(s.credit_proposal)
    assert pending_summary(s) == ""
    assert "confirmo" not in customer_reply(s, "O pedido anterior está preservado. Quer retomar a entrevista?")
    assert s.credit_proposal == proposal


def test_credit_may_ask_amount_for_an_actual_new_request_without_interview_origin():
    s = interview()
    s.active_agent = "credito"
    text = "Qual o novo limite total que deseja solicitar?"
    assert grounded_reply(s, text) == text


@pytest.mark.parametrize("question", ["Qual é sua renda mensal?", "Você confirma emprego autônomo?",
    "Confirma sua renda e seu tipo de emprego?", "Qual o seu salário?"])
def test_repeated_confirmed_interview_question_is_reoriented_only_as_output_recovery(question):
    s = interview()
    s.flow.pending_interview_answers.clear()
    s.flow.interview_answers["tipo_emprego"] = "autonomo"
    before = copy.deepcopy(s.flow)
    assert grounded_reply(s, question) == "Qual é o valor das suas despesas fixas mensais?"
    assert s.flow == before and "já confirmados" in s.last_failure_reason
    assert not s.credit_proposal["confirmada"]


@pytest.mark.parametrize("question", [
    "Sua renda já está confirmada. Qual é o valor de suas despesas fixas?",
    "Você confirma sua renda e quais são suas despesas fixas?",
    "Seu emprego é autônomo. Quantos dependentes você tem?",
    "Você tem dívidas ativas?", "Quer consultar seu limite ou atualizar sua renda?",
    "Quer entender como sua renda influencia o score?", "Precisa de mais alguma coisa?",
])
def test_missing_fields_new_topics_and_context_keep_natural_ai_question(question):
    s = interview()
    s.flow.pending_interview_answers.clear()
    s.flow.interview_answers["tipo_emprego"] = "autonomo"
    assert grounded_reply(s, question) == question


def test_explicit_revision_can_reask_a_previously_confirmed_field():
    s = interview()
    s.flow.pending_interview_answers.clear()
    s.flow.interview_revision_requested = True
    question = "Qual é sua renda mensal corrigida?"
    assert grounded_reply(s, question) == question


@pytest.mark.parametrize("mode", ["native", "structured"])
@pytest.mark.parametrize("question", ["Qual é sua renda mensal?", "Você confirma emprego autônomo?"])
def test_repeated_question_after_confirmation_advances_without_another_generation_or_financial_effect(repo_ana_620, mode, question):
    from banco_agil.flow import FlowController
    from orchestrator_fakes import logged_in
    from structured_fakes import make, message, check_protocol
    from support import snapshot_csv as csv
    orch = make(repo_ana_620, [message(question)], mode)
    s = logged_in(orch, agent="entrevista")
    FlowController.start_interview(s)
    s.flow.interview_answers = {"renda_mensal": 123000}
    s.flow.pending_interview_answers = {"tipo_emprego": "autonomo"}
    s.credit_proposal = dict(valor=31000, novo_limite=31000, modalidade="limite_total", confirmada=False)
    before, proposal = csv(orch.repo), copy.deepcopy(s.credit_proposal)
    reply = orch.handle(s, "confirmo")
    assert "despesas fixas" in reply and "?" in reply and "**confirmo**" not in reply
    assert s.flow.interview_answers == {"renda_mensal": 123000, "tipo_emprego": "autonomo"}
    assert not s.flow.pending_interview_answers and s.credit_proposal == proposal
    assert csv(orch.repo) == before and len(orch.client.requests) == 1
    assert "já confirmados" in s.last_failure_reason
    check_protocol(orch, s)
