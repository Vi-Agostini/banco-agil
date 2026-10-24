import pytest

from banco_agil.flow import Awaiting
from banco_agil.interceptors import ConversationInterceptors, interview_offer_reply, limit_amount_reply
from banco_agil.session import Session


@pytest.mark.parametrize(("text", "expected"), [
    ("sim", True),
    ("Claro, aceito", True),
    ("não", False),
    ("prefiro não", False),
    ("qual é o meu limite atual?", None),
])
def test_interview_offer_intercepts_only_clear_yes_or_no(text, expected):
    assert interview_offer_reply(text) is expected


@pytest.mark.parametrize("text", ["8000", "R$ 8.000,00", "0", "-10"])
def test_limit_interceptor_accepts_clearly_numeric_values(text):
    assert limit_amount_reply(text) == text


@pytest.mark.parametrize("text", ["qual é o meu limite atual?", "quero aumentar", "oito mil"])
def test_limit_interceptor_leaves_non_numeric_text_for_llm(text):
    assert limit_amount_reply(text) is None


@pytest.mark.parametrize(("state", "text"), [
    ("interview_offer", "qual é o meu limite atual?"),
    ("limit_amount", "qual é o meu limite atual?"),
    ("limit_amount", "quero aumentar"),
])
def test_pending_state_non_answers_fall_through_to_llm(state, text):
    session = Session(active_agent="credito")
    if state == "interview_offer":
        session.flow.awaiting = Awaiting.INTERVIEW_OFFER
    else:
        session.flow.awaiting = Awaiting.LIMIT_AMOUNT

    assert ConversationInterceptors(object(), object(), object()).try_handle(session, text) is None
    if state == "interview_offer":
        assert session.flow.awaiting is Awaiting.INTERVIEW_OFFER
    else:
        assert session.flow.awaiting is Awaiting.LIMIT_AMOUNT
