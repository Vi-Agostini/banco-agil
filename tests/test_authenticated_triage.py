"""A Triagem autenticada encaminha assuntos sem autorizar operações financeiras."""
import copy

import pytest

from banco_agil.response_guard import grounded_reply
from orchestrator_fakes import logged_in
from structured_fakes import action, check_protocol, make, message
from support import snapshot_csv as csv


TOPIC = "Quero conversar sobre meu limite de crédito."
OLD_REPLY = "Por favor, informe seu CPF para que eu possa prosseguir com a solicitação de limite de crédito."


@pytest.mark.parametrize("mode", ["native", "structured"])
def test_authenticated_credit_topic_routes_before_conversational_reply(repo_ana_620, mode):
    answer = "Prefere consultar o limite atual ou conversar sobre um aumento?"
    orch = make(repo_ana_620, [message(answer)], mode)
    session = logged_in(orch)
    before = csv(orch.repo)
    assert orch.handle(session, TOPIC) == answer
    assert session.active_agent == "credito" and session.authenticated and session.auth_failures == 0
    assert session.tool_facts["transferir_para_agente"]["result"]["ok"] is True
    assert session.credit_proposal is None and csv(orch.repo) == before
    users = [m["content"] for m in orch.client.requests[0]["messages"] if m["role"] == "user"]
    assert users[-1] == TOPIC
    assert "consulta de limite de crédito" in orch.client.requests[0]["messages"][0]["content"]
    check_protocol(orch, session)


@pytest.mark.parametrize("mode", ["native", "structured"])
def test_recorded_two_replies_cannot_restart_authenticated_session(repo_ana_620, mode):
    orch = make(repo_ana_620, [message("Como posso ajudar com seu limite de crédito?"), message(OLD_REPLY)], mode)
    session = logged_in(orch)
    identity = (session.cpf, session.nome, session.correlation_id)
    before = csv(orch.repo)
    orch.handle(session, TOPIC)
    reply = orch.handle(session, "8000 reais")
    assert "cpf" not in reply.casefold() and "nascimento" not in reply.casefold()
    assert session.last_failure_category == "ungrounded_response"
    assert session.active_agent == "credito" and session.authenticated and session.auth_failures == 0
    assert identity == (session.cpf, session.nome, session.correlation_id)
    assert len(orch.client.requests) == 2 and session.credit_proposal is None
    assert csv(orch.repo) == before and not orch.repo.list_requests()
    assert session.history[-2]["content"] == "8000 reais"
    check_protocol(orch, session)


@pytest.mark.parametrize("mode", ["native", "structured"])
def test_amount_context_survives_local_handoff_and_needs_confirmation(repo_ana_620, mode):
    orch = make(repo_ana_620, [message("Que limite total pretende solicitar?"),
        action("registrar_intencao_credito", valor=8000, modalidade="limite_total"),
        message("Confirma o limite total de R$ 8.000,00?"),
        action("solicitar_aumento_limite", novo_limite=8000),
        message("O pedido foi aprovado e o limite passou a R$ 8.000,00.")], mode)
    session = logged_in(orch)
    before = csv(orch.repo)
    orch.handle(session, TOPIC)
    orch.handle(session, "8000 reais")
    assert session.credit_proposal["valor"] == session.credit_proposal["novo_limite"] == 8000
    assert session.credit_proposal["modalidade"] == "limite_total"
    assert session.credit_proposal["confirmada"] is False and csv(orch.repo) == before
    assert not orch.repo.list_requests()
    assert any(m.get("content") == TOPIC for m in orch.client.requests[1]["messages"])
    assert any(m.get("content") == "8000 reais" for m in orch.client.requests[1]["messages"])
    orch.handle(session, "sim")
    assert len(orch.repo.list_requests()) == 1 and orch.repo.get_client(session.cpf)["limite_credito"] == 8000
    assert len(orch.client.requests) == 5
    check_protocol(orch, session)


@pytest.mark.parametrize("mode", ["native", "structured"])
def test_unauthenticated_credit_still_collects_identity(repo_ana_620, mode):
    answer = "Para começar a identificação, informe seu CPF."
    orch = make(repo_ana_620, [message(answer)], mode)
    session = orch.new_session()
    before = csv(orch.repo)
    assert orch.handle(session, TOPIC) == answer
    assert not session.authenticated and session.active_agent == "triagem"
    assert not session.tool_facts and csv(orch.repo) == before
    check_protocol(orch, session)


@pytest.mark.parametrize("text", [
    "Não quero conversar sobre meu limite.", "Qual é o limite de velocidade?",
    "Qual a cotação do dólar antes de conversar sobre meu limite?",
    "Quero falar sobre a entrevista e meu limite.", "8000 reais",
])
def test_uncertain_or_multiple_topics_remain_with_IA(repo_ana_620, text):
    answer = "Qual desses assuntos deseja abordar primeiro?"
    orch = make(repo_ana_620, [message(answer)])
    session = logged_in(orch)
    assert orch.handle(session, text) == answer
    assert session.active_agent == "triagem" and "transferir_para_agente" not in session.tool_facts
    check_protocol(orch, session)


@pytest.mark.parametrize("mode", ["native", "structured"])
def test_exchange_handoff_and_explicit_close_are_preserved(repo_ana_620, mode):
    orch = make(repo_ana_620, [action("transferir_para_agente", destino="cambio"),
        message("Qual moeda deseja consultar?")], mode)
    session = logged_in(orch)
    before = csv(orch.repo)
    orch.handle(session, "Qual é a cotação do dólar?")
    assert session.active_agent == "cambio" and session.authenticated
    orch.handle(session, "quero encerrar todo o atendimento")
    assert session.ended and csv(orch.repo) == before
    assert len(orch.client.requests) == 2
    check_protocol(orch, session)


@pytest.mark.parametrize("text", [
    OLD_REPLY, "Poderia me informar seu CPF?", "Qual é sua data de nascimento?",
    "Preciso do seu CPF para continuar.", "Informe seu CPF; não farei nenhuma operação.",
    "Não entendi, informe seu CPF para continuar.",
])
def test_authenticated_identity_request_is_blocked_without_session_mutation(repo_ana_620, text):
    orch = make(repo_ana_620, [])
    session = logged_in(orch)
    before = copy.deepcopy((session.cpf, session.nome, session.authenticated, session.auth_failures,
                           session.history, session.credit_proposal))
    assert grounded_reply(session, text) != text
    assert before == (session.cpf, session.nome, session.authenticated, session.auth_failures,
                      session.history, session.credit_proposal)
    assert not orch.client.requests and not orch.repo.list_requests()


@pytest.mark.parametrize("text", [
    "Não precisa informar CPF novamente.", "Nunca envie seu CPF em mensagens públicas.",
    "Seu CPF continua protegido.", "Qual a data de atualização da cotação?",
])
def test_legitimate_identity_mentions_and_other_questions_are_preserved(repo_ana_620, text):
    orch = make(repo_ana_620, [])
    session = logged_in(orch)
    assert grounded_reply(session, text) == text
