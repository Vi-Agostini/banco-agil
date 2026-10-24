"""Confirmar dados da entrevista pela IA não autoriza uma operação financeira."""
import copy
import json

import pytest

from banco_agil.agents import AGENTS
from banco_agil.flow import FlowController
from banco_agil.intents import is_interview_data_confirmation
from banco_agil.response_guard import grounded_reply
from fakes import failure
from orchestrator_fakes import logged_in
from structured_fakes import action, message, make, check_protocol
from support import snapshot_csv as csv


CONFIRM = "confirmar_dados_entrevista"


def session(orch):
    s = logged_in(orch, agent="entrevista")
    FlowController.start_interview(s)
    s.flow.pending_interview_answers = {"renda_mensal": 123000}
    s.credit_proposal = dict(valor=31000, modalidade="limite_total", novo_limite=31000, confirmada=False)
    return s


@pytest.mark.parametrize("mode", ["native", "structured"])
@pytest.mark.parametrize("text", ["Corretíssimo", "É exatamente isso", "Perfeitamente"])
def test_free_language_confirmation_changes_state_and_continues_even_if_ai_only_thanks(repo_ana_620, mode, text):
    assert not is_interview_data_confirmation(text)  # A resposta exige interpretação da IA, sem criar outro sinônimo local.
    orch = make(repo_ana_620, [action(CONFIRM), message("Obrigado pela confirmação."),
                              message("Quais são suas despesas fixas mensais?")], mode)
    s = session(orch)
    before, proposal = csv(orch.repo), copy.deepcopy(s.credit_proposal)
    reply = orch.handle(s, text)
    assert "?" in reply and "emprego" in reply and "Dados para confirmação: renda" not in reply
    assert s.flow.interview_answers == {"renda_mensal": 123000} and not s.flow.pending_interview_answers
    orch.handle(s, "Qual próxima pergunta")
    assert not s.flow.pending_interview_answers and not s.flow.interview_revision_requested
    assert s.credit_proposal == proposal and csv(orch.repo) == before
    assert s.authenticated and s.active_agent == "entrevista" and len(orch.client.requests) == 3
    assert "respostas_confirmadas" in str(orch.client.requests[1]["messages"])
    check_protocol(orch, s)


@pytest.mark.parametrize("mode", ["native", "structured"])
def test_failed_wording_after_semantic_confirmation_preserves_the_confirmed_data(repo_ana_620, mode):
    orch = make(repo_ana_620, [action(CONFIRM), failure("json_validate_failed"), failure("json_validate_failed")], mode)
    s = session(orch)
    before = csv(orch.repo)
    reply = orch.handle(s, "Corretíssimo")
    assert "emprego" in reply and not s.flow.pending_interview_answers
    assert s.flow.interview_answers == {"renda_mensal": 123000} and csv(orch.repo) == before
    assert not s.credit_proposal["confirmada"] and len(orch.client.requests) == 3
    check_protocol(orch, s)


@pytest.mark.parametrize("mode", ["native", "structured"])
def test_correction_and_condition_are_interpreted_without_confirming_old_or_new_data(repo_ana_620, mode):
    orch = make(repo_ana_620, [action("registrar_respostas_entrevista", renda_mensal=12000),
                              message("Você confirma renda de R$ 12.000,00?"),
                              message("Posso esclarecer sua dúvida antes de confirmar.")], mode)
    s = session(orch)
    before = csv(orch.repo)
    orch.handle(s, "Na verdade, minha renda é doze mil, não cento e vinte e três mil")
    assert s.flow.pending_interview_answers == {"renda_mensal": 12000} and not s.flow.interview_answers
    orch.handle(s, "Confirmo se você não aumentar meu limite automaticamente")
    assert s.flow.pending_interview_answers == {"renda_mensal": 12000} and not s.flow.interview_answers
    assert csv(orch.repo) == before and not s.credit_proposal["confirmada"]
    check_protocol(orch, s)


@pytest.mark.parametrize("invalid", ["no_pending", "no_interview", "unauthenticated", "fresh", "no_reply", "wrong_agent"])
def test_confirmation_tool_is_unavailable_without_earlier_proposal_and_new_reply(repo_ana_620, invalid):
    orch = make(repo_ana_620, [])
    s = session(orch)
    s.flow.interview_revision_requested = True
    s.turn = 2
    if invalid == "no_pending": s.flow.pending_interview_answers.clear()
    elif invalid == "no_interview": s.flow.interview_started = False
    elif invalid == "unauthenticated": s.authenticated = False
    elif invalid == "fresh": s.tool_facts["registrar_respostas_entrevista"] = {"turn": s.turn}
    elif invalid == "no_reply": s.flow.interview_revision_requested = False
    elif invalid == "wrong_agent": s.active_agent = "credito"
    state, before = copy.deepcopy(s.flow), csv(orch.repo)
    result = orch.tool_executor.execute_from_llm(AGENTS[s.active_agent], s, CONFIRM, "{}")
    assert "erro" in result and s.flow == state and csv(orch.repo) == before
    assert not s.credit_proposal["confirmada"]


@pytest.mark.parametrize("mode", ["native", "structured"])
def test_confirmation_cannot_be_combined_with_collecting_new_data(repo_ana_620, mode):
    # Todo o lote nativo é validado antes de executar qualquer ação.
    from types import SimpleNamespace as NS
    from banco_agil.tool_contracts import ToolContractError, validate_batch
    orch = make(repo_ana_620, [])
    s = session(orch)
    s.flow.interview_revision_requested = True
    before = copy.deepcopy(s.flow)
    calls = [NS(function=NS(name=CONFIRM, arguments="{}")),
             NS(function=NS(name="registrar_respostas_entrevista", arguments=json.dumps({"tipo_emprego": "formal"})))]
    with pytest.raises(ToolContractError, match="confirmacao_isolada_obrigatoria"):
        validate_batch(AGENTS["entrevista"], s, calls)
    assert s.flow == before


def test_text_cannot_announce_pending_income_as_confirmed(repo_ana_620):
    orch = make(repo_ana_620, [])
    s = session(orch)
    before = copy.deepcopy(s.flow)
    result = grounded_reply(s, "Obrigado pela confirmação. Sua renda mensal de R$ 123.000,00 foi confirmada.")
    assert "me envie **confirmo** para continuar" in result and "foi confirmada" not in result
    assert s.flow == before


def test_wording_may_refer_to_a_different_already_confirmed_field(repo_ana_620):
    orch = make(repo_ana_620, [])
    s = session(orch)
    s.flow.interview_answers = {"tipo_emprego": "formal"}
    text = "Seu emprego formal está confirmado. Você confirma a renda mensal informada?"
    assert grounded_reply(s, text) == text


def test_confirming_again_cannot_mutate_state_or_authorize_credit(repo_ana_620):
    orch = make(repo_ana_620, [])
    s = session(orch)
    s.flow.interview_revision_requested = True
    result = orch.tool_executor.execute_from_llm(AGENTS["entrevista"], s, CONFIRM, "{}")
    assert result["ok"] and not s.flow.pending_interview_answers
    state, before = copy.deepcopy(s.flow), csv(orch.repo)
    again = orch.tool_executor.execute_from_llm(AGENTS["entrevista"], s, CONFIRM, "{}")
    assert "erro" in again and s.flow == state and csv(orch.repo) == before
    assert not s.credit_proposal["confirmada"]


@pytest.mark.parametrize("mode", ["native", "structured"])
def test_last_field_confirmation_still_allows_score_once(repo_ana_620, mode):
    answers = dict(renda_mensal=6000, tipo_emprego="formal", despesas_fixas=2000,
                   num_dependentes=1, tem_dividas=False)
    orch = make(repo_ana_620, [action(CONFIRM), action("registrar_entrevista_e_calcular_score", **answers),
                              message("Seu score foi atualizado.")], mode)
    s = session(orch)
    s.flow.interview_answers = {k: v for k, v in answers.items() if k != "tem_dividas"}
    s.flow.pending_interview_answers = {"tem_dividas": False}
    orch.handle(s, "Corretíssimo")
    assert orch.repo.get_client(s.cpf)["score"] == 570
    assert not orch.repo.list_requests() and not s.credit_proposal["confirmada"]
    assert not s.flow.interview_started and not s.flow.pending_interview_answers
    check_protocol(orch, s)
