"""Encaminhamento e retomada do crédito com decisões simuladas nos dois protocolos."""
import copy
import json

import pytest

from banco_agil.agents import AGENTS
from banco_agil.flow import Awaiting
from banco_agil.llm import _tool_schemas
from orchestrator_fakes import logged_in
from structured_fakes import action, check_protocol, make, message
from support import snapshot_csv as csv


@pytest.mark.parametrize("mode", ["native", "structured"])
def test_cr16_query_then_amount_needs_consent_and_never_restarts_login(repo_ana_620, mode):
    orch = make(repo_ana_620, [
        action("transferir_para_agente", destino="credito"),
        action("consultar_limite"),
        message("Seu limite atual é R$ 2.000,00."),
        action("registrar_intencao_credito", valor=7000, modalidade="limite_total"),
        action("registrar_intencao_credito", valor=7000, modalidade="limite_total"),
        action("solicitar_aumento_limite", novo_limite=7000),
    ], mode)
    s = logged_in(orch, cpf="52998224725", nome="Bruno Lima")
    identity = (s.cpf, s.nome, s.correlation_id)
    before = csv(orch.repo)

    # A IA interpreta o erro de digitação, sem acrescentar uma regra local de palavras-chave.
    reply = orch.handle(s, "qual meu limte")
    assert "2.000,00" in reply and "deseja solicitar" not in reply
    assert s.active_agent == "credito" and s.credit_proposal is None
    assert s.tool_facts["consultar_limite"]["result"]["limite_credito_atual"] == 2000
    assert csv(orch.repo) == before
    assert orch.client.requests[0]["messages"][-1]["content"] == "qual meu limte"
    assert orch.client.requests[1]["messages"][-1]["role"] in {"system", "tool"}

    reply = orch.handle(s, "7000")
    assert "7.000,00" in reply and "me envie **confirmo** para continuar" in reply
    assert "identidade foi confirmada" not in reply
    assert s.last_failure_category == "progress_stalled"
    assert not s.credit_proposal["confirmada"] and not s.ended
    assert csv(orch.repo) == before and not orch.repo.list_requests()
    assert s.authenticated and s.auth_failures == 0 and identity == (s.cpf, s.nome, s.correlation_id)

    reply = orch.handle(s, "confirmo")
    assert "não foi aprovado" in reply
    assert len(orch.repo.list_requests()) == 1
    assert orch.repo.get_client(s.cpf)["limite_credito"] == 2000
    assert s.flow.awaiting is Awaiting.INTERVIEW_OFFER
    check_protocol(orch, s)


@pytest.mark.parametrize("mode", ["native", "structured"])
def test_authenticated_triage_exposes_routing_and_blocks_old_collection(repo_ana_620, mode):
    orch = make(repo_ana_620, [action("registrar_intencao_credito", valor=0, modalidade="limite_total")], mode)
    s = logged_in(orch)
    s.llm_mode = mode
    names = {t["function"]["name"] for t in _tool_schemas(AGENTS["triagem"], s)}
    assert "transferir_para_agente" in names
    assert "registrar_intencao_credito" not in names
    assert "autenticar_cliente" not in names
    before = csv(orch.repo)
    direct = orch.tool_executor.execute_from_llm(AGENTS["triagem"], s,
        "registrar_intencao_credito", '{"valor":7000,"modalidade":"limite_total"}')
    assert direct["codigo"] == "encaminhamento_credito_obrigatorio"
    reply = orch.handle(s, "qual meu limte")
    assert "identidade foi confirmada" not in reply
    assert s.credit_proposal is None and s.authenticated and s.active_agent == "triagem"
    assert s.last_failure_category in {"invalid_tool_response", "invalid_structured_response"}
    assert csv(orch.repo) == before and not orch.repo.list_requests()
    # O decoder estruturado recebe o mesmo schema restrito com strict=true.
    request = orch.client.requests[0]
    offered = json.dumps(request.get("response_format", request.get("tools")), ensure_ascii=False)
    assert "registrar_intencao_credito" not in offered
    check_protocol(orch, s)


@pytest.mark.parametrize("mode", ["native", "structured"])
def test_prelogin_amount_survives_routing_and_timeout(repo_ana_620, mode):
    orch = make(repo_ana_620, [
        action("registrar_intencao_credito", valor=7000, modalidade="limite_total"),
        message("Para continuar, informe seu CPF."),
        action("transferir_para_agente", destino="credito"),
        TimeoutError(),
        message("Você deseja solicitar esse novo limite?"),
    ], mode)
    s = orch.new_session()
    before = csv(orch.repo)
    orch.handle(s, "Quero sete mil de limite")
    proposal = copy.deepcopy(s.credit_proposal)
    assert proposal["valor"] == 7000 and not proposal["confirmada"]
    # Autentica pelo coletor local existente.
    orch.handle(s, "52998224725")
    orch.handle(s, "30/11/1985")
    assert s.authenticated
    reply = orch.handle(s, "podemos prosseguir")
    assert s.active_agent == "credito" and s.last_failure_category == "timeout"
    assert s.credit_proposal == proposal
    assert "7.000,00" in reply and "me envie **confirmo** para continuar" in reply
    orch.handle(s, "continue de onde paramos")
    assert s.active_agent == "credito" and s.credit_proposal == proposal
    assert s.authenticated and s.auth_failures == 0 and csv(orch.repo) == before
    check_protocol(orch, s)


@pytest.mark.parametrize("mode", ["native", "structured"])
@pytest.mark.parametrize("pending", [False, True])
def test_triage_progress_guard_preserves_pending_data_without_login_greeting(repo_ana_620, mode, pending):
    orch = make(repo_ana_620, [action("registrar_entrada_invalida"), action("registrar_entrada_invalida")], mode)
    s = logged_in(orch)
    # Simula uma proposta guardada antes da autenticação ou por uma sessão antiga.
    if pending:
        s.credit_proposal = {"valor": 8000, "modalidade": "limite_total", "novo_limite": 8000,
                             "turno": 0, "confirmada": False}
    proposal = copy.deepcopy(s.credit_proposal)
    before = csv(orch.repo)
    reply = orch.handle(s, "podemos prosseguir")
    assert "identidade foi confirmada" not in reply
    assert s.authenticated and s.auth_failures == 0 and not s.ended
    assert s.last_failure_category == "progress_stalled" and s.credit_proposal == proposal
    assert csv(orch.repo) == before and not orch.repo.list_requests()
    if pending:
        assert s.active_agent == "credito" and "8.000,00" in reply
        assert "me envie **confirmo** para continuar" in reply
        assert s.tool_facts["transferir_para_agente"]["result"]["ok"]
    else:
        assert s.active_agent == "triagem" and "Como posso ajudar" not in reply
    check_protocol(orch, s)


@pytest.mark.parametrize("mode", ["native", "structured"])
@pytest.mark.parametrize("exhausted", ["same_handoff", "transfer_budget"])
def test_triage_recovery_cannot_bypass_existing_transfer_guards(repo_ana_620, mode, exhausted):
    if exhausted == "same_handoff":
        script = [action("transferir_para_agente", destino="credito"),
                  action("transferir_para_agente", destino="triagem"),
                  action("transferir_para_agente", destino="credito")]
        transitions = 2
    else:
        script = [action("transferir_para_agente", destino="cambio"),
                  action("transferir_para_agente", destino="credito"),
                  action("transferir_para_agente", destino="triagem"),
                  action("registrar_entrada_invalida"), action("registrar_entrada_invalida")]
        transitions = 3
    orch = make(repo_ana_620, script, mode)
    s = logged_in(orch)
    s.credit_proposal = {"valor": 8000, "modalidade": "limite_total", "novo_limite": 8000,
                         "turno": 0, "confirmada": False}
    proposal = copy.deepcopy(s.credit_proposal)
    before = csv(orch.repo)
    execute = orch.tool_executor.execute_from_llm
    handoffs = []

    def counted(agent, session, name, args):
        result = execute(agent, session, name, args)
        if name == "transferir_para_agente" and result.get("ok"):
            handoffs.append(result)
        return result

    orch.tool_executor.execute_from_llm = counted
    reply = orch.handle(s, "podemos prosseguir")
    assert len(handoffs) == transitions
    assert s.active_agent == "triagem" and s.authenticated and not s.ended
    assert s.credit_proposal == proposal and s.last_failure_category == "progress_stalled"
    assert "identidade foi confirmada" not in reply
    assert csv(orch.repo) == before and not orch.repo.list_requests()
    check_protocol(orch, s)
