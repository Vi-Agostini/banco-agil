"""Inspeciona o request local; respostas simuladas não validam o modelo remoto."""
import copy
import json
import logging

from jsonschema import Draft202012Validator
import pytest

from banco_agil.agent_context import without_completed_proposals
from banco_agil.agents import AGENTS
from banco_agil.flow import FlowController
from banco_agil.llm import _tool_schemas
from banco_agil.session import Session
from fakes import failure
from orchestrator_fakes import logged_in
from structured_fakes import action, check_protocol, make, message
from support import snapshot_csv as csv


COLLECT = "registrar_respostas_entrevista"


def envelope(value, *, role="assistant"):
    return {"role": role, "content": json.dumps({"resposta": value}, ensure_ascii=False)}


def receipt(name, result):
    return {"role": "system", "content": json.dumps({
        "resultado_local": {"acao": name, "resultado": result}}, ensure_ascii=False)}


def proposed_actions(messages):
    names = []
    for item in messages:
        if item.get("role") != "assistant":
            continue
        try:
            value = json.loads(item["content"])["resposta"]
            if value["tipo"] == "acao":
                names.append(value["acao"]["nome"])
        except (ValueError, KeyError, TypeError):
            pass
    return names


def local_receipts(messages):
    names = []
    for item in messages:
        if item.get("role") != "system":
            continue
        try:
            names.append(json.loads(item["content"])["resultado_local"]["acao"])
        except (ValueError, KeyError, TypeError):
            pass
    return names


def assert_presentation_request(request):
    """As instruções não podem sugerir uma ação excluída do schema."""
    fmt = request["response_format"]
    assert fmt["json_schema"]["strict"] is True
    validator = Draft202012Validator(fmt["json_schema"]["schema"])
    assert validator.is_valid({"resposta": message("Podemos continuar?")})
    assert not validator.is_valid({"resposta": action(COLLECT, tipo_emprego="formal")})
    assert "tools" not in request
    prefix = request["messages"][0]["content"]
    assert "tipo=acao" not in prefix
    assert "chame registrar_respostas_entrevista" not in prefix
    assert "chame registrar_entrevista_e_calcular_score" not in prefix
    trusted = "\n".join(m["content"] for m in request["messages"] if m["role"] == "system")
    assert "Identidade já validada" in trusted
    assert not proposed_actions(request["messages"])


def forbid_financial_execution(monkeypatch, orch):
    original = orch.tool_executor.execute_from_llm
    calls = []

    def execute(agent, session, name, raw):
        calls.append(name)
        assert name not in {"solicitar_aumento_limite", "registrar_entrevista_e_calcular_score"}
        return original(agent, session, name, raw)

    monkeypatch.setattr(orch.tool_executor, "execute_from_llm", execute)
    return calls


def test_income_confirmation_employment_and_next_question_have_consistent_wire_requests(
        repo_ana_620, monkeypatch):
    orch = make(repo_ana_620, [
        action(COLLECT, renda_mensal=50000),
        message("Você confirma renda mensal de R$ 50.000,00?"),
        message("Qual é o seu tipo de emprego?"),
        action(COLLECT, tipo_emprego="formal"),
        message("Você confirma emprego formal?"),
        message("Quais são suas despesas fixas mensais?"),
    ])
    s = logged_in(orch, agent="entrevista")
    FlowController.start_interview(s)
    s.credit_proposal = dict(valor=90000, modalidade="limite_total", novo_limite=90000,
                             confirmada=False)
    proposal_before, files_before = copy.deepcopy(s.credit_proposal), csv(orch.repo)
    calls = forbid_financial_execution(monkeypatch, orch)

    assert orch.handle(s, "50 mil reais").startswith("Você confirma renda mensal")
    assert s.flow.pending_interview_answers == {"renda_mensal": 50000}
    assert not s.flow.interview_answers
    assert orch.handle(s, "confirmo").startswith("Qual é o seu tipo de emprego?")
    assert s.flow.interview_answers == {"renda_mensal": 50000}
    assert not s.flow.pending_interview_answers
    assert orch.handle(s, "formal").startswith("Você confirma emprego formal?")
    assert s.flow.pending_interview_answers == {"tipo_emprego": "formal"}
    assert s.flow.interview_answers == {"renda_mensal": 50000}
    assert orch.handle(s, "confirmo").startswith("Quais são suas despesas fixas mensais?")
    assert s.flow.interview_answers == {"renda_mensal": 50000, "tipo_emprego": "formal"}
    assert not s.flow.pending_interview_answers

    assert len(orch.client.requests) == 6
    for index in (1, 2, 4, 5):
        assert_presentation_request(orch.client.requests[index])
    assert COLLECT in local_receipts(orch.client.requests[1]["messages"])
    assert "confirmacao_entrevista" in local_receipts(orch.client.requests[2]["messages"])
    assert COLLECT in local_receipts(orch.client.requests[4]["messages"])
    assert "confirmacao_entrevista" in local_receipts(orch.client.requests[5]["messages"])
    assert proposed_actions(s.history) == [COLLECT, COLLECT]
    assert calls == [COLLECT, COLLECT]
    assert s.credit_proposal == proposal_before and csv(orch.repo) == files_before
    assert s.active_agent == "entrevista" and s.authenticated and not s.ended
    assert not s.last_failure_category
    check_protocol(orch, s)


def test_out_of_order_data_and_revision_leave_collection_to_ai_and_preserve_old_values_until_confirmed(
        repo_ana_620, monkeypatch):
    orch = make(repo_ana_620, [
        action(COLLECT, num_dependentes=2, tem_dividas=False),
        message("Você confirma dois dependentes e ausência de dívidas?"),
        action(COLLECT, renda_mensal=7000, num_dependentes=3),
        message("Você confirma renda mensal de R$ 7.000,00 e três dependentes?"),
        message("Qual é seu tipo de emprego e quais são suas despesas fixas?"),
    ])
    s = logged_in(orch, agent="entrevista")
    FlowController.start_interview(s)
    s.flow.interview_answers = {"renda_mensal": 6000}
    before = csv(orch.repo)
    calls = forbid_financial_execution(monkeypatch, orch)
    orch.handle(s, "Tenho dois filhos e nenhuma dívida")
    assert s.flow.pending_interview_answers == {"num_dependentes": 2, "tem_dividas": False}
    assert s.flow.interview_answers == {"renda_mensal": 6000}
    assert_presentation_request(orch.client.requests[1])
    orch.handle(s, "Na verdade tenho três dependentes e minha renda mudou para sete mil reais")
    assert s.flow.pending_interview_answers == {"num_dependentes": 3, "tem_dividas": False,
                                               "renda_mensal": 7000}
    assert s.flow.interview_answers == {"renda_mensal": 6000}
    assert_presentation_request(orch.client.requests[3])
    assert orch.handle(s, "sim").startswith("Qual é seu tipo de emprego")
    assert s.flow.interview_answers == {"renda_mensal": 7000, "num_dependentes": 3,
                                        "tem_dividas": False}
    assert not s.flow.pending_interview_answers
    assert_presentation_request(orch.client.requests[4])
    assert csv(orch.repo) == before and calls == [COLLECT, COLLECT]
    check_protocol(orch, s)


@pytest.mark.parametrize("exhausted", [False, True])
def test_presentation_recovery_keeps_mode_receipts_confirmed_data_and_financial_consent(
        repo_ana_620, monkeypatch, caplog, exhausted):
    script = [failure("json_validate_failed")]
    script += [failure("json_validate_failed") if exhausted else message("Qual é seu tipo de emprego?")]
    orch = make(repo_ana_620, script)
    s = logged_in(orch, agent="entrevista")
    FlowController.start_interview(s)
    s.flow.pending_interview_answers = {"renda_mensal": 50000}
    s.credit_proposal = dict(valor=8000, modalidade="limite_total", novo_limite=8000,
                             confirmada=False)
    before, proposal = csv(orch.repo), copy.deepcopy(s.credit_proposal)
    calls = forbid_financial_execution(monkeypatch, orch)
    with caplog.at_level(logging.INFO, logger="banco_agil"):
        orch.handle(s, "confirmo")
    assert len(orch.client.requests) == 2
    for request in orch.client.requests:
        assert_presentation_request(request)
        assert "confirmacao_entrevista" in local_receipts(request["messages"])
    assert s.flow.interview_answers == {"renda_mensal": 50000}
    assert not s.flow.pending_interview_answers
    assert csv(orch.repo) == before and s.credit_proposal == proposal and not calls
    assert s.authenticated and s.active_agent == "entrevista" and not s.ended
    assert s.last_failure_cause == ("json_validate_failed" if exhausted else None)
    events = [json.loads(r.message) for r in caplog.records if r.message.startswith("{")]
    calls = [e for e in events if e.get("event") == "llm_call"]
    assert len(calls) == 2
    assert all(e["structured_strict"] is True and e["generation_phase"] == "message_only" for e in calls)
    check_protocol(orch, s)


@pytest.mark.parametrize("rejected", [action(COLLECT, tipo_emprego="formal"),
    action("solicitar_aumento_limite", novo_limite=8000)])
def test_recovery_cannot_turn_presentation_into_collection_or_a_financial_operation(
        repo_ana_620, monkeypatch, rejected):
    orch = make(repo_ana_620, [action(COLLECT, tipo_emprego="formal"),
                             failure("json_validate_failed"), rejected])
    s = logged_in(orch, agent="entrevista")
    FlowController.start_interview(s)
    s.flow.interview_answers = {"renda_mensal": 50000}
    s.credit_proposal = dict(valor=8000, modalidade="limite_total", novo_limite=8000,
                             confirmada=False)
    before, proposal = csv(orch.repo), copy.deepcopy(s.credit_proposal)
    calls = forbid_financial_execution(monkeypatch, orch)
    orch.handle(s, "formal")
    assert len(orch.client.requests) == 3
    assert calls == [COLLECT] and csv(orch.repo) == before and s.credit_proposal == proposal
    for request in orch.client.requests[1:]:
        assert_presentation_request(request)
    assert s.flow.interview_answers == {"renda_mensal": 50000}
    assert s.flow.pending_interview_answers == {"tipo_emprego": "formal"}
    assert s.last_failure_category in {"invalid_structured_response", "invalid_tool_response"}
    assert s.authenticated and s.active_agent == "entrevista" and not s.ended
    check_protocol(orch, s)


@pytest.mark.parametrize("mode,revision", [("structured", False), ("structured", True), ("native", False)])
def test_pending_presentation_is_narrow_and_revision_restores_action_exposure(mode, revision):
    s = Session(active_agent="entrevista", authenticated=True, cpf="local-only", llm_mode=mode)
    FlowController.start_interview(s)
    s.flow.interview_answers = {"renda_mensal": 6000}
    s.flow.pending_interview_answers = {"tipo_emprego": "formal"}
    s.flow.interview_revision_requested = revision
    before = copy.deepcopy(s.flow)
    names = {t["function"]["name"] for t in _tool_schemas(AGENTS["entrevista"], s)}
    if mode == "structured" and not revision:
        assert not names
    else:
        assert "encerrar_atendimento" in names
        assert (COLLECT in names) is revision
    assert s.flow == before


def test_completed_proposal_is_removed_only_from_wire_copy_with_receipt_and_user_text_preserved():
    proposed = envelope(action(COLLECT, renda_mensal=50000))
    result = receipt(COLLECT, {"confirmacao_pendente": True,
                               "respostas_propostas": {"renda_mensal": 50000}})
    customer = envelope(action(COLLECT, renda_mensal=999999), role="user")
    question = {"role": "assistant", "content": "Você confirma essa renda mensal?"}
    history = [customer, proposed, result, question]
    before = copy.deepcopy(history)
    assert without_completed_proposals(history) == [customer, result, question]
    assert history == before


@pytest.mark.parametrize("kind", ["mismatch", "error", "ok_false", "success_false", "unknown",
    "incomplete", "not_adjacent", "user", "pending_financial", "approved_unapplied", "nested_uncertain"])
def test_wire_projection_preserves_unsettled_or_unproven_action_groups(kind):
    name = "unknown_tool" if kind == "unknown" else COLLECT
    proposed = envelope(action(COLLECT, renda_mensal=50000))
    if kind == "unknown":
        value = json.loads(proposed["content"])
        value["resposta"]["acao"]["nome"] = name
        proposed["content"] = json.dumps(value)
    result = {"ok": True}
    if kind == "mismatch":
        name = "consultar_limite"
    elif kind == "error":
        result = {"erro": "Falha local"}
    elif kind == "ok_false":
        result = {"ok": False}
    elif kind == "success_false":
        result = {"sucesso": False}
    elif kind in {"pending_financial", "approved_unapplied", "nested_uncertain"}:
        name = "solicitar_aumento_limite"
        proposed = envelope(action(name, novo_limite=8000))
        result = ({"status_pedido": "pendente"} if kind == "pending_financial" else
                  {"status_pedido": "aprovado", "limite_aplicado": False})
        if kind == "nested_uncertain":
            name = "registrar_entrevista_e_calcular_score"
            value = json.loads(proposed["content"])
            value["resposta"]["acao"]["nome"] = name
            proposed["content"] = json.dumps(value)
            result = {"reanalise_automatica": {"executada": True, **result}}
    if kind == "user":
        proposed["role"] = "user"
    history = [proposed, receipt(name, result)]
    if kind == "incomplete":
        history.pop()
    elif kind == "not_adjacent":
        history.insert(1, {"role": "assistant", "content": "Confira os dados."})
    before = copy.deepcopy(history)
    assert without_completed_proposals(history) == before
    assert history == before


def test_completed_financial_receipt_survives_omission_of_assistant_proposal():
    proposed = envelope(action("solicitar_aumento_limite", novo_limite=8000))
    result = receipt("solicitar_aumento_limite", {
        "status_pedido": "aprovado", "limite_aplicado": True,
        "novo_limite_solicitado": 8000, "solicitacao_id": 17})
    history = [proposed, result]
    before = copy.deepcopy(history)
    assert without_completed_proposals(history) == [result]
    assert history == before
