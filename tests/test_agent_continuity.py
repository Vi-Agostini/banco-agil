"""Retomada e correção da entrevista com provedor simulado e CSVs isolados."""
import copy
import json

import pytest

from banco_agil.agents import AGENTS
from banco_agil.flow import Awaiting, FlowController
from banco_agil.repository import DataError
from banco_agil import tools
from fakes import failure
from orchestrator_fakes import logged_in
from structured_fakes import action, message, make, check_protocol
from support import snapshot_csv as csv


@pytest.mark.parametrize("mode", ["native", "structured"])
@pytest.mark.parametrize("confirmation", ["isso", "isso mesmo", "está certo", "já confirmei mil vezes", "sim",
                                          "confirmo, podemos continuar"])
def test_manual_confirmation_survives_failure_and_next_field_advances(repo_ana_620, mode, confirmation):
    orch = make(repo_ana_620, [failure("json_validate_failed"), failure("json_validate_failed"),
        action("registrar_respostas_entrevista", tipo_emprego="formal", tem_dividas=False),
        message("Confirma emprego formal e ausência de dívidas?"),
        message("Quantos dependentes você tem?")], mode)
    session = logged_in(orch, agent="entrevista")
    FlowController.start_interview(session)
    session.flow.pending_interview_answers = {"renda_mensal": 20000}
    session.credit_proposal = dict(valor=50000, modalidade="limite_total", novo_limite=50000, confirmada=False)
    proposal = copy.deepcopy(session.credit_proposal)
    before = csv(orch.repo)

    orch.handle(session, confirmation)

    assert session.active_agent == "entrevista" and session.flow.interview_started
    assert session.flow.interview_answers == {"renda_mensal": 20000}
    assert not session.flow.pending_interview_answers
    assert session.credit_proposal == proposal  # Confirmar dados da entrevista não autoriza crédito.
    for request in orch.client.requests:
        assert "confirmacao_entrevista" in str(request["messages"])
        assert any("20000" in m["content"] for m in request["messages"] if m["role"] == "system")

    orch.handle(session, "formal e não tenho dívidas")
    assert session.flow.pending_interview_answers == {"tipo_emprego": "formal", "tem_dividas": False}
    assert orch.handle(session, "isso").startswith("Quantos dependentes você tem?")
    assert session.flow.interview_answers == {"renda_mensal": 20000, "tipo_emprego": "formal", "tem_dividas": False}
    assert session.credit_proposal == proposal and csv(orch.repo) == before
    assert session.flow.attempts == session.flow.unclear_count == 0
    check_protocol(orch, session)


@pytest.mark.parametrize("mode", ["native", "structured"])
def test_employment_confirmation_with_continuation_does_not_remain_pending(repo_ana_620, mode):
    orch = make(repo_ana_620, [action("registrar_respostas_entrevista", tipo_emprego="formal"),
        message("Você confirma emprego formal?"), message("Quais são suas despesas fixas mensais?"),
        message("Qual é o valor das suas despesas fixas mensais?")], mode)
    s = logged_in(orch, agent="entrevista")
    FlowController.start_interview(s)
    s.flow.interview_answers = {"renda_mensal": 80000}
    s.credit_proposal = dict(valor=90000, modalidade="limite_total", novo_limite=90000, confirmada=False)
    proposal, before = copy.deepcopy(s.credit_proposal), csv(orch.repo)
    orch.handle(s, "meu emprego é formal")
    assert s.flow.pending_interview_answers == {"tipo_emprego": "formal"}
    for text in ("confirmo, podemos continuar", "Qual próxima pergunta"):
        reply = orch.handle(s, text)
        assert "despesas fixas" in reply and "Dados para confirmação: emprego" not in reply
        assert s.flow.interview_answers == {"renda_mensal": 80000, "tipo_emprego": "formal"}
        assert not s.flow.pending_interview_answers and not s.flow.interview_revision_requested
    assert s.authenticated and s.active_agent == "entrevista"
    assert csv(orch.repo) == before and s.credit_proposal == proposal
    check_protocol(orch, s)


@pytest.mark.parametrize("mode", ["native", "structured"])
def test_interview_continuation_confirmation_does_not_authorize_credit(repo_ana_620, mode):
    orch = make(repo_ana_620, [message("Podemos conferir a proposta antes de prosseguir.")], mode)
    s = logged_in(orch, agent="credito")
    s.credit_proposal = dict(valor=8000, modalidade="limite_total", novo_limite=8000, confirmada=False)
    before = csv(orch.repo)
    orch.handle(s, "confirmo, podemos continuar")
    assert s.credit_proposal["confirmada"] is False and csv(orch.repo) == before
    assert not orch.repo.list_requests()
    check_protocol(orch, s)


@pytest.mark.parametrize("mode", ["native", "structured"])
def test_proposal_survives_failure_before_summary_and_resumes_in_free_order(repo_ana_620, mode):
    orch = make(repo_ana_620, [action("registrar_respostas_entrevista", num_dependentes=2, despesas_fixas=3000),
        failure("tool_use_failed"), failure("tool_use_failed"),
        message("Qual é sua situação de emprego?")], mode)
    session = logged_in(orch, agent="entrevista")
    FlowController.start_interview(session)
    session.flow.interview_answers = {"renda_mensal": 20000}
    before = csv(orch.repo)
    response = orch.handle(session, "Tenho dois dependentes e gasto três mil")
    assert "3.000" in response and "2" in response
    assert session.flow.pending_interview_answers == {"num_dependentes": 2, "despesas_fixas": 3000}
    assert session.active_agent == "entrevista" and session.flow.interview_answers == {"renda_mensal": 20000}
    assert orch.handle(session, "está certo") == "Qual é sua situação de emprego?"
    assert session.flow.interview_answers == {"renda_mensal": 20000, "num_dependentes": 2, "despesas_fixas": 3000}
    assert not session.flow.pending_interview_answers and csv(orch.repo) == before
    check_protocol(orch, session)


@pytest.mark.parametrize("mode", ["native", "structured"])
@pytest.mark.parametrize("complete", [False, True])
def test_correction_of_confirmed_data_requires_new_confirmation_after_failure(repo_ana_620, mode, complete):
    orch = make(repo_ana_620, [action("registrar_respostas_entrevista", renda_mensal=7000),
        failure("json_validate_failed"), failure("json_validate_failed"), message("Podemos continuar a entrevista.")], mode)
    session = logged_in(orch, agent="entrevista")
    FlowController.start_interview(session)
    confirmed = {"renda_mensal": 6000, "tipo_emprego": "formal"}
    if complete:
        confirmed.update(despesas_fixas=2000, num_dependentes=1, tem_dividas=False)
    session.flow.interview_answers = dict(confirmed)
    before = csv(orch.repo)

    orch.handle(session, "Preciso corrigir minha renda: na verdade ganho sete mil")
    assert session.flow.interview_answers == confirmed
    assert session.flow.pending_interview_answers == {"renda_mensal": 7000}
    assert session.active_agent == "entrevista" and csv(orch.repo) == before
    blocked = orch.tool_executor.execute_from_llm(AGENTS["entrevista"], session,
        "registrar_entrevista_e_calcular_score", json.dumps({**confirmed, "renda_mensal": 7000}))
    assert "erro" in blocked

    orch.handle(session, "isso mesmo")
    assert session.flow.interview_answers == {**confirmed, "renda_mensal": 7000}
    assert not session.flow.pending_interview_answers and csv(orch.repo) == before
    check_protocol(orch, session)


@pytest.mark.parametrize("mode", ["native", "structured"])
def test_financial_consent_survives_ai_timeout_until_single_execution(repo_ana_620, mode):
    orch = make(repo_ana_620, [TimeoutError(), action("solicitar_aumento_limite", novo_limite=8000),
                            message("Seu limite aprovado é R$ 8.000,00.")], mode)
    session = logged_in(orch, agent="credito")
    session.credit_proposal = dict(valor=8000, modalidade="limite_total", novo_limite=8000, confirmada=False)
    session.flow.awaiting = Awaiting.LIMIT_AMOUNT
    before = csv(orch.repo)
    orch.handle(session, "sim")
    assert session.credit_proposal["confirmada"] and not session.credit_proposal.get("executada")
    assert session.flow.awaiting is Awaiting.LIMIT_AMOUNT
    assert csv(orch.repo) == before and session.active_agent == "credito"
    orch.handle(session, "continue o pedido autorizado")
    assert len(orch.repo.list_requests()) == 1 and session.credit_proposal["executada"]
    assert orch.repo.get_client(session.cpf)["limite_credito"] == 8000
    check_protocol(orch, session)


@pytest.mark.parametrize("method,after,amount", [("create_request", False, 8000), ("create_request", True, 8000),
    ("set_request_status", False, 8000), ("update_limit", False, 8000), ("update_limit", True, 8000),
    ("create_request", False, 20000), ("create_request", True, 20000), ("set_request_status", False, 20000)])
def test_uncertain_financial_write_resumes_without_duplicate(auth_ctx_ana_620, monkeypatch, method, after, amount):
    ctx = auth_ctx_ana_620
    original = getattr(ctx.repo, method)
    writes = []
    def interrupted(*args, **kwargs):
        writes.append(1)
        if after:
            original(*args, **kwargs)
        raise TimeoutError("simulated local storage interruption")
    monkeypatch.setattr(ctx.repo, method, interrupted)
    with pytest.raises(TimeoutError):
        tools.solicitar_aumento_limite(ctx, amount)
    assert ctx.session.credit_operation is not None
    monkeypatch.setattr(ctx.repo, method, original)

    result = tools.solicitar_aumento_limite(ctx, amount)
    assert "erro" not in result and result["id"] == 0
    assert len(ctx.repo.list_requests()) == 1 and len(writes) == 1
    assert ctx.repo.get_client(ctx.session.cpf)["limite_credito"] == (8000 if amount == 8000 else 5000)
    assert ctx.session.credit_operation is None and ctx.session.credit_proposal["executada"]


def test_increment_resumes_after_timeout_after_limit_applied(auth_ctx_ana_620, monkeypatch):
    ctx = auth_ctx_ana_620
    ctx.session.credit_proposal = dict(valor=3000, modalidade="acrescimo", limite_base=5000,
                                     novo_limite=8000, confirmada=True)
    original = ctx.repo.update_limit
    def interrupted(*args, **kwargs):
        original(*args, **kwargs)
        raise TimeoutError()
    monkeypatch.setattr(ctx.repo, "update_limit", interrupted)
    with pytest.raises(TimeoutError):
        tools.solicitar_aumento_limite(ctx, 8000)
    monkeypatch.setattr(ctx.repo, "update_limit", original)
    result = tools.solicitar_aumento_limite(ctx, 8000)
    assert result["limite_aplicado"] and len(ctx.repo.list_requests()) == 1


def test_uncertain_receipt_cannot_claim_another_sessions_row(auth_ctx_ana_620, monkeypatch):
    ctx = auth_ctx_ana_620
    original = ctx.repo._write
    monkeypatch.setattr(ctx.repo, "_write", lambda *a, **k: (_ for _ in ()).throw(DataError("disk")))
    with pytest.raises(DataError):
        tools.solicitar_aumento_limite(ctx, 20000)
    monkeypatch.setattr(ctx.repo, "_write", original)
    # Outra sessão gravou o mesmo valor na posição antes disponível.
    ctx.credit.request_increase(ctx.session.cpf, 20000)
    before = csv(ctx.repo)
    result = tools.solicitar_aumento_limite(ctx, 20000)
    assert result["codigo"] == "pedido_estado_incerto" and csv(ctx.repo) == before
    assert len(ctx.repo.list_requests()) == 1


def test_uncertain_operation_blocks_different_amount(auth_ctx_ana_620, monkeypatch):
    ctx = auth_ctx_ana_620
    original = ctx.repo.set_request_status
    monkeypatch.setattr(ctx.repo, "set_request_status", lambda *a, **k: (_ for _ in ()).throw(TimeoutError()))
    with pytest.raises(TimeoutError):
        tools.solicitar_aumento_limite(ctx, 8000)
    monkeypatch.setattr(ctx.repo, "set_request_status", original)
    result = tools.solicitar_aumento_limite(ctx, 9000)
    assert result["codigo"] == "pedido_estado_incerto"
    assert len(ctx.repo.list_requests()) == 1


@pytest.mark.parametrize("mode", ["native", "structured"])
def test_repeated_storage_failures_preserve_stage_and_authorization(repo_ana_620, monkeypatch, mode):
    orch = make(repo_ana_620, [action("solicitar_aumento_limite", novo_limite=8000)] * 4 +
                [action("solicitar_aumento_limite", novo_limite=8000), message("Seu limite aprovado é R$ 8.000,00.")], mode)
    session = logged_in(orch, agent="credito")
    session.credit_proposal = dict(valor=8000, modalidade="limite_total", novo_limite=8000, confirmada=True)
    session.flow.awaiting = Awaiting.LIMIT_AMOUNT
    session.flow.input_kind = "limit"
    session.flow.input_prompt = "Qual valor deseja solicitar?"
    session.flow.attempts = 1
    before = copy.deepcopy(session.flow)
    proposal = copy.deepcopy(session.credit_proposal)
    original = orch.repo.create_request
    monkeypatch.setattr(orch.repo, "create_request", lambda *a, **k: (_ for _ in ()).throw(TimeoutError()))
    for _ in range(4):
        orch.handle(session, "continue o pedido autorizado")
        assert session.active_agent == "credito" and session.flow == before
        assert session.credit_proposal == proposal and not orch.repo.list_requests()
    monkeypatch.setattr(orch.repo, "create_request", original)
    orch.handle(session, "retome o pedido autorizado")
    assert len(orch.repo.list_requests()) == 1
    assert orch.repo.get_client(session.cpf)["limite_credito"] == 8000
    check_protocol(orch, session)


@pytest.mark.parametrize("mode", ["native", "structured"])
@pytest.mark.parametrize("after", [False, True])
def test_automatic_reanalysis_timeout_never_creates_second_reanalysis(repo_ana_620, monkeypatch, mode, after):
    answers = dict(renda_mensal=6000, tipo_emprego="formal", despesas_fixas=2000,
                   num_dependentes=1, tem_dividas=False)
    orch = make(repo_ana_620, [action("registrar_entrevista_e_calcular_score", **answers),
                            TimeoutError(), message("Podemos consultar o status do seu pedido.")], mode)
    session = logged_in(orch, agent="credito", cpf="52998224725", nome="Bruno Lima")
    orch.tool_executor.execute(AGENTS["credito"], session, "solicitar_aumento_limite", '{"novo_limite":7000}')
    session.active_agent = "entrevista"
    FlowController.start_interview(session)
    session.flow.awaiting = Awaiting.NONE
    session.flow.interview_answers = dict(answers)
    original = orch.repo.create_request
    def interrupted(*args, **kwargs):
        if after:
            original(*args, **kwargs)
        raise TimeoutError()
    monkeypatch.setattr(orch.repo, "create_request", interrupted)
    orch.handle(session, "sim")
    assert session.active_agent == "credito" and session.automatic_reanalysis_source_id == 0
    assert orch.repo.get_client(session.cpf)["score"] == 570
    assert len(orch.repo.list_requests()) == (2 if after else 1)
    if after:
        assert session.last_request_id == 1 and session.last_request_status == "aprovado"
        assert orch.repo.get_client(session.cpf)["limite_credito"] == 7000
    else:
        assert "erro" in session.automatic_reanalysis_result
    monkeypatch.setattr(orch.repo, "create_request", original)
    orch.handle(session, "como ficou o pedido?")
    assert len(orch.repo.list_requests()) == (2 if after else 1)
    check_protocol(orch, session)
