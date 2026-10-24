"""Regressões da entrevista conduzida pelo LLM e protegida por confirmação local."""
from __future__ import annotations

import json
import copy

import pytest

from banco_agil.agents import AGENTS
from banco_agil.exchange import ExchangeService
from banco_agil.flow import Awaiting, FlowController
from banco_agil.orchestrator import Orchestrator
from banco_agil.statuses import STATUS_APROVADO, STATUS_REPROVADO

from fakes import FakeLLM, assert_history_valid, reply
from orchestrator_fakes import logged_in, make


BRUNO = "52998224725"
SCORE_ARGS = {
    "renda_mensal": 6000,
    "tipo_emprego": "formal",
    "despesas_fixas": 2000,
    "num_dependentes": 1,
    "tem_dividas": False,
}


class ErrorScriptLLM(FakeLLM):
    """Cliente simulado que intercala uma falha do provedor."""

    def _create(self, **kwargs):
        self.requests.append(copy.deepcopy(kwargs["messages"]))
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class SimulatedRateLimitError(Exception):
    status_code = 429
    headers = {"Retry-After": "1"}


def test_llm_collects_natural_combined_reply_as_a_proposal(repo):
    """A compreensão fica no LLM; código só valida e aguarda confirmação."""
    orch = make(repo, [
        reply(calls=[("registrar_respostas_entrevista", SCORE_ARGS)]),
        reply("Entendi: emprego formal, renda de R$ 6.000, despesas de R$ 2.000, um dependente e sem dívidas. Confirma?"),
    ])
    session = logged_in(orch, cpf=BRUNO, nome="Bruno Lima", agent="entrevista")
    FlowController.start_interview(session)
    before_score = repo.get_client(BRUNO)["score"]

    response = orch.handle(
        session,
        "Sou CLT. Ganho uns seis mil por mês, gasto dois mil, tenho um filho e nenhuma dívida.",
    )

    assert "Confirma" in response
    assert session.flow.interview_answers == {}
    assert session.flow.pending_interview_answers == SCORE_ARGS
    assert repo.get_client(BRUNO)["score"] == before_score
    assert len(orch.client.requests) == 2
    assert "Ganho uns seis mil" in str(orch.client.requests[0])
    orch.client.assert_exhausted()


def test_complementary_tool_calls_in_one_turn_accumulate_answers_and_keep_valid_history(repo):
    """O modelo pode dividir a interpretação de uma mensagem em duas chamadas."""
    first = {"renda_mensal": 6000, "tipo_emprego": "formal"}
    second = {"despesas_fixas": 2000, "num_dependentes": 1, "tem_dividas": False}
    orch = make(repo, [
        reply(calls=[
            ("registrar_respostas_entrevista", first),
            ("registrar_respostas_entrevista", second),
        ]),
        reply("Resumo: renda, emprego, despesas, dependente e dívidas. Confirma?"),
    ])
    session = logged_in(orch, cpf=BRUNO, nome="Bruno Lima", agent="entrevista")
    FlowController.start_interview(session)

    response = orch.handle(session, "Ganho seis mil, sou CLT, gasto dois mil, tenho um filho e não tenho dívidas.")

    assert "Confirma" in response
    assert session.flow.interview_answers == {}
    assert session.flow.pending_interview_answers == SCORE_ARGS
    assert_history_valid(session.history)
    orch.client.assert_exhausted()


def test_repeated_proposal_with_same_values_is_idempotent(repo):
    orch = make(repo, [])
    session = logged_in(orch, cpf=BRUNO, nome="Bruno Lima", agent="entrevista")
    FlowController.start_interview(session)
    proposal = {"renda_mensal": 6000, "tipo_emprego": "formal"}

    first = orch.tool_executor.execute(
        AGENTS["entrevista"], session, "registrar_respostas_entrevista", json.dumps(proposal),
    )
    repeated = orch.tool_executor.execute(
        AGENTS["entrevista"], session, "registrar_respostas_entrevista", json.dumps(proposal),
    )

    assert first["respostas_propostas"] == repeated["respostas_propostas"] == proposal
    assert session.flow.pending_interview_answers == proposal
    orch.client.assert_exhausted()


def test_conflicting_proposal_preserves_pending_and_confirmed_answers(repo):
    orch = make(repo, [])
    session = logged_in(orch, cpf=BRUNO, nome="Bruno Lima", agent="entrevista")
    FlowController.start_interview(session)
    session.flow.interview_answers = {"renda_mensal": 6000}
    session.flow.pending_interview_answers = {"tipo_emprego": "formal"}

    result = orch.tool_executor.execute(
        AGENTS["entrevista"],
        session,
        "registrar_respostas_entrevista",
        json.dumps({"renda_mensal": 7000, "tipo_emprego": "autônomo", "despesas_fixas": 2000}),
    )

    assert result["codigo"] == "dados_entrevista_conflitantes"
    assert set(result["campos_conflitantes"]) == {"renda_mensal", "tipo_emprego"}
    assert session.flow.interview_answers == {"renda_mensal": 6000}
    assert session.flow.pending_interview_answers == {"tipo_emprego": "formal"}
    orch.client.assert_exhausted()


def test_confirmed_value_is_not_reintroduced_as_pending(repo):
    orch = make(repo, [])
    session = logged_in(orch, cpf=BRUNO, nome="Bruno Lima", agent="entrevista")
    FlowController.start_interview(session)
    session.flow.interview_answers = {"renda_mensal": 6000}

    result = orch.tool_executor.execute(
        AGENTS["entrevista"],
        session,
        "registrar_respostas_entrevista",
        json.dumps({"renda_mensal": 6000, "tipo_emprego": "formal"}),
    )

    assert result["ok"] is True
    assert session.flow.interview_answers == {"renda_mensal": 6000}
    assert session.flow.pending_interview_answers == {"tipo_emprego": "formal"}
    orch.client.assert_exhausted()


def test_score_tool_blocks_invented_or_unconfirmed_values_without_writing(repo):
    orch = make(repo, [])
    session = logged_in(orch, cpf=BRUNO, nome="Bruno Lima", agent="entrevista")
    FlowController.start_interview(session)
    before_clients = repo.clients_path.read_bytes()
    before_requests = repo.requests_path.read_bytes()

    missing = orch.tool_executor.execute(
        AGENTS["entrevista"], session, "registrar_entrevista_e_calcular_score", json.dumps(SCORE_ARGS),
    )
    assert missing["codigo"] == "entrevista_incompleta"

    session.flow.pending_interview_answers = dict(SCORE_ARGS)
    unconfirmed = orch.tool_executor.execute(
        AGENTS["entrevista"], session, "registrar_entrevista_e_calcular_score", json.dumps(SCORE_ARGS),
    )
    assert unconfirmed["codigo"] == "entrevista_aguardando_confirmacao"
    assert repo.clients_path.read_bytes() == before_clients
    assert repo.requests_path.read_bytes() == before_requests
    orch.client.assert_exhausted()


def test_score_tool_rejects_arguments_that_differ_from_confirmed_answers_without_writing(repo):
    """Uma chamada do LLM não pode alterar respostas confirmadas nem persistir divergências."""
    orch = make(repo, [])
    session = logged_in(orch, cpf=BRUNO, nome="Bruno Lima", agent="entrevista")
    FlowController.start_interview(session)
    session.flow.interview_answers = dict(SCORE_ARGS)
    before_clients = repo.clients_path.read_bytes()
    before_requests = repo.requests_path.read_bytes()

    result = orch.tool_executor.execute(
        AGENTS["entrevista"],
        session,
        "registrar_entrevista_e_calcular_score",
        json.dumps({**SCORE_ARGS, "despesas_fixas": 0}),
    )

    assert result["codigo"] == "dados_entrevista_nao_conferem"
    assert session.flow.interview_answers == SCORE_ARGS
    assert repo.get_client(BRUNO)["score"] == 350
    assert repo.get_client(BRUNO)["limite_credito"] == 2000
    assert repo.clients_path.read_bytes() == before_clients
    assert repo.requests_path.read_bytes() == before_requests
    orch.client.assert_exhausted()


def test_invalid_llm_proposal_is_not_saved_or_confirmable(repo):
    orch = make(repo, [])
    session = logged_in(orch, cpf=BRUNO, nome="Bruno Lima", agent="entrevista")
    FlowController.start_interview(session)

    result = orch.tool_executor.execute(
        AGENTS["entrevista"], session, "registrar_respostas_entrevista", json.dumps({"renda_mensal": -1}),
    )

    assert result["codigo"] == "valor_negativo"
    assert session.flow.interview_answers == {}
    assert session.flow.pending_interview_answers == {}
    orch.client.assert_exhausted()


def test_rejected_confirmation_preserves_confirmed_answers_and_discards_proposal(repo):
    orch = make(repo, [reply("Certo. Qual valor devo corrigir?")])
    session = logged_in(orch, cpf=BRUNO, nome="Bruno Lima", agent="entrevista")
    FlowController.start_interview(session)
    session.flow.interview_answers = {"renda_mensal": 6000}
    session.flow.pending_interview_answers = {"tipo_emprego": "formal", "tem_dividas": False}

    assert "Qual valor" in orch.handle(session, "não")
    assert session.flow.interview_answers == {"renda_mensal": 6000}
    assert session.flow.pending_interview_answers == {}
    orch.client.assert_exhausted()


@pytest.mark.parametrize("confirmation", [
    "sim", "OK!", "Okay.", "positivo", "CONFIRMO", "confirmado!", "correto",
    "Está correto.", "com certeza", "Pode prosseguir!",
])
def test_explicit_positive_confirmation_confirms_pending_interview_proposal_without_writing(
    repo, confirmation,
):
    orch = make(repo, [reply("Qual é o valor das suas despesas fixas?")])
    session = logged_in(orch, cpf=BRUNO, nome="Bruno Lima", agent="entrevista")
    FlowController.start_interview(session)
    session.flow.interview_answers = {"tipo_emprego": "formal"}
    session.flow.pending_interview_answers = {"renda_mensal": 6000}
    before_clients = repo.clients_path.read_bytes()
    before_requests = repo.requests_path.read_bytes()

    orch.handle(session, confirmation)

    assert session.flow.interview_answers == {"tipo_emprego": "formal", "renda_mensal": 6000}
    assert session.flow.pending_interview_answers == {}
    assert not session.flow.interview_revision_requested
    assert repo.get_client(BRUNO)["score"] == 350
    assert repo.get_client(BRUNO)["limite_credito"] == 2000
    assert repo.clients_path.read_bytes() == before_clients
    assert repo.requests_path.read_bytes() == before_requests
    orch.client.assert_exhausted()


@pytest.mark.parametrize(("confirmation", "expected_pending"), [
    ("sim, mas minha renda é R$ 3.000", SCORE_ARGS),
    ("ok, só muda meus gastos", SCORE_ARGS),
    ("não", {}),
    ("negativo", {}),
    ("talvez", SCORE_ARGS),
    ("acho que sim", SCORE_ARGS),
])
def test_nonexplicit_confirmation_never_persists_financial_data(
    repo, confirmation, expected_pending,
):
    orch = make(repo, [reply("Vamos revisar as informações antes de continuar.")])
    session = logged_in(orch, cpf=BRUNO, nome="Bruno Lima", agent="entrevista")
    FlowController.start_interview(session)
    session.flow.pending_interview_answers = dict(SCORE_ARGS)
    before_clients = repo.clients_path.read_bytes()
    before_requests = repo.requests_path.read_bytes()

    orch.handle(session, confirmation)

    assert session.flow.interview_answers == {}
    assert session.flow.pending_interview_answers == expected_pending
    assert repo.get_client(BRUNO)["score"] == 350
    assert repo.get_client(BRUNO)["limite_credito"] == 2000
    assert repo.clients_path.read_bytes() == before_clients
    assert repo.requests_path.read_bytes() == before_requests
    orch.client.assert_exhausted()


def test_ambiguous_affirmative_revises_pending_proposal_before_new_confirmation(repo):
    orch = make(repo, [
        reply(calls=[("registrar_respostas_entrevista", {"renda_mensal": 3000})]),
        reply("Resumo atualizado: renda mensal de R$ 3.000,00. Confirma?"),
        reply("Qual é o valor das suas despesas fixas?"),
    ])
    session = logged_in(orch, cpf=BRUNO, nome="Bruno Lima", agent="entrevista")
    FlowController.start_interview(session)
    session.flow.interview_answers = {"tipo_emprego": "formal"}
    session.flow.pending_interview_answers = {"renda_mensal": 6000}

    response = orch.handle(session, "sim, mas minha renda é R$ 3.000")

    assert "3.000" in response
    assert session.flow.interview_answers == {"tipo_emprego": "formal"}
    assert session.flow.pending_interview_answers == {"renda_mensal": 3000}
    assert not session.flow.interview_revision_requested

    orch.handle(session, "sim")

    assert session.flow.interview_answers == {"tipo_emprego": "formal", "renda_mensal": 3000}
    assert session.flow.pending_interview_answers == {}
    orch.client.assert_exhausted()


def test_ambiguous_income_correction_keeps_full_proposal_pending_before_score(repo):
    """A correção não confirma a proposta anterior nem libera escrita financeira."""
    corrected = {**SCORE_ARGS, "renda_mensal": 3000}
    orch = make(repo, [
        reply(calls=[("registrar_respostas_entrevista", {"renda_mensal": 3000})]),
        reply("Resumo atualizado: renda mensal de R$ 3.000,00. Confirma?"),
    ])
    session = logged_in(orch, cpf=BRUNO, nome="Bruno Lima", agent="entrevista")
    FlowController.start_interview(session)
    session.flow.pending_interview_answers = dict(SCORE_ARGS)
    before_clients = repo.clients_path.read_bytes()
    before_requests = repo.requests_path.read_bytes()

    response = orch.handle(session, "sim, mas minha renda é R$ 3.000")

    assert "3.000" in response
    assert session.flow.interview_answers == {}
    assert session.flow.pending_interview_answers == corrected
    blocked = orch.tool_executor.execute(
        AGENTS["entrevista"], session, "registrar_entrevista_e_calcular_score", json.dumps(corrected),
    )
    assert blocked["codigo"] == "entrevista_aguardando_confirmacao"
    assert repo.get_client(BRUNO)["score"] == 350
    assert repo.get_client(BRUNO)["limite_credito"] == 2000
    assert repo.clients_path.read_bytes() == before_clients
    assert repo.requests_path.read_bytes() == before_requests
    orch.client.assert_exhausted()


def test_ambiguous_interview_offer_keeps_attached_information_for_the_agents(repo):
    orch = make(repo, [
        reply(calls=[("transferir_para_agente", {"destino": "entrevista"})]),
        reply(calls=[("registrar_respostas_entrevista", {"renda_mensal": 3000})]),
        reply("Resumo: renda mensal de R$ 3.000,00. Confirma?"),
    ])
    session = logged_in(orch, cpf=BRUNO, nome="Bruno Lima", agent="credito")
    session.flow.awaiting = Awaiting.INTERVIEW_OFFER

    response = orch.handle(session, "sim, minha renda é R$ 3.000")

    assert "Confirma" in response
    assert session.active_agent == "entrevista"
    assert session.flow.pending_interview_answers == {"renda_mensal": 3000}
    assert "minha renda" in str(orch.client.requests[1]).lower()
    orch.client.assert_exhausted()


def test_confirmed_answers_calculate_score_and_reanalyze_once(repo):
    orch = make(repo, [
        reply('Como você descreveria sua renda mensal hoje?'),
        reply(calls=[("registrar_respostas_entrevista", SCORE_ARGS)]),
        reply("Entendi os cinco dados. Confirma esse resumo?"),
        reply(calls=[("registrar_entrevista_e_calcular_score", SCORE_ARGS)]),
        reply("Seu pedido foi aprovado e o novo limite de R$ 7.000,00 já está ativo."),
        reply("Seu limite permanece em R$ 7.000,00."),
    ])
    session = logged_in(orch, cpf=BRUNO, nome="Bruno Lima", agent="credito")

    rejected = orch.tool_executor.execute(
        AGENTS["credito"], session, "solicitar_aumento_limite", json.dumps({"novo_limite": 7000}),
    )
    assert rejected["status_pedido"] == STATUS_REPROVADO
    assert "renda" in orch.handle(session, "sim").lower()

    proposed = orch.handle(
        session,
        "Sou CLT. Ganho uns seis mil por mês, gasto dois mil, tenho um filho e nenhuma dívida.",
    )
    assert "confirma" in proposed.lower()
    assert session.flow.interview_answers == {}
    assert session.flow.pending_interview_answers == SCORE_ARGS
    assert repo.get_client(BRUNO)["score"] == 350

    final = orch.handle(session, "sim")

    assert "aprovado" in final
    assert repo.get_client(BRUNO)["score"] == 570
    assert repo.get_client(BRUNO)["limite_credito"] == 7000
    assert [row["status_pedido"] for row in repo.list_requests()] == [STATUS_REPROVADO, STATUS_APROVADO]
    assert session.active_agent == "credito"
    assert session.flow.interview_answers == {}
    assert session.flow.pending_interview_answers == {}
    assert session.automatic_reanalysis_source_id == 0
    assert session.last_request_id == 1
    assert len(orch.client.requests) == 5  # A IA também formula a pergunta inicial.

    orch.handle(session, "qual é meu limite agora?")

    assert [row["status_pedido"] for row in repo.list_requests()] == [STATUS_REPROVADO, STATUS_APROVADO]
    orch.client.assert_exhausted()


def test_score_without_previous_rejected_request_does_not_reanalyze(repo):
    orch = make(repo, [])
    session = logged_in(orch, cpf=BRUNO, nome="Bruno Lima", agent="entrevista")
    FlowController.start_interview(session)
    session.flow.interview_answers = dict(SCORE_ARGS)

    result = orch.tool_executor.execute(
        AGENTS["entrevista"], session, "registrar_entrevista_e_calcular_score", json.dumps(SCORE_ARGS),
    )

    assert result["novo_score"] == 570
    assert result["reanalise_automatica"] == {"executada": False, "codigo": "reanalise_sem_origem_valida"}
    assert repo.get_client(BRUNO)["score"] == 570
    assert repo.list_requests() == []
    orch.client.assert_exhausted()


def test_rate_limit_after_automatic_reanalysis_does_not_duplicate_request(repo, monkeypatch):
    monkeypatch.setattr("banco_agil.llm.time.sleep", lambda _: None)
    client = ErrorScriptLLM([
        reply(calls=[("registrar_entrevista_e_calcular_score", SCORE_ARGS)]),
        SimulatedRateLimitError(),
        SimulatedRateLimitError(),
        reply("Seu pedido reanalisado foi aprovado."),
    ])
    orch = Orchestrator(repo, ExchangeService(), client, "fake")
    session = logged_in(orch, cpf=BRUNO, nome="Bruno Lima", agent="credito")
    rejected = orch.tool_executor.execute(
        AGENTS["credito"], session, "solicitar_aumento_limite", json.dumps({"novo_limite": 7000}),
    )
    assert rejected["status_pedido"] == STATUS_REPROVADO
    session.active_agent = "entrevista"
    FlowController.start_interview(session)
    session.flow.awaiting = Awaiting.NONE
    session.flow.interview_answers = dict(SCORE_ARGS)

    response = orch.handle(session, "sim")

    assert "aprovado" in response.lower()
    assert session.last_failure_category == "http_429"
    assert [row["status_pedido"] for row in repo.list_requests()] == [STATUS_REPROVADO, STATUS_APROVADO]

    orch.handle(session, "obrigado")

    assert [row["status_pedido"] for row in repo.list_requests()] == [STATUS_REPROVADO, STATUS_APROVADO]
    assert len(client.requests) == 4
    client.assert_exhausted()


def test_new_interview_cancellation_and_end_clear_confirmed_and_pending_answers(repo):
    orch = make(repo, [])
    session = logged_in(orch, cpf=BRUNO, nome="Bruno Lima", agent="entrevista")
    session.flow.interview_answers = dict(SCORE_ARGS)
    session.flow.pending_interview_answers = {"renda_mensal": 8000}

    orch.handle(session, "encerrar a entrevista")

    assert session.active_agent == "credito"
    assert session.flow.interview_answers == {}
    assert session.flow.pending_interview_answers == {}

    session.active_agent = "entrevista"
    FlowController.start_interview(session)
    session.flow.interview_answers = dict(SCORE_ARGS)
    session.flow.pending_interview_answers = {"renda_mensal": 8000}
    orch.handle(session, "encerrar o atendimento")

    assert session.ended
    assert session.flow.interview_answers == {}
    assert session.flow.pending_interview_answers == {}
    orch.client.assert_exhausted()
