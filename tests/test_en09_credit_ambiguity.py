"""Um valor de crédito ambíguo precisa de esclarecimento, sem criar ciclos.

O modelo é simulado nos dois protocolos. As ferramentas reais criam as
propostas no repositório isolado, mantendo as validações da aplicação.
"""
import copy
from types import SimpleNamespace

import pytest

from banco_agil.intents import normalize_text
from fakes import assert_history_valid, failure
from orchestrator_fakes import logged_in
from structured_fakes import action, completion, make, message
from support import snapshot_csv as csv


CLARIFY = (
    "O valor informado foi R$ 7.000,00. Você deseja que esse seja o limite total "
    "ou um acréscimo ao limite atual?"
)


class TurnClient:
    """Decisões simuladas finitas, seguidas do mesmo pedido de esclarecimento.

    Permite a recuperação limitada sem exigir outra geração quando
    o estado já permite apresentar a pergunta necessária.
    """

    def __init__(self, script, tail=None):
        self.script = list(script)
        self.tail = message(CLARIFY) if tail is None else tail
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **request):
        self.requests.append(copy.deepcopy(request))
        value = self.script.pop(0) if self.script else self.tail
        if isinstance(value, Exception):
            raise value
        return completion(value, mode="structured" if "response_format" in request else "native")


def turn(orch, session, text, script, *, tail=None):
    client = TurnClient(script, tail)
    orch.client = client
    answer = orch.handle(session, text)
    # Todo turno deve terminar, inclusive quando há recuperação da decisão.
    assert len(client.requests) <= orch.max_steps + 2
    assert_history_valid(session.history)
    if orch.llm_mode == "structured":
        assert all(r["response_format"]["json_schema"]["strict"] is True for r in client.requests)
        assert not any("tools" in r for r in client.requests)
    return answer


def start_ambiguous(repo, mode, *, cpf="11144477735", name="Ana Souza"):
    orch = make(repo, [], mode)
    session = logged_in(orch, cpf=cpf, nome=name, agent="credito")
    before = csv(repo)
    answer = turn(orch, session, "quero limite de 7000", [
        action("registrar_intencao_credito", valor=7000, modalidade="indefinida"),
        message(CLARIFY),
    ])
    assert_ambiguous(orch, session, answer, before)
    return orch, session, before


def assert_ambiguous(orch, session, answer, before):
    wording = normalize_text(answer)
    assert "limite total" in wording and "acrescimo" in wording
    assert "?" in answer, "A proposta ambígua precisa de uma pergunta acionável."
    assert "confirmo" not in wording, "Ainda não existe proposta completa para autorizar."
    assert session.credit_proposal["modalidade"] == "indefinida"
    assert not session.credit_proposal["confirmada"]
    assert not session.credit_proposal.get("executada")
    assert session.credit_proposal["valor"] == 7000
    assert session.authenticated and session.cpf and session.active_agent == "credito"
    assert not session.ended and csv(orch.repo) == before
    assert not orch.repo.list_requests()


@pytest.mark.parametrize("mode", ["native", "structured"])
def test_en09_model_cannot_present_undefined_amount_as_authorizable_total(repo_ana_620, mode):
    orch = make(repo_ana_620, [], mode)
    session = logged_in(orch, agent="credito")
    before = csv(orch.repo)

    answer = turn(orch, session, "quero limite de 7000", [
        action("registrar_intencao_credito", valor=7000, modalidade="indefinida"),
        message('Você deseja um limite total de R$ 7.000,00. Responda "confirmo" para autorizar.'),
    ])

    assert_ambiguous(orch, session, answer, before)


@pytest.mark.parametrize("mode", ["native", "structured"])
@pytest.mark.parametrize("cycle", ["handoff", "recollection"])
def test_en09_repeated_confirmations_cannot_restart_same_ambiguous_loop(repo_ana_620, mode, cycle):
    orch, session, before = start_ambiguous(repo_ana_620, mode)
    for _ in range(3):
        decisions = ([action("transferir_para_agente", destino="triagem"),
                      action("transferir_para_agente", destino="credito"),
                      action("transferir_para_agente", destino="triagem")]
                     if cycle == "handoff" else
                     [action("registrar_intencao_credito", valor=7000, modalidade="indefinida"),
                      action("registrar_intencao_credito", valor=7000, modalidade="indefinida")])

        answer = turn(orch, session, "confirmo", decisions)

        assert_ambiguous(orch, session, answer, before)
        assert session.credit_confirmation_turn != session.turn
        assert "transferir_para_agente" not in session.tool_facts
        assert session.tool_facts["registrar_intencao_credito"]["turn"] == 1
        assert session.flow.attempts == 0
        assert session.flow.unclear_count == 0


@pytest.mark.parametrize("mode", ["native", "structured"])
@pytest.mark.parametrize("plain", [
    "Dados da proposta: valor solicitado: R$ 7.000,00.",
    "Obrigado pela confirmação. Podemos continuar.",
    "",
])
def test_en09_summary_or_acknowledgement_alone_does_not_end_ambiguity_turn(repo_ana_620, mode, plain):
    orch, session, before = start_ambiguous(repo_ana_620, mode)

    for _ in range(3):
        answer = turn(orch, session, "confirmo", [message(plain)], tail=message(plain))

        assert_ambiguous(orch, session, answer, before)


@pytest.mark.parametrize("mode", ["native", "structured"])
def test_en09_ai_resolves_total_then_confirmation_records_one_real_score_decision(repo_ana_620, mode):
    orch, session, before = start_ambiguous(
        repo_ana_620, mode, cpf="52998224725", name="Bruno Lima",
    )
    proposed = turn(orch, session, "O limite total desejado é 7000", [
        action("registrar_intencao_credito", valor=7000, modalidade="limite_total"),
        message("O novo limite total solicitado é R$ 7.000,00. Confira o valor para autorizar."),
    ])
    assert "me envie **confirmo** para continuar" in proposed
    assert session.credit_proposal["modalidade"] == "limite_total"
    assert session.credit_proposal["novo_limite"] == 7000
    assert not session.credit_proposal["confirmada"] and csv(orch.repo) == before

    answer = turn(orch, session, "confirmo", [
        action("solicitar_aumento_limite", novo_limite=7000),
    ])

    requests = orch.repo.list_requests()
    assert len(requests) == 1 and requests[0]["status_pedido"] == "reprovado"
    assert float(requests[0]["novo_limite_solicitado"]) == 7000
    assert orch.repo.get_client(session.cpf)["limite_credito"] == 2000
    assert session.credit_proposal["executada"] and "não foi aprovado" in answer
    assert "entrevista" in answer.casefold() and not session.ended
    assert csv(orch.repo)["score_limite.csv"] == before["score_limite.csv"]


@pytest.mark.parametrize("mode", ["native", "structured"])
def test_en09_ai_resolves_increment_and_only_confirmed_computed_total_is_applied(repo_ana_620, mode):
    orch, session, before = start_ambiguous(repo_ana_620, mode)
    proposed = turn(orch, session, "É um acréscimo de 2000", [
        action("registrar_intencao_credito", valor=2000, modalidade="acrescimo"),
        message("O acréscimo de R$ 2.000,00 resulta em novo limite de R$ 7.000,00. Confira os valores."),
    ])
    assert "me envie **confirmo** para continuar" in proposed
    assert session.credit_proposal["novo_limite"] == 7000
    assert session.credit_proposal["limite_base"] == 5000
    assert not session.credit_proposal["confirmada"] and csv(orch.repo) == before

    answer = turn(orch, session, "confirmo", [
        action("solicitar_aumento_limite", novo_limite=7000),
        message("Seu pedido foi aprovado e o novo limite de R$ 7.000,00 já está ativo."),
    ])

    requests = orch.repo.list_requests()
    assert len(requests) == 1 and requests[0]["status_pedido"] == "aprovado"
    assert float(requests[0]["novo_limite_solicitado"]) == 7000
    assert orch.repo.get_client(session.cpf)["limite_credito"] == 7000
    assert session.credit_proposal["executada"] and "aprovado" in answer
    assert csv(orch.repo)["score_limite.csv"] == before["score_limite.csv"]


class RateLimited(Exception):
    status_code = 429
    headers = {"Retry-After": "2"}
    body = {"error": {"code": "rate_limit_exceeded", "type": "tokens"}}


@pytest.mark.parametrize("mode", ["native", "structured"])
@pytest.mark.parametrize("kind", ["400", "429", "timeout"])
def test_en09_provider_failure_preserves_ambiguity_and_actionable_resume(repo_ana_620, mode, kind, monkeypatch):
    from fakes import Clock
    clock = Clock()
    monkeypatch.setattr("banco_agil.llm.time.sleep", clock.sleep)
    orch, session, before = start_ambiguous(repo_ana_620, mode)
    error = {"400": failure("json_validate_failed"), "429": RateLimited(),
             "timeout": TimeoutError()}[kind]

    answer = turn(orch, session, "confirmo", [error], tail=error)

    assert_ambiguous(orch, session, answer, before)
    assert all(word not in answer.casefold() for word in ("http", "json", "schema", "429"))
    assert clock.waits == ([20.0] if kind == "429" else [])


@pytest.mark.parametrize("mode", ["native", "structured"])
def test_en09_financial_tool_before_resolving_ambiguity_is_blocked(repo_ana_620, mode):
    orch, session, before = start_ambiguous(repo_ana_620, mode)

    answer = turn(orch, session, "confirmo", [
        action("solicitar_aumento_limite", novo_limite=7000),
    ])

    assert_ambiguous(orch, session, answer, before)
    assert "solicitar_aumento_limite" not in session.tool_facts


@pytest.mark.parametrize("mode", ["native", "structured"])
def test_en09_valid_clarification_remains_ai_written_once(repo_ana_620, mode):
    orch = make(repo_ana_620, [], mode)
    session = logged_in(orch, agent="credito")

    answer = turn(orch, session, "quero limite de 7000", [
        action("registrar_intencao_credito", valor=7000, modalidade="indefinida"),
        message(CLARIFY),
    ])

    assert answer == CLARIFY.replace(". Você", ".\n\nVocê")
    assert answer.count("?") == 1 and len(orch.client.requests) == 2
    assert not orch.repo.list_requests()


@pytest.mark.parametrize("mode", ["native", "structured"])
def test_en09_consultation_is_reported_without_forcing_old_ambiguity(repo_ana_620, mode):
    orch, session, before = start_ambiguous(repo_ana_620, mode)

    answer = turn(orch, session, "Antes disso, qual meu limite atual?", [
        action("consultar_limite"), message("Seu limite de crédito atual é R$ 5.000,00."),
    ])

    assert answer == "Seu limite de crédito atual é R$ 5.000,00."
    assert session.credit_proposal["modalidade"] == "indefinida"
    assert csv(orch.repo) == before and not session.ended


@pytest.mark.parametrize("mode", ["native", "structured"])
def test_en09_new_topic_stays_with_ai_and_is_not_replaced_by_old_proposal(repo_ana_620, mode):
    orch, session, before = start_ambiguous(repo_ana_620, mode)
    text = "Posso orientar sobre os serviços disponíveis neste atendimento."

    answer = turn(orch, session,
                  "Deixa esse pedido para lá. Me explique como posso falar com um atendente.", [message(text)])

    assert answer == text and session.credit_proposal["modalidade"] == "indefinida"
    assert csv(orch.repo) == before and not session.ended


@pytest.mark.parametrize("mode", ["native", "structured"])
def test_en09_ambiguity_catalog_expires_for_topic_change_and_explicit_close(repo_ana_620, mode):
    orch, session, before = start_ambiguous(repo_ana_620, mode)
    turn(orch, session, "confirmo", [message(CLARIFY)])
    proposal = copy.deepcopy(session.credit_proposal)

    answer = turn(orch, session, "Agora quero consultar câmbio", [
        action("transferir_para_agente", destino="cambio"),
        message("Qual moeda você deseja consultar?"),
    ])

    assert answer == "Qual moeda você deseja consultar?" and session.active_agent == "cambio"
    assert session.credit_proposal == proposal and session.credit_clarification_turn < session.turn
    assert session.authenticated and csv(orch.repo) == before
    turn(orch, session, "encerrar atendimento", [])
    assert session.ended and session.credit_proposal is None and csv(orch.repo) == before
