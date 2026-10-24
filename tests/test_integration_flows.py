"""Fluxos completos de crédito, entrevista e reanálise com CSVs temporários.

O modelo simulado propõe ações e as ferramentas reais executam. Cada etapa
confere agente ativo, chamadas e estado persistido, sem avaliar linguagem natural.
"""
from banco_agil.flow import Awaiting
from banco_agil.statuses import STATUS_APROVADO, STATUS_REPROVADO
from fakes import assert_history_valid, reply

from orchestrator_fakes import logged_in, make

ANA = "11144477735"


def _snapshot(repo, session):
    """Estado observável por etapa: agente, pendências e CSVs."""
    return {
        "active_agent": session.active_agent,
        "awaiting": session.flow.awaiting,
        "ended": session.ended,
        "tools": [
            call["function"]["name"]
            for message in session.history
            for call in message.get("tool_calls", [])
        ],
        "requests": [
            (r["cpf_cliente"], r["novo_limite_solicitado"], r["status_pedido"])
            for r in repo.list_requests()
        ],
        "client": {k: repo.get_client(ANA)[k] for k in ("limite_credito", "score")},
    }


# Um pedido acima da faixa deve ser reprovado e persistido.
def test_cr09_deterministic_amount_flow_rejects_and_persists(repo_ana_620):
    """Caminho sem LLM (interceptors): valor acima do teto do score grava 1 reprovado."""
    orch = make(repo_ana_620, [reply("Que novo limite total deseja solicitar?")])
    s = logged_in(orch, agent="credito")
    before = _snapshot(repo_ana_620, s)

    ask = orch.handle(s, "quero aumentar meu limite")
    assert "?" in ask and s.flow.awaiting is Awaiting.LIMIT_AMOUNT

    rejected = orch.handle(s, "20000")  # acima do teto de 8000 para score 620

    assert "reprovado" in rejected and "entrevista" in rejected
    after = _snapshot(repo_ana_620, s)
    assert after["requests"] == [(ANA, "20000.00", STATUS_REPROVADO)]  # exatamente 1 pedido
    assert after["client"]["limite_credito"] == before["client"]["limite_credito"]  # intacto
    assert after["awaiting"] is Awaiting.INTERVIEW_OFFER
    assert len(orch.client.requests) == 1  # Pergunta livre; execução numérica determinística.
    orch.client.assert_exhausted()


def test_cr09_llm_tool_call_rejects_and_persists_identically(repo_ana_620):
    """Caminho com LLM: mesma persistência do caminho determinístico."""
    orch = make(repo_ana_620, [
        reply(calls=[("solicitar_aumento_limite", {"novo_limite": 20000})]),
    ])
    s = logged_in(orch, agent="credito")
    before = _snapshot(repo_ana_620, s)

    rejected = orch.handle(s, "quero aumento para 20000 reais")

    assert "não foi aprovado" in rejected and "entrevista" in rejected
    after = _snapshot(repo_ana_620, s)
    assert after["requests"] == [(ANA, "20000.00", STATUS_REPROVADO)]
    assert after["client"]["limite_credito"] == before["client"]["limite_credito"]
    assert after["awaiting"] is Awaiting.INTERVIEW_OFFER
    orch.client.assert_exhausted()


def test_cr09_llm_interpretation_failure_does_not_mutate_files(repo_ana_620):
    """Sem chamada de ferramenta, não há alteração de CSV; isso distingue falhas
    de interpretação das falhas de execução ou gravação."""
    orch = make(repo_ana_620, [
        reply("Desculpe, não consegui processar esse pedido agora."),
    ])
    s = logged_in(orch, agent="credito")
    before = _snapshot(repo_ana_620, s)

    orch.handle(s, "quero aumento para 20000 reais")

    assert _snapshot(repo_ana_620, s) == before  # nada mudou nos CSVs nem no fluxo
    orch.client.assert_exhausted()


# A entrevista após a reprovação atualiza o score e registra a reanálise.
def test_en01_interview_updates_score_and_reanalysis_persists(repo_ana_620):
    """Reprovação, entrevista, novo score e reanálise aprovada, conferindo o estado a cada etapa."""
    orch = make(repo_ana_620, [
        reply(calls=[("solicitar_aumento_limite", {"novo_limite": 20000})]),       # reprova
        reply("Qual informação sobre sua renda gostaria de fornecer primeiro?"),
        reply(calls=[("registrar_respostas_entrevista", {
            "renda_mensal": 20000, "tipo_emprego": "formal", "despesas_fixas": 1000,
            "num_dependentes": 0, "tem_dividas": False})]),
        reply("Entendi os cinco dados. Confirma esse resumo?"),
        reply(calls=[("registrar_entrevista_e_calcular_score", {
            "renda_mensal": 20000, "tipo_emprego": "formal", "despesas_fixas": 1000,
            "num_dependentes": 0, "tem_dividas": False})]),                        # score 1000
        reply("Seu pedido foi aprovado e o novo limite de R$ 20.000,00 já está ativo."),
    ])
    s = logged_in(orch, agent="credito")
    snapshots = [_snapshot(repo_ana_620, s)]

    # 1) Reprovação do pedido acima da faixa
    rejected = orch.handle(s, "quero aumento para 20000 reais")
    assert "não foi aprovado" in rejected and s.flow.awaiting is Awaiting.INTERVIEW_OFFER
    snapshots.append(_snapshot(repo_ana_620, s))

    # 2) A IA formula a pergunta inicial sem liberar o cálculo de score.
    assert "renda" in orch.handle(s, "sim").lower()
    assert s.active_agent == "entrevista"
    snapshots.append(_snapshot(repo_ana_620, s))

    # 3) O LLM propõe os dados; ainda não há persistência antes da confirmação.
    proposed = orch.handle(
        s, "renda 20000, formal, despesas 1000, zero dependentes, sem dívidas",
    )
    assert "confirma" in proposed.lower()
    assert s.active_agent == "entrevista"
    assert repo_ana_620.get_client(ANA)["score"] == 620

    # 4) Confirmação -> score e reanálise no mesmo turno do pipeline.
    approved = orch.handle(s, "sim")
    assert "aprovado" in approved and "20.000" in approved
    assert s.active_agent == "credito" and not s.ended
    snapshots.append(_snapshot(repo_ana_620, s))

    # Persistência verificada etapa a etapa
    assert snapshots[1]["requests"] == [(ANA, "20000.00", STATUS_REPROVADO)]
    assert snapshots[1]["client"] == {"limite_credito": 5000.0, "score": 620}
    assert snapshots[2]["active_agent"] == "entrevista"
    assert snapshots[2]["client"] == {"limite_credito": 5000.0, "score": 620}  # ainda não muda
    final = snapshots[3]
    assert "registrar_entrevista_e_calcular_score" in final["tools"]
    assert final["client"] == {"limite_credito": 20000.0, "score": 1000}  # score e limite novos
    assert final["requests"] == [
        (ANA, "20000.00", STATUS_REPROVADO),   # decisão original imutável
        (ANA, "20000.00", STATUS_APROVADO),    # reanálise após entrevista
    ]
    assert final["awaiting"] is Awaiting.NONE
    assert_history_valid(s.history)
    orch.client.assert_exhausted()


def test_en01_interview_declined_keeps_score_and_limit_unchanged(repo_ana_620):
    """Recusar a entrevista não altera score, limite nem gera nova solicitação."""
    orch = make(repo_ana_620, [
        reply(calls=[("solicitar_aumento_limite", {"novo_limite": 20000})]),
    ])
    s = logged_in(orch, agent="credito")
    before = _snapshot(repo_ana_620, s)

    orch.handle(s, "quero aumento para 20000 reais")
    rejected = _snapshot(repo_ana_620, s)  # estado após a reprovação
    declined = orch.handle(s, "não")  # interceptor de oferta, sem LLM

    assert "consulta de limite, câmbio" in declined and s.active_agent == "triagem"
    after = _snapshot(repo_ana_620, s)
    assert after["requests"] == rejected["requests"]  # só o reprovado original
    assert after["client"] == rejected["client"]  # score e limite intactos
    assert after["awaiting"] is Awaiting.NONE
    orch.client.assert_exhausted()
