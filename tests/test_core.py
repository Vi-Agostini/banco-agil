import pytest

from banco_agil import tools
from banco_agil.scoring import calculate_score
from banco_agil.statuses import STATUS_APROVADO, STATUS_REPROVADO


def test_score_formula():
    # 5000/(2000+1)*30 = 74.96 + 300 + 100 + 100
    assert calculate_score(5000, 2000, "formal", 0, False) == 575
    assert calculate_score(100000, 0, "formal", 0, False) == 1000  # teto
    assert calculate_score(0, 0, "desempregado", 3, True) == 0  # piso (-70 -> 0)


def test_score_invalid_employment():
    with pytest.raises(ValueError):
        calculate_score(1000, 500, "astronauta", 0, False)


def test_auth_success_with_br_date(ctx):
    r = tools.autenticar_cliente(ctx, "111.444.777-35", "12/05/1990")
    assert r["sucesso"] and ctx.session.authenticated


def test_auth_three_failures_end_session(ctx):
    for i in range(2):
        r = tools.autenticar_cliente(ctx, "11144477735", "01/01/2000")
        assert not r["sucesso"] and r["tentativas_restantes"] == 2 - i
    r = tools.autenticar_cliente(ctx, "11144477735", "01/01/2000")
    assert r["encerrado"] and ctx.session.ended


def test_auth_invalid_format_counts_as_attempt(ctx):
    result = tools.autenticar_cliente(ctx, "123", "12/05/1990")

    assert result["motivo"] == "dados_nao_conferem"
    assert ctx.session.auth_failures == 1


def test_limit_request_approved(auth_ctx_ana_620):
    r = tools.solicitar_aumento_limite(auth_ctx_ana_620, 8000)  # score 620 -> max 8000
    assert r["status_pedido"] == STATUS_APROVADO
    assert auth_ctx_ana_620.repo.get_client("11144477735")["limite_credito"] == 8000
    rows = auth_ctx_ana_620.repo._read(auth_ctx_ana_620.repo.requests_path)
    assert rows[-1]["status_pedido"] == STATUS_APROVADO


def test_limit_request_rejected_then_interview_flow(auth_ctx_ana_620):
    r = tools.solicitar_aumento_limite(auth_ctx_ana_620, 20000)
    assert r["status_pedido"] == STATUS_REPROVADO
    auth_ctx_ana_620.session.active_agent = "entrevista"
    auth_ctx_ana_620.session.flow.interview_answers = {
        "renda_mensal": 20000, "tipo_emprego": "formal", "despesas_fixas": 1000,
        "num_dependentes": 0, "tem_dividas": False,
    }
    out = tools.registrar_entrevista_e_calcular_score(auth_ctx_ana_620, 20000, "formal", 1000, 0, False)
    assert out["novo_score"] == 1000 and auth_ctx_ana_620.session.active_agent == "credito"
    assert auth_ctx_ana_620.repo.get_client("11144477735")["score"] == 1000
    assert out["reanalise_automatica"]["status_pedido"] == STATUS_APROVADO
    assert auth_ctx_ana_620.repo.get_client("11144477735")["limite_credito"] == 20000
    assert [row["status_pedido"] for row in auth_ctx_ana_620.repo.list_requests()] == [
        STATUS_REPROVADO, STATUS_APROVADO,
    ]


def test_limit_must_increase(auth_ctx):
    assert "erro" in tools.solicitar_aumento_limite(auth_ctx, 100)


def test_transfer_requires_auth_and_respects_scope(ctx):
    assert "erro" in tools.transferir_para_agente(ctx, "credito")  # sem autenticação
    ctx.session.authenticated, ctx.session.cpf = True, "11144477735"
    assert "erro" in tools.transferir_para_agente(ctx, "entrevista")  # triagem não vai direto à entrevista
    assert tools.transferir_para_agente(ctx, "credito")["ok"]
    assert tools.transferir_para_agente(ctx, "entrevista")["ok"]  # crédito pode encaminhar à entrevista
