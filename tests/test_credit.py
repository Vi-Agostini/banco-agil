
import pytest

from banco_agil import tools
from banco_agil.messages import limit_request_reply
from banco_agil.repository import DataError
from banco_agil.statuses import STATUS_APROVADO, STATUS_PENDENTE, STATUS_REPROVADO

ANA = "11144477735"  # score 620 -> teto R$ 8.000; limite atual 5.000


def limit(repo, cpf=ANA):
    return repo.get_client(cpf)["limite_credito"]


def rows(repo):
    return repo.list_requests()


def test_consulta_limite(auth_ctx_ana_620):
    assert tools.consultar_limite(auth_ctx_ana_620) == {"limite_credito_atual": 5000.0}


def test_approved_updates_limit_and_status(auth_ctx_ana_620):
    r = tools.solicitar_aumento_limite(auth_ctx_ana_620, 8000)
    assert r["status_pedido"] == STATUS_APROVADO and r["limite_aplicado"]
    assert limit(auth_ctx_ana_620.repo) == 8000
    last = rows(auth_ctx_ana_620.repo)[-1]
    assert (last["status_pedido"], last["limite_atual"], last["novo_limite_solicitado"]) == (STATUS_APROVADO, "5000.00", "8000.00")


def test_rejected_never_changes_limit(auth_ctx_ana_620):
    r = tools.solicitar_aumento_limite(auth_ctx_ana_620, 8000.01)
    assert r["status_pedido"] == STATUS_REPROVADO and not r["limite_aplicado"]
    assert limit(auth_ctx_ana_620.repo) == 5000
    assert rows(auth_ctx_ana_620.repo)[-1]["status_pedido"] == STATUS_REPROVADO


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -10, 0, "abc", None, True, 5000, 100])
def test_invalid_requested_limit_creates_nothing(auth_ctx_ana_620, bad):
    result = tools.solicitar_aumento_limite(auth_ctx_ana_620, bad)
    assert "erro" in result and "codigo" in result
    assert rows(auth_ctx_ana_620.repo) == [] and limit(auth_ctx_ana_620.repo) == 5000


def test_limit_not_above_current_has_stable_error_code(auth_ctx_ana_620):
    result = tools.solicitar_aumento_limite(auth_ctx_ana_620, 5000)

    assert result["codigo"] == "limite_menor_ou_igual_ao_atual"
    assert "erro" in result


def test_large_amount_is_evaluated_by_score_without_global_cap(auth_ctx):
    result = tools.solicitar_aumento_limite(auth_ctx, 92019201)

    assert result["status_pedido"] == STATUS_REPROVADO
    assert rows(auth_ctx.repo)[-1]["novo_limite_solicitado"] == "92019201.00"


def test_csv_contract_columns_preserved(auth_ctx_ana_620):
    result = tools.solicitar_aumento_limite(auth_ctx_ana_620, 6000)
    header = auth_ctx_ana_620.repo.requests_path.read_text().splitlines()[0]
    assert header == "cpf_cliente,data_hora_solicitacao,limite_atual,novo_limite_solicitado,status_pedido"
    saved = rows(auth_ctx_ana_620.repo)[-1]
    from datetime import datetime
    assert saved["cpf_cliente"] == ANA
    assert datetime.fromisoformat(saved["data_hora_solicitacao"]).isoformat() == saved["data_hora_solicitacao"]
    assert float(saved["limite_atual"]) == 5000.0
    assert float(saved["novo_limite_solicitado"]) == 6000.0
    assert saved["status_pedido"] == result["status_pedido"] == STATUS_APROVADO


def test_two_requests_same_second_update_correct_rows(auth_ctx, monkeypatch):
    import banco_agil.repository as R

    class Frozen(R.datetime):
        @classmethod
        def now(cls, tz=None):
            return R.datetime(2026, 1, 1, 12, 0, 0, 0)

    monkeypatch.setattr(R, "datetime", Frozen)
    repo = auth_ctx.repo
    a = repo.create_request(ANA, 5000, 6000)
    b = repo.create_request(ANA, 5000, 7000)
    assert a["id"] != b["id"] and a["data_hora_solicitacao"] == b["data_hora_solicitacao"]
    repo.set_request_status(b["id"], ANA, STATUS_REPROVADO)
    statuses = [r["status_pedido"] for r in rows(repo)]
    assert statuses == [STATUS_PENDENTE, STATUS_REPROVADO]


def test_decided_request_is_immutable(auth_ctx):
    repo = auth_ctx.repo
    req = repo.create_request(ANA, 5000, 6000)
    repo.set_request_status(req["id"], ANA, STATUS_APROVADO)
    with pytest.raises(DataError):
        repo.set_request_status(req["id"], ANA, STATUS_APROVADO)  # reprocessamento acidental
    with pytest.raises(DataError):
        repo.set_request_status(req["id"], "52998224725", STATUS_REPROVADO)  # CPF de outro cliente
    with pytest.raises(ValueError):
        repo.set_request_status(req["id"], ANA, "re" + "jeitado")


def test_repeating_same_request_is_not_approved_twice(auth_ctx_ana_620):
    assert tools.solicitar_aumento_limite(auth_ctx_ana_620, 8000)["status_pedido"] == STATUS_APROVADO
    again = tools.solicitar_aumento_limite(auth_ctx_ana_620, 8000)
    assert "erro" in again and len(rows(auth_ctx_ana_620.repo)) == 1


def test_failure_applying_limit_is_recoverable(auth_ctx_ana_620, monkeypatch):
    repo, svc = auth_ctx_ana_620.repo, auth_ctx_ana_620.credit
    original = repo.update_limit

    def boom(*a, **k):
        raise DataError("disco cheio")

    monkeypatch.setattr(repo, "update_limit", boom)
    r = tools.solicitar_aumento_limite(auth_ctx_ana_620, 8000)
    assert r["status_pedido"] == STATUS_APROVADO and r["limite_aplicado"] is False
    assert "não consegui aplicar" in limit_request_reply(r)
    assert limit(repo) == 5000 and rows(repo)[-1]["status_pedido"] == STATUS_APROVADO  # registro recuperável
    monkeypatch.setattr(repo, "update_limit", original)
    assert svc.reconcile() == 1 and limit(repo) == 8000
    assert svc.reconcile() == 0  # idempotente


def test_failure_after_intent_is_resolved_by_reconcile(auth_ctx_ana_620, monkeypatch):
    repo, svc = auth_ctx_ana_620.repo, auth_ctx_ana_620.credit
    original = repo.set_request_status
    monkeypatch.setattr(repo, "set_request_status", lambda *a, **k: (_ for _ in ()).throw(DataError("x")))
    with pytest.raises(DataError):
        svc.request_increase(ANA, 8000)
    assert [r["status_pedido"] for r in rows(repo)] == [STATUS_PENDENTE] and limit(repo) == 5000
    monkeypatch.setattr(repo, "set_request_status", original)
    svc.reconcile()
    assert [r["status_pedido"] for r in rows(repo)] == [STATUS_APROVADO] and limit(repo) == 8000


def test_reconcile_rejects_orphan_pending_above_score(auth_ctx_ana_620):
    repo = auth_ctx_ana_620.repo
    repo.create_request(ANA, 5000, 20000)
    auth_ctx_ana_620.credit.reconcile()
    assert rows(repo)[-1]["status_pedido"] == STATUS_REPROVADO and limit(repo) == 5000


def test_reconcile_does_not_overwrite_external_limit_change(auth_ctx_ana_620):
    repo = auth_ctx_ana_620.repo
    tools.solicitar_aumento_limite(auth_ctx_ana_620, 8000)
    repo.update_limit(ANA, 1234)  # ajuste manual posterior
    assert auth_ctx_ana_620.credit.reconcile() == 0 and limit(repo) == 1234


def test_score_boundary_is_inclusive(auth_ctx):
    assert auth_ctx.repo.max_limit_for_score(699) == 8000
    assert auth_ctx.repo.max_limit_for_score(700) == 15000


def test_cannot_touch_other_customer(auth_ctx_ana_620):
    with pytest.raises(TypeError):
        tools.solicitar_aumento_limite(auth_ctx_ana_620, 3000, cpf="52998224725")
    tools.solicitar_aumento_limite(auth_ctx_ana_620, 8000)
    assert limit(auth_ctx_ana_620.repo, "52998224725") == 2000


def test_missing_score_table_is_controlled_error(auth_ctx_ana_620):
    repo = auth_ctx_ana_620.repo
    repo.score_path.unlink()
    with pytest.raises(DataError):
        tools.solicitar_aumento_limite(auth_ctx_ana_620, 6000)
    # Uma tabela inválida não pode criar pedido pendente ou alterar o limite.
    assert rows(repo) == []
    assert limit(repo) == 5000


def test_credit_tool_rejects_customer_score_5000_without_financial_mutation(auth_ctx):
    repo = auth_ctx.repo
    repo._update_client(ANA, score=5000)
    clients_before = repo.clients_path.read_bytes()
    requests_before = repo.requests_path.read_bytes()

    with pytest.raises(DataError, match="Registro de cliente inválido"):
        auth_ctx.credit.request_increase(ANA, 8000)

    assert repo.clients_path.read_bytes() == clients_before
    assert repo.requests_path.read_bytes() == requests_before
