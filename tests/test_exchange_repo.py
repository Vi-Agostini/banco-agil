
import pytest
import requests

from banco_agil import tools
from banco_agil.exchange import ExchangeError, ExchangeService
from banco_agil.privacy import mask_cpf, redact_history, redact_text
from banco_agil.repository import BankRepository, DataError, normalize_cpf, normalize_date

from provider_fakes import Resp


def fx(*responses):
    """Serviço de câmbio com HTTP simulado; cada item é Resp ou exceção, consumido em ordem."""
    queue = list(responses)

    def get(url, **kw):
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    return ExchangeService(http_get=get)


AWESOME = {"USDBRL": {"bid": "5.4321", "create_date": "2026-10-16 10:00:00"}}
FRANK = {"rates": {"BRL": 5.5}, "date": "2026-10-16"}


def test_rate_ok():
    r = fx(Resp(AWESOME)).get_rate("usd")
    assert r["moeda"] == "USD" and r["valor_em_brl"] == 5.4321 and r["fonte"] == "AwesomeAPI"
    assert r["atualizado_em"] == "16/10/2026 às 10:00:00 (horário de São Paulo)"


def test_awesomeapi_timestamp_is_converted_to_sao_paulo_time():
    data = {"USDBRL": {"bid": "5.4321", "timestamp": "1792184531"}}

    result = fx(Resp(data)).get_rate("USD")

    assert result["atualizado_em"] == "16/10/2026 às 18:02:11 (horário de São Paulo)"


def test_fallback_when_primary_unavailable():
    assert fx(requests.ConnectionError("x"), Resp(FRANK)).get_rate("USD")["fonte"].startswith("Frankfurter")


def test_timeout_is_handled():
    r = fx(requests.Timeout("t"), requests.Timeout("t"))
    with pytest.raises(ExchangeError) as e:
        r.get_rate("USD")
    assert "Timeout" not in str(e.value) and "http" not in str(e.value)  # sem detalhes internos


@pytest.mark.parametrize("payload", [{"status": 404, "code": "CoinNotExists"}, [], "x", {"USDBRL": {}},
                                     {"USDBRL": {"bid": "abc"}}, {"USDBRL": {"bid": "-1"}}, {"USDBRL": {"bid": "nan"}}])
def test_unexpected_payload_falls_back_then_errors(payload):
    with pytest.raises(ExchangeError):
        fx(Resp(payload), Resp(payload)).get_rate("USD")


def test_http_error_and_bad_json():
    with pytest.raises(ExchangeError):
        fx(Resp(status=503), Resp(bad_json=True)).get_rate("USD")


@pytest.mark.parametrize("code", ["", "US", "DOLAR", "BRL", "12$", None, 5])
def test_invalid_currency(code):
    with pytest.raises(ExchangeError):
        fx().get_rate(code)


def test_tool_returns_safe_error(auth_ctx):
    auth_ctx.fx = fx(requests.ConnectionError("secret-host"), requests.ConnectionError("secret-host"))
    r = tools.consultar_cotacao(auth_ctx, "USD")
    assert "erro" in r and "secret" not in str(r)


# Repositório e privacidade.
def test_normalizers():
    assert normalize_cpf("111.444.777-35") == "11144477735" and normalize_cpf("123") is None
    assert normalize_date("12/05/1990") == normalize_date("1990-05-12") == "1990-05-12"
    assert normalize_date("31/02/1990") is None
    assert normalize_cpf("Meu CPF é 111.444.777-35") is None
    assert normalize_date("CPF 11144477735") is None
    assert normalize_date("12/05/1850") is None


def test_csv_with_formatted_cpf_and_br_dates_still_authenticates(tmp_path):
    (tmp_path / "clientes.csv").write_text(
        "cpf,nome,data_nascimento,limite_credito,score\n111.444.777-35,Ana,12/05/1990,5000.00,620\n")
    repo = BankRepository(tmp_path)
    assert repo.authenticate("11144477735", "1990-05-12")["nome"] == "Ana"
    repo.update_score("11144477735", 700)
    assert repo.get_client("11144477735")["score"] == 700


@pytest.mark.parametrize("invalid_score", ["-1", "1001", "5000", "5000.5", "nan", "inf"])
def test_client_score_out_of_range_is_rejected_without_rewriting_csv(tmp_path, invalid_score):
    clients = tmp_path / "clientes.csv"
    clients.write_text(
        "cpf,nome,data_nascimento,limite_credito,score\n"
        f"11144477735,Ana,1990-05-12,5000.00,{invalid_score}\n",
        encoding="utf-8",
    )
    original = clients.read_bytes()
    repo = BankRepository(tmp_path)

    with pytest.raises(DataError, match="Registro de cliente inválido"):
        repo.get_client("11144477735")

    assert clients.read_bytes() == original


def test_missing_and_corrupt_files_raise_data_error(tmp_path):
    repo = BankRepository(tmp_path)
    with pytest.raises(DataError):
        repo.get_client("11144477735")
    (tmp_path / "clientes.csv").write_text("cpf,nome\n11144477735,Ana\n")
    with pytest.raises(DataError):
        repo.get_client("11144477735")


def test_original_data_untouched_by_fixtures(repo):
    from pathlib import Path
    original_path = Path(__file__).resolve().parents[1] / "data" / "clientes.csv"
    before = original_path.read_bytes()

    repo.update_score("11144477735", 1)
    repo.update_limit("11144477735", 999.0)

    assert original_path.read_bytes() == before  # nenhum byte alterado no original
    # e a cópia mutável do repo de fato mudou (o teste tem poder de detecção)
    assert "999.00" in repo.clients_path.read_text()


def test_privacy_helpers():
    assert mask_cpf("111.444.777-35") == "***.***.***-35" and mask_cpf(None) == "***"
    out = redact_text("cpf 111.444.777-35 e 11144477735 nasci em 12/05/1990")
    assert "35" in out and "1114" not in out and "1990" not in out
    h = [{"role": "assistant", "content": "", "tool_calls": [{"id": "1", "type": "function",
         "function": {"name": "x", "arguments": '{"cpf": "11144477735", "data_nascimento": "12/05/1990"}'}}]}]
    redact_history(h)
    assert "11144477735" not in str(h) and "12/05/1990" not in str(h)


def test_privacy_keeps_unrelated_dates_and_selects_only_birth_date():
    text = (
        "CPF 11144477735; vencimento em 20/11/2026; nasci em 12/05/1990; "
        "pagamento agendado para 25/11/2026."
    )
    out = redact_text(text)

    assert "11144477735" not in out
    assert "12/05/1990" not in out
    assert "20/11/2026" in out and "25/11/2026" in out
    assert redact_text("20/11/2026", redact_birth_dates=True) == "[data de nascimento]"


# Tabela de score vazia ou ausente deve causar uma falha controlada.
def _write_minimal_repo(tmp_path, score_rows: str):
    """Cria um repo mínimo com clientes.csv e score_limite.csv controlados."""
    (tmp_path / "clientes.csv").write_text(
        "cpf,nome,data_nascimento,limite_credito,score\n"
        "11144477735,Ana Souza,1990-05-12,5000.00,620\n",
        encoding="utf-8",
    )
    (tmp_path / "score_limite.csv").write_text(score_rows, encoding="utf-8")
    (tmp_path / "solicitacoes_aumento_limite.csv").write_text(
        "cpf_cliente,data_hora_solicitacao,limite_atual,novo_limite_solicitado,status_pedido\n",
        encoding="utf-8",
    )
    return BankRepository(tmp_path)


def test_empty_score_table_raises_data_error(tmp_path):
    """Tabela score_limite.csv só com cabeçalho -> DataError, sem retornar teto 0.0."""
    repo = _write_minimal_repo(
        tmp_path, "score_minimo,score_maximo,limite_maximo\n"
    )
    with pytest.raises(DataError, match="não contém faixas"):
        repo.max_limit_for_score(620)


def test_no_matching_band_raises_data_error(tmp_path):
    """Score fora de qualquer faixa -> DataError (não retorna 0.0)."""
    repo = _write_minimal_repo(
        tmp_path,
        "score_minimo,score_maximo,limite_maximo\n500,550,1000.00\n",
    )
    with pytest.raises(DataError, match="não contém faixa"):
        repo.max_limit_for_score(620)


def test_invalid_score_raises_data_error(tmp_path):
    """Score fora do intervalo 0..1000 -> DataError."""
    repo = _write_minimal_repo(
        tmp_path, "score_minimo,score_maximo,limite_maximo\n0,1000,5000.00\n"
    )
    with pytest.raises(DataError, match="Score de cliente inválido"):
        repo.max_limit_for_score(1001)


def test_empty_score_table_no_mutation_on_request_increase(tmp_path):
    """request_increase com tabela vazia levanta DataError e não muta limite nem cria pedido."""
    from banco_agil.credit_service import CreditService

    repo = _write_minimal_repo(
        tmp_path, "score_minimo,score_maximo,limite_maximo\n"
    )
    credit = CreditService(repo)
    with pytest.raises(DataError):
        credit.request_increase("11144477735", 6000.0)

    # Limite permanece o original
    assert repo.get_client("11144477735")["limite_credito"] == 5000.0
    # Nenhuma solicitação foi criada
    assert repo.list_requests() == []


def test_valid_score_table_returns_limit(tmp_path):
    """Tabela válida retorna o teto correto (regressão positiva)."""
    repo = _write_minimal_repo(
        tmp_path,
        "score_minimo,score_maximo,limite_maximo\n"
        "500,699,8000.00\n"
        "700,1000,15000.00\n",
    )
    assert repo.max_limit_for_score(620) == 8000.0
    assert repo.max_limit_for_score(750) == 15000.0
