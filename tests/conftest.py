import shutil
from pathlib import Path

import pytest

from banco_agil.exchange import ExchangeService
from banco_agil.repository import BankRepository
from banco_agil.session import Session
from banco_agil.tools import ToolContext

DATA = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture
def repo(tmp_path):
    for f in DATA.glob("*.csv"):
        if f.name == "solicitacoes_aumento_limite.csv":
            continue
        shutil.copy(f, tmp_path / f.name)
    (tmp_path / "solicitacoes_aumento_limite.csv").write_text(
        "cpf_cliente,data_hora_solicitacao,limite_atual,novo_limite_solicitado,status_pedido\n",
        encoding="utf-8",
    )
    # Fixa o estado inicial na cópia temporária, independente dos atendimentos manuais.
    (tmp_path / "clientes.csv").write_text(
        "cpf,nome,data_nascimento,limite_credito,score\n"
        "11144477735,Ana Souza,1990-05-12,8888.00,1000\n"
        "52998224725,Bruno Lima,1985-11-30,2000.00,350\n"
        "12345678909,Carla Mendes,1978-03-08,15000.00,780\n",
        encoding="utf-8",
    )
    return BankRepository(tmp_path)


@pytest.fixture
def ctx(repo):
    return ToolContext(Session(), repo, ExchangeService())


@pytest.fixture
def auth_ctx(ctx):
    ctx.session.authenticated, ctx.session.cpf, ctx.session.nome = True, "11144477735", "Ana Souza"
    return ctx


# Cenário de crédito: Ana com score 620 e limite de R$ 5.000.
@pytest.fixture
def repo_ana_620(tmp_path):
    """Repositório temporário com score 620 e limite de R$ 5.000 para Ana."""
    for f in DATA.glob("*.csv"):
        if f.name == "solicitacoes_aumento_limite.csv":
            continue
        shutil.copy(f, tmp_path / f.name)
    (tmp_path / "clientes.csv").write_text(
        "cpf,nome,data_nascimento,limite_credito,score\n"
        "11144477735,Ana Souza,1990-05-12,5000.00,620\n"
        "52998224725,Bruno Lima,1985-11-30,2000.00,350\n"
        "12345678909,Carla Mendes,1978-03-08,15000.00,780\n",
        encoding="utf-8",
    )
    (tmp_path / "solicitacoes_aumento_limite.csv").write_text(
        "cpf_cliente,data_hora_solicitacao,limite_atual,novo_limite_solicitado,status_pedido\n",
        encoding="utf-8",
    )
    return BankRepository(tmp_path)


@pytest.fixture
def ctx_ana_620(repo_ana_620):
    return ToolContext(Session(), repo_ana_620, ExchangeService())


@pytest.fixture
def auth_ctx_ana_620(ctx_ana_620):
    ctx_ana_620.session.authenticated, ctx_ana_620.session.cpf, ctx_ana_620.session.nome = (
        True,
        "11144477735",
        "Ana Souza",
    )
    return ctx_ana_620
