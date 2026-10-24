"""Concorrência financeira com executor real, CSV isolado e threads com prazo."""
import csv
import json
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event

import pytest

from banco_agil.agents import AGENTS
from banco_agil.credit_service import CreditService
from banco_agil.exchange import ExchangeService
from banco_agil.repository import BankRepository
from banco_agil.session import Session
from banco_agil.tool_executor import ToolExecutor

ANA = '11144477735'
BRUNO = '52998224725'


def isolated_repo(tmp_path):
    (tmp_path / 'clientes.csv').write_text(
        'cpf,nome,data_nascimento,limite_credito,score\n'
        f'{ANA},Ana,1990-05-12,5000.00,750\n'
        f'{BRUNO},Bruno,1985-11-30,2000.00,350\n', encoding='utf-8')
    (tmp_path / 'score_limite.csv').write_text(
        'score_minimo,score_maximo,limite_maximo\n'
        '0,499,3000.00\n500,699,8000.00\n700,1000,15000.00\n', encoding='utf-8')
    (tmp_path / 'solicitacoes_aumento_limite.csv').write_text(
        'cpf_cliente,data_hora_solicitacao,limite_atual,novo_limite_solicitado,status_pedido\n',
        encoding='utf-8')
    return BankRepository(tmp_path)


def public_call(repo, cpf, amount, confirmed=True):
    def reject_http(*args, **kwargs):
        raise AssertionError('Financial operation must not use external HTTP')
    executor = ToolExecutor(repo, ExchangeService(http_get=reject_http), CreditService(repo))
    session = Session(active_agent='credito', authenticated=True, cpf=cpf)
    session.credit_proposal = dict(valor=amount, modalidade='limite_total', novo_limite=amount,
                                   confirmada=confirmed)
    session.history = [{'role': 'user', 'content': 'continue o pedido previamente autorizado'}]
    def run():
        return executor.execute_from_llm(AGENTS['credito'], session, 'solicitar_aumento_limite',
                                        json.dumps({'novo_limite': amount}))
    return run, session


def concurrent(first, second, *, first_must_finish=False):
    start, done = Barrier(2), Event()
    def worker(operation, leader):
        start.wait(timeout=5)
        if first_must_finish and not leader:
            assert done.wait(timeout=5)
        try:
            return operation()
        finally:
            if leader:
                done.set()
    with ThreadPoolExecutor(max_workers=2) as pool:
        a = pool.submit(worker, first, True)
        b = pool.submit(worker, second, False)
        return a.result(timeout=5), b.result(timeout=5)


def check_csv(repo):
    rows = repo.list_requests()
    with repo.requests_path.open(newline='', encoding='utf-8') as saved:
        persisted = list(csv.DictReader(saved))
    assert len(persisted) == len(rows)
    assert all(set(row) == {'cpf_cliente', 'data_hora_solicitacao', 'limite_atual',
                           'novo_limite_solicitado', 'status_pedido'} for row in persisted)
    assert len({row['id'] for row in rows}) == len(rows)
    for cpf in (ANA, BRUNO):
        client = repo.get_client(cpf)
        assert client['limite_credito'] <= repo.max_limit_for_score(client['score'])
    return rows


@pytest.mark.parametrize('run', range(20))
def test_same_client_two_confirmed_sessions_no_lost_update_or_lower_limit(tmp_path, run):
    repo = isolated_repo(tmp_path)
    other = BankRepository(tmp_path)
    assert repo.lock is other.lock
    eight, _ = public_call(repo, ANA, 8000)
    nine, _ = public_call(other, ANA, 9000)
    result8, result9 = concurrent(eight, nine)
    rows = check_csv(repo)
    assert repo.get_client(ANA)['limite_credito'] == 9000
    assert result9['status_pedido'] == 'aprovado'
    assert sum(float(row['novo_limite_solicitado']) == 9000 for row in rows) == 1
    if 'erro' in result8:
        assert result8['codigo'] == 'limite_menor_ou_igual_ao_atual'
        assert len(rows) == 1
    else:
        assert result8['status_pedido'] == 'aprovado'
        assert len(rows) == 2
        assert [float(row['novo_limite_solicitado']) for row in rows] == [8000, 9000]
        assert rows[1]['limite_atual'] == '8000.00'


def test_larger_limit_completes_first_smaller_operation_is_blocked(tmp_path):
    repo = isolated_repo(tmp_path)
    nine, _ = public_call(repo, ANA, 9000)
    eight, _ = public_call(BankRepository(tmp_path), ANA, 8000)
    result9, result8 = concurrent(nine, eight, first_must_finish=True)
    assert result9['status_pedido'] == 'aprovado'
    assert result8['codigo'] == 'limite_menor_ou_igual_ao_atual'
    assert repo.get_client(ANA)['limite_credito'] == 9000
    assert len(check_csv(repo)) == 1


@pytest.mark.parametrize('run', range(10))
def test_second_session_without_consent_cannot_write_or_apply(tmp_path, run):
    repo = isolated_repo(tmp_path)
    legitimate, _ = public_call(repo, ANA, 8000)
    unconfirmed, session = public_call(BankRepository(tmp_path), ANA, 9000, confirmed=False)
    approved, blocked = concurrent(legitimate, unconfirmed)
    assert approved['status_pedido'] == 'aprovado'
    assert blocked['codigo'] == 'confirmacao_pedido_necessaria'
    assert session.credit_proposal['confirmada'] is False
    assert repo.get_client(ANA)['limite_credito'] == 8000
    rows = check_csv(repo)
    assert len(rows) == 1 and float(rows[0]['novo_limite_solicitado']) == 8000


@pytest.mark.parametrize('run', range(10))
def test_two_clients_concurrent_updates_are_both_preserved(tmp_path, run):
    repo = isolated_repo(tmp_path)
    ana, _ = public_call(repo, ANA, 8000)
    bruno, _ = public_call(BankRepository(tmp_path), BRUNO, 2500)
    result_ana, result_bruno = concurrent(ana, bruno)
    assert result_ana['status_pedido'] == result_bruno['status_pedido'] == 'aprovado'
    assert repo.get_client(ANA)['limite_credito'] == 8000
    assert repo.get_client(BRUNO)['limite_credito'] == 2500
    rows = check_csv(repo)
    assert len(rows) == 2 and {r['cpf_cliente'] for r in rows} == {ANA, BRUNO}


@pytest.mark.parametrize('run', range(5))
def test_concurrent_larger_request_cannot_exceed_actual_score_ceiling(tmp_path, run):
    repo = isolated_repo(tmp_path)
    repo.update_score(ANA, 620)  # O teto desta fixture é R$ 8.000.
    eight, _ = public_call(repo, ANA, 8000)
    nine, _ = public_call(BankRepository(tmp_path), ANA, 9000)
    result8, result9 = concurrent(eight, nine)
    assert result8['status_pedido'] == 'aprovado'
    assert result9['status_pedido'] == 'reprovado' and not result9['limite_aplicado']
    assert repo.get_client(ANA)['limite_credito'] == 8000
    rows = check_csv(repo)
    assert len(rows) == 2
    assert {(float(r['novo_limite_solicitado']), r['status_pedido']) for r in rows} == {
        (8000, 'aprovado'), (9000, 'reprovado')}
