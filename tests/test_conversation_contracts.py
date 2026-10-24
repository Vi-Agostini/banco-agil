"""LLM falso verifica contratos e fluxo; não comprova compreensão do Groq."""
import copy
import json

import pytest
from banco_agil.scoring import parse_money

from banco_agil.agents import AGENTS
from banco_agil.flow import Awaiting, FlowController
from banco_agil.llm import _tool_schemas
from fakes import reply, assert_history_valid
from orchestrator_fakes import make, logged_in
from support import snapshot_csv as snapshots


@pytest.mark.parametrize('text', ['8000 reais', 'R$ 8.000,00', '8 mil', 'oito mil', 'quero aumentar para oito mil'])
def test_interpreted_amount_is_stored_before_financial_request(repo_ana_620, text):
    orch = make(repo_ana_620, [
        reply(calls=[('registrar_intencao_credito', {'valor': 8000, 'modalidade': 'limite_total'})]),
        reply('Entendi o valor. Você quer prosseguir com esse pedido?'),
    ])
    session = logged_in(orch, agent='credito')
    before = snapshots(orch.repo)
    orch.handle(session, text)
    assert session.credit_proposal['novo_limite'] == 8000
    assert session.last_request_id is None and session.desired_limit is None
    assert snapshots(orch.repo) == before
    assert text in json.dumps(orch.client.requests, ensure_ascii=False)
    orch.client.assert_exhausted()


@pytest.mark.parametrize('agent', ['triagem', 'credito', 'entrevista', 'cambio'])
def test_collected_credit_context_survives_short_history_and_agent_changes(repo_ana_620, agent):
    orch = make(repo_ana_620, [reply('Podemos continuar a conversa.') for _ in range(3)])
    session = logged_in(orch, agent=agent)
    result = orch.tool_executor.execute(AGENTS[agent], session, 'registrar_intencao_credito',
        '{"valor":8000,"modalidade":"limite_total"}')
    assert result['ok']
    for message in ['uma dúvida', 'mais um esclarecimento', 'vamos continuar']:
        orch.handle(session, message)
    payload = orch.client.requests[-1]
    assert any('8000' in m['content'] for m in payload if m['role'] == 'system')
    assert session.credit_proposal['novo_limite'] == 8000
    assert orch.repo.list_requests() == []


def test_unauthenticated_collection_cannot_perform_bank_operations(repo_ana_620):
    orch = make(repo_ana_620, [])
    session = orch.new_session()
    before = snapshots(orch.repo)
    result = orch.tool_executor.execute_from_llm(AGENTS['triagem'], session,
        'registrar_intencao_credito', '{"valor":3000,"modalidade":"acrescimo"}')
    assert result['ok'] and 'novo_limite' not in session.credit_proposal
    names = {t['function']['name'] for t in _tool_schemas(AGENTS['triagem'], session)}
    assert 'transferir_para_agente' not in names and 'autenticar_cliente' not in names
    assert snapshots(orch.repo) == before
    assert session.auth_failures == 0 and not session.authenticated


def test_increment_is_calculated_by_code_and_only_then_requested(repo_ana_620):
    orch = make(repo_ana_620, [
        reply(calls=[('registrar_intencao_credito', {'valor': 3000, 'modalidade': 'acrescimo'})]),
        reply(calls=[('solicitar_aumento_limite', {'novo_limite': 8000})]),
        reply('Seu novo limite é R$ 8.000,00.'),
    ])
    session = logged_in(orch, agent='credito')
    orch.handle(session, 'quero aumentar em 3000 reais')
    rows = orch.repo.list_requests()
    assert len(rows) == 1 and rows[0]['novo_limite_solicitado'] == '8000.00'
    assert orch.repo.get_client(session.cpf)['limite_credito'] == 8000
    orch.client.assert_exhausted()


def test_ambiguous_amount_is_clarified_without_financial_execution(repo_ana_620):
    orch = make(repo_ana_620, [
        reply(calls=[('registrar_intencao_credito', {'valor': 3000, 'modalidade': 'limite_total'})]),
        reply('Esse valor representa o total desejado ou um acréscimo ao limite atual?'),
    ])
    session = logged_in(orch, agent='credito')
    before = snapshots(orch.repo)
    response = orch.handle(session, 'quero um aumento de 3000')
    assert session.credit_proposal['modalidade'] == 'indefinida'
    assert '?' in response
    assert 'me envie **confirmo** para continuar' not in response
    assert 'solicitar_aumento_limite' not in {t['function']['name'] for t in _tool_schemas(AGENTS['credito'], session)}
    assert snapshots(orch.repo) == before


@pytest.mark.parametrize('bad_calls', [
    [('consultar_limite', '{invalid')], [('consultar_limite', {'cpf': 'terceiro'})],
    [('solicitar_aumento_limite', {'novo_limite': None})],
    [('ferramenta_inexistente', {})], [('consultar_limite<|channel|>commentary', {})],
    [('transferir_para_agente', {'destino': 'cambio'})],
    [('encerrar_atendimento', {})],
])
def test_entire_batch_is_rejected_before_any_effect(repo_ana_620, bad_calls):
    orch = make(repo_ana_620, [reply(calls=[('solicitar_aumento_limite', {'novo_limite': 8000}), *bad_calls])])
    session = logged_in(orch, agent='credito')
    before = snapshots(orch.repo)
    orch.handle(session, 'quero limite para 8000')
    assert snapshots(orch.repo) == before
    assert not session.ended and session.active_agent == 'credito'
    assert not any(m.get('tool_calls') for m in session.history)
    assert session.last_failure_category == 'invalid_tool_response'


def test_runtime_rechecks_agent_scope(repo_ana_620):
    orch = make(repo_ana_620, [])
    session = logged_in(orch, agent='credito')
    assert orch.tool_executor.execute_from_llm(AGENTS['credito'], session,
        'transferir_para_agente', '{"destino":"cambio"}')['ok']
    before = snapshots(orch.repo)
    result = orch.tool_executor.execute_from_llm(AGENTS['credito'], session,
        'solicitar_aumento_limite', '{"novo_limite":8000}')
    assert result['codigo'] == 'ferramenta_fora_escopo'
    assert snapshots(orch.repo) == before


def test_interview_answers_out_of_order_and_handoff_preserve_confirmation(repo_ana_620):
    orch = make(repo_ana_620, [
        reply(calls=[('registrar_respostas_entrevista', {'tem_dividas': False, 'tipo_emprego': 'formal'})]),
        reply('Entendi emprego e dívidas. Você confirma esse resumo?'),
        reply('Quais informações sobre seus gastos gostaria de acrescentar?'),
    ])
    session = logged_in(orch, agent='entrevista')
    FlowController.start_interview(session)
    before = snapshots(orch.repo)
    orch.handle(session, 'sou CLT e não tenho dívidas')
    orch.handle(session, 'sim')
    confirmed = copy.deepcopy(session.flow.interview_answers)
    orch.tool_executor.execute_from_llm(AGENTS['entrevista'], session, 'transferir_para_agente', '{"destino":"credito"}')
    orch.tool_executor.execute_from_llm(AGENTS['credito'], session, 'transferir_para_agente', '{"destino":"entrevista"}')
    assert session.flow.interview_answers == confirmed == {'tem_dividas': False, 'tipo_emprego': 'formal'}
    assert snapshots(orch.repo) == before
    assert_history_valid(session.history)


@pytest.mark.parametrize('fake_text', [
    'Seu limite foi aumentado para R$ 8.000,00.', 'Seu pedido foi aprovado.',
    'Registrei sua solicitação.', 'Seu score foi atualizado para 900.',
])
def test_reply_cannot_announce_nonexistent_operation(repo_ana_620, fake_text):
    orch = make(repo_ana_620, [reply(fake_text)])
    session = logged_in(orch, agent='credito')
    before = snapshots(orch.repo)
    response = orch.handle(session, 'quero limite para 8000')
    assert response != fake_text
    assert session.last_failure_category == 'ungrounded_response'
    assert snapshots(orch.repo) == before


@pytest.mark.parametrize('wrong', ['A cotação do dólar é R$ 9,99.', 'A cotação do euro é R$ 5,4321.'])
def test_currency_and_value_must_match_tool_result(repo_ana_620, wrong):
    orch = make(repo_ana_620, [reply(calls=[('consultar_cotacao', {'moeda': 'USD'})]), reply(wrong)])
    orch.fx.get_rate = lambda _: {'moeda': 'USD', 'valor_em_brl': 5.4321, 'fonte': 'HTTP local', 'atualizado_em': '2026-10-21'}
    session = logged_in(orch, agent='cambio')
    before = snapshots(orch.repo)
    response = orch.handle(session, 'cotação do dólar')
    assert parse_money(response.split('R$ ')[1].split('.')[0]) == 5.4321 and 'HTTP local' in response
    assert response != wrong and session.last_failure_category == 'ungrounded_response'
    assert snapshots(orch.repo) == before


def test_natural_wording_of_grounded_financial_reply_is_preserved(repo_ana_620):
    text = 'Concluímos! Seu limite foi aumentado para R$\u202f8.000,00. Em que mais posso ajudar?'
    orch = make(repo_ana_620, [reply(calls=[('solicitar_aumento_limite', {'novo_limite': 8000})]), reply(text)])
    session = logged_in(orch, agent='credito')
    assert orch.handle(session, 'quero aumentar para 8000') == 'Concluímos! Seu limite foi aumentado para R$\u202f8.000,00.\n\nEm que mais posso ajudar?'
    assert session.last_failure_category is None


@pytest.mark.parametrize('text', ['Se o pedido for aprovado, podemos continuar.',
                                 'Você deseja que o limite seja aumentado?',
                                 'Quais são seus gastos mais importantes hoje?'])
def test_questions_and_conditional_explanations_are_not_success_claims(repo_ana_620, text):
    orch = make(repo_ana_620, [reply(text)])
    session = logged_in(orch, agent='credito')
    assert orch.handle(session, 'quero entender as opções') == text
    assert session.last_failure_category is None
    assert orch.repo.list_requests() == []


def test_old_success_cannot_confirm_a_new_operation(repo_ana_620):
    orch = make(repo_ana_620, [reply(calls=[('solicitar_aumento_limite', {'novo_limite': 8000})]),
                             reply('Seu pedido foi aprovado.'), reply('Seu pedido foi aprovado.')])
    session = logged_in(orch, agent='credito')
    orch.handle(session, 'quero aumentar para 8000')
    orch.handle(session, 'agora quero aumentar para 9000')
    assert session.last_failure_category == 'ungrounded_response'
    assert len(orch.repo.list_requests()) == 1
    assert orch.repo.get_client(session.cpf)['limite_credito'] == 8000


def test_interview_state_is_presented_to_other_agents(repo_ana_620):
    orch = make(repo_ana_620, [])
    session = logged_in(orch, agent='entrevista')
    FlowController.start_interview(session)
    session.flow.interview_answers = {'renda_mensal': 6000}
    session.flow.pending_interview_answers = {'despesas_fixas': 2000}
    orch.tool_executor.execute_from_llm(AGENTS['entrevista'], session,
        'transferir_para_agente', '{"destino":"credito"}')
    prompt = AGENTS['credito'].instructions() + AGENTS['credito'].session_context(session)
    assert 'renda_mensal' in prompt and '6000' in prompt
    assert 'despesas_fixas' in prompt and '2000' in prompt


def test_a_new_complete_request_is_not_blocked_by_an_old_executed_proposal(repo_ana_620):
    orch = make(repo_ana_620, [
        reply(calls=[('solicitar_aumento_limite', {'novo_limite': 6000})]), reply('Pedido aprovado.'),
        reply(calls=[('solicitar_aumento_limite', {'novo_limite': 8000})]), reply('Pedido aprovado.'),
    ])
    session = logged_in(orch, agent='credito')
    orch.handle(session, 'quero aumentar para 6000')
    orch.handle(session, 'agora quero aumentar para 8000')
    assert len(orch.repo.list_requests()) == 2
    assert orch.repo.get_client(session.cpf)['limite_credito'] == 8000
    orch.client.assert_exhausted()


def test_prelogin_increment_must_be_resolved_after_authentication(repo_ana_620):
    orch = make(repo_ana_620, [])
    session = orch.new_session()
    orch.tool_executor.execute_from_llm(AGENTS['triagem'], session, 'registrar_intencao_credito',
        '{"valor":3000,"modalidade":"acrescimo"}')
    session.authenticated, session.cpf, session.active_agent = True, '11144477735', 'credito'
    assert 'solicitar_aumento_limite' not in {t['function']['name'] for t in _tool_schemas(AGENTS['credito'], session)}
    result = orch.tool_executor.execute_from_llm(AGENTS['credito'], session, 'registrar_intencao_credito',
        '{"valor":3000,"modalidade":"acrescimo"}')
    assert result['proposta']['novo_limite'] == 8000
    assert orch.repo.list_requests() == []


def test_spelled_amount_requires_confirmation_before_operation(repo_ana_620):
    orch = make(repo_ana_620, [
        reply(calls=[('registrar_intencao_credito', {'valor': 8000, 'modalidade': 'limite_total'})]),
        reply('Entendi: o limite total desejado é R$ 8.000,00. Confirma?'),
        reply(calls=[('solicitar_aumento_limite', {'novo_limite': 8000})]), reply('Seu pedido foi aprovado.'),
    ])
    session = logged_in(orch, agent='credito')
    orch.handle(session, 'quero aumentar para oito mil')
    assert not session.credit_proposal['confirmada'] and orch.repo.list_requests() == []
    assert 'solicitar_aumento_limite' not in {t['function']['name'] for t in _tool_schemas(AGENTS['credito'], session)}
    orch.handle(session, 'sim')
    assert len(orch.repo.list_requests()) == 1
    assert orch.repo.get_client(session.cpf)['limite_credito'] == 8000


def test_model_cannot_invent_a_financial_request_from_a_greeting(repo_ana_620):
    orch = make(repo_ana_620, [reply(calls=[('solicitar_aumento_limite', {'novo_limite': 8000})])])
    session = logged_in(orch, agent='credito')
    before = snapshots(orch.repo)
    orch.handle(session, 'bom dia')
    assert snapshots(orch.repo) == before
    assert session.last_failure_category == 'invalid_tool_response'


@pytest.mark.parametrize('wrong', [
    'A cotação do dólar é R$ 5,4321, fonte AwesomeAPI.',
    'A cotação do dólar é R$ 5,4321, atualizada em 20/10/2026.',
    'A cotação do dólar é R$ 5,4321, atualizada às 12:00.',
])
def test_quote_metadata_cannot_be_invented(repo_ana_620, wrong):
    orch = make(repo_ana_620, [reply(calls=[('consultar_cotacao', {'moeda': 'USD'})]), reply(wrong)])
    orch.fx.get_rate = lambda _: {'moeda': 'USD', 'valor_em_brl': 5.4321, 'fonte': 'Frankfurter (BCE)', 'atualizado_em': '2026-10-21'}
    session = logged_in(orch, agent='cambio')
    response = orch.handle(session, 'cotação do dólar')
    assert response != wrong and session.last_failure_category == 'ungrounded_response'
    assert '2026-10-21' in response and '12:00' not in response


@pytest.mark.parametrize('wrong', ['Seu limite atual é 9000.', 'Seu limite atual é nove mil reais.'])
def test_untagged_or_spelled_financial_claim_is_not_trusted(repo_ana_620, wrong):
    orch = make(repo_ana_620, [reply(calls=[('consultar_limite', {})]), reply(wrong)])
    session = logged_in(orch, agent='credito')
    response = orch.handle(session, 'consulte meu limite')
    assert response != wrong and session.last_failure_category == 'ungrounded_response'
    assert '5.000,00' in response and orch.repo.list_requests() == []


def test_untagged_quote_value_is_checked(repo_ana_620):
    orch = make(repo_ana_620, [reply(calls=[('consultar_cotacao', {'moeda': 'USD'})]),
                             reply('A cotação do dólar é 9,99.')])
    orch.fx.get_rate = lambda _: {'moeda': 'USD', 'valor_em_brl': 5.4321, 'fonte': 'local', 'atualizado_em': '2026-10-21'}
    session = logged_in(orch, agent='cambio')
    response = orch.handle(session, 'cotação do dólar')
    assert '5,4321' in response and '9,99' not in response
    assert session.last_failure_category == 'ungrounded_response'


def test_interview_can_start_with_a_question_other_than_income(repo_ana_620):
    text = 'Você trabalha por conta própria ou tem carteira assinada?'
    orch = make(repo_ana_620, [reply(text)])
    session = logged_in(orch, agent='credito')
    session.flow.awaiting = Awaiting.INTERVIEW_OFFER
    assert orch.handle(session, 'sim') == text
    assert session.active_agent == 'entrevista'
    assert orch.repo.list_requests() == []


def test_increment_uses_decimal_arithmetic(repo_ana_620):
    orch = make(repo_ana_620, [])
    session = logged_in(orch, agent='credito')
    orch.repo.update_limit(session.cpf, 0.1)
    before = snapshots(orch.repo)
    session.history.append({'role': 'user', 'content': 'quero aumentar em 0,20'})
    result = orch.tool_executor.execute_from_llm(AGENTS['credito'], session,
        'registrar_intencao_credito', '{"valor":0.2,"modalidade":"acrescimo"}')
    assert result['proposta']['novo_limite'] == 0.3
    assert snapshots(orch.repo) == before


def test_confirmation_question_cannot_substitute_a_different_amount(repo_ana_620):
    orch = make(repo_ana_620, [
        reply(calls=[('registrar_intencao_credito', {'valor': 8000, 'modalidade': 'limite_total'})]),
        reply('O limite desejado é R$ 9.000,00. Confirma?'),
    ])
    session = logged_in(orch, agent='credito')
    response = orch.handle(session, 'quero aumentar para oito mil')
    assert '8.000,00' in response and '9.000,00' not in response
    assert not session.credit_proposal['confirmada']
    assert session.last_failure_category == 'ungrounded_response'
    assert orch.repo.list_requests() == []


def test_interview_summary_cannot_change_the_proposed_income(repo_ana_620):
    orch = make(repo_ana_620, [
        reply(calls=[('registrar_respostas_entrevista', {'renda_mensal': 6000})]),
        reply('Sua renda é R$ 9.000,00. Confirma?'),
    ])
    session = logged_in(orch, agent='entrevista')
    FlowController.start_interview(session)
    before = snapshots(orch.repo)
    response = orch.handle(session, 'minha renda é seis mil')
    assert '6.000,00' in response and '9.000,00' not in response
    assert session.flow.pending_interview_answers == {'renda_mensal': 6000}
    assert snapshots(orch.repo) == before


def test_a_consultation_question_does_not_authorize_an_increase(repo_ana_620):
    orch = make(repo_ana_620, [reply(calls=[('solicitar_aumento_limite', {'novo_limite': 8000})])])
    session = logged_in(orch, agent='credito')
    before = snapshots(orch.repo)
    orch.handle(session, 'meu limite é 8000?')
    assert snapshots(orch.repo) == before
    assert session.last_failure_category == 'invalid_tool_response'


def test_provider_failure_does_not_hide_the_data_pending_confirmation(repo_ana_620):
    from fakes import ScriptedClient, ToolUseFailed
    orch = make(repo_ana_620, [])
    orch.client = ScriptedClient([
        reply(calls=[('registrar_intencao_credito', {'valor': 8000, 'modalidade': 'limite_total'})]),
        ToolUseFailed(),
        ToolUseFailed(),  # Esgota a recuperação seletiva antes da resposta segura.
    ])
    session = logged_in(orch, agent='credito')
    before = snapshots(orch.repo)
    response = orch.handle(session, 'quero aumentar para oito mil')
    assert '8.000,00' in response
    assert not session.credit_proposal['confirmada']
    assert session.last_failure_category == 'http_400'
    assert len(orch.client.requests) == 3
    assert snapshots(orch.repo) == before
