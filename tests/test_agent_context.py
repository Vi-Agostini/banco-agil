"""A seleção de contexto preserva o estado e as autorizações financeiras."""
import copy
import json

import pytest

from banco_agil.agent_context import select_history
from banco_agil.agents import AGENTS
from banco_agil.flow import FlowController
from banco_agil.llm import _build_payload, _context_history, _estimate_request_tokens
from banco_agil.reply_recorder import ReplyRecorder
from banco_agil.session import Session
from banco_agil.structured_output import build_payload, record_result, response_format
from fakes import assert_history_valid, failure
from orchestrator_fakes import logged_in
from structured_fakes import action, message, make, check_protocol
from support import snapshot_csv as csv


ANSWERS = dict(renda_mensal=6000, tipo_emprego='formal', despesas_fixas=2000,
               num_dependentes=1, tem_dividas=False)


def systems(request):
    return '\n'.join(m['content'] for m in request['messages'] if m['role'] == 'system')


def test_projections_keep_auth_outstanding_work_source_and_only_relevant_old_facts():
    s = Session(authenticated=True, nome='Pessoa fictícia', active_agent='entrevista', turn=8)
    FlowController.start_interview(s)
    s.flow.interview_answers = {'renda_mensal': 6000}
    s.flow.pending_interview_answers = {'tipo_emprego': 'autonomo'}
    s.desired_limit, s.last_request_id, s.last_request_status = 50000, 3, 'reprovado'
    s.credit_proposal = dict(valor=50000, modalidade='limite_total', novo_limite=50000, executada=True)
    s.tool_facts = {
        'consultar_cotacao': {'turn': 2, 'result': {'moeda': 'USD', 'valor_em_brl': 5.4321, 'fonte': 'fictícia'}},
        'consultar_limite': {'turn': 1, 'result': {'limite_credito_atual': 5000}},
        'solicitar_aumento_limite': {'turn': 6, 'result': {'status_pedido': 'reprovado', 'novo_limite_solicitado': 50000}},
    }
    before = copy.deepcopy((s.flow, s.tool_facts, s.credit_proposal))
    interview = AGENTS['entrevista'].session_context(s)
    credit = AGENTS['credito'].session_context(s)
    exchange = AGENTS['cambio'].session_context(s)
    for prompt in (interview, credit, exchange, AGENTS['triagem'].session_context(s)):
        assert 'Cliente autenticado' in prompt and 'não peça CPF ou data' in prompt
    assert 'renda_mensal' in interview and 'tipo_emprego' in interview
    assert '50000' in interview and '"id":3' in interview and 'reprovado' in interview
    assert 'USD' not in interview and 'consultar_limite' not in interview and '5000,' not in interview
    assert 'consultar_limite' in credit and 'USD' not in credit
    assert 'USD' in exchange and 'entrevista em andamento preservada' in exchange
    assert 'renda_mensal' not in exchange and 'tipo_emprego' not in exchange
    assert (s.flow, s.tool_facts, s.credit_proposal) == before


@pytest.mark.parametrize('mode', ['native', 'structured'])
def test_completed_foreign_receipts_are_omitted_as_whole_groups_with_measurable_reduction(mode):
    s = Session(authenticated=True, active_agent='entrevista', turn=2)
    FlowController.start_interview(s)
    s.history = [{'role': 'user', 'content': 'Qual é a cotação do dólar?'}]
    quote = {'moeda': 'USD', 'valor_em_brl': 5.4321, 'fonte': 'API fictícia',
             'atualizado_em': '2026-10-22', 'instrucao': 'Apresente a cotação com a fonte e atualização recebidas.'}
    if mode == 'native':
        s.history.extend([{'role': 'assistant', 'content': None, 'tool_calls': [{'id': 'fx', 'type': 'function',
                           'function': {'name': 'consultar_cotacao', 'arguments': '{"moeda":"USD"}'}}]},
                          {'role': 'tool', 'tool_call_id': 'fx', 'content': json.dumps(quote)}])
    else:
        s.history.append({'role': 'assistant', 'content': json.dumps({'resposta': action('consultar_cotacao', moeda='USD')})})
        record_result(s, 'consultar_cotacao', quote)
    s.history.extend([{'role': 'assistant', 'content': 'A cotação do dólar é R$ 5,4321. Deseja retomar a entrevista?'},
                      {'role': 'user', 'content': 'sim, retome a entrevista'}])
    before = copy.deepcopy(s.history)
    raw = _context_history(s.history)
    projected = _build_payload(AGENTS['entrevista'], s)
    baseline = projected[:2] + raw
    if mode == 'structured':
        fmt = response_format([])
        assert _estimate_request_tokens(build_payload(projected), [], fmt) < _estimate_request_tokens(build_payload(baseline), [], fmt)
    else:
        assert _estimate_request_tokens(projected, []) < _estimate_request_tokens(baseline, [])
        assert_history_valid(projected)
    assert projected[2:] == [raw[0], *raw[-2:]]
    assert s.history == before  # A exibição, o estado financeiro e o histórico completo permanecem intactos.


@pytest.mark.parametrize('kind', ['current', 'error', 'unknown', 'incomplete', 'mixed', 'ok_false', 'pending'])
def test_filter_keeps_current_failed_unknown_incomplete_and_mixed_native_batches(kind):
    s = Session(authenticated=True, active_agent='entrevista', turn=5)
    name = 'unknown_tool' if kind == 'unknown' else 'consultar_cotacao'
    calls = [{'id': 'fx', 'function': {'name': name, 'arguments': '{}'}}]
    results = [{'role': 'tool', 'tool_call_id': 'fx', 'content': json.dumps({'erro': 'indisponível'} if kind == 'error' else {'ok': True})}]
    if kind == 'ok_false':
        results[0]['content'] = '{"ok":false}'
    elif kind == 'pending':
        calls[0]['function']['name'] = 'solicitar_aumento_limite'
        results[0]['content'] = '{"status_pedido":"aprovado","limite_aplicado":false}'
    if kind == 'mixed':
        calls.append({'id': 'collection', 'function': {'name': 'registrar_respostas_entrevista', 'arguments': '{}'}})
        results.append({'role': 'tool', 'tool_call_id': 'collection', 'content': '{"ok":true}'})
    if kind == 'incomplete':
        results[0]['tool_call_id'] = 'another-call'
    history = [{'role': 'user', 'content': 'mensagem anterior'}, {'role': 'assistant', 'content': None, 'tool_calls': calls}, *results]
    if kind != 'current':
        history.append({'role': 'user', 'content': 'mensagem atual'})
    before = copy.deepcopy(history)
    assert select_history('entrevista', s, history) == before
    assert history == before


def test_mismatched_structured_proposal_result_group_is_kept_intact():
    s = Session(active_agent='entrevista', authenticated=True)
    history = [{'role': 'user', 'content': 'mensagem anterior'},
        {'role': 'assistant', 'content': json.dumps({'resposta': action('registrar_respostas_entrevista', renda_mensal=6000)})},
        {'role': 'system', 'content': json.dumps({'resultado_local': {'acao': 'consultar_cotacao', 'resultado': {'ok': True}}})},
        {'role': 'user', 'content': 'mensagem atual'}]
    assert select_history('entrevista', s, history) == history


@pytest.mark.parametrize('mode', ['native', 'structured'])
def test_credit_interview_reanalysis_keeps_exact_rejected_source_after_first_approval_and_failure(repo_ana_620, mode):
    orch = make(repo_ana_620, [action('solicitar_aumento_limite', novo_limite=2500),
        message('Seu novo limite aprovado é R$ 2.500,00.'), action('solicitar_aumento_limite', novo_limite=7000),
        message('Por quais informações sobre sua renda gostaria de começar?'),
        action('registrar_respostas_entrevista', **ANSWERS), message('Você confirma os cinco dados informados?'),
        action('registrar_entrevista_e_calcular_score', **ANSWERS), TimeoutError(),
        action('consultar_limite'), message('Seu limite de crédito atual é R$ 7.000,00.')], mode)
    s = logged_in(orch, agent='credito', cpf='52998224725', nome='Bruno Lima')
    orch.handle(s, 'Quero aumentar para 2500')
    orch.handle(s, 'Agora quero aumentar para 7000')
    assert s.last_request_id == 1 and s.last_request_status == 'reprovado'
    orch.handle(s, 'sim')
    assert s.active_agent == 'entrevista' and '"id":1' in systems(orch.client.requests[-1])
    orch.handle(s, 'Ganho 6000, sou CLT, gasto 2000, tenho um dependente e não tenho dívidas')
    result = orch.handle(s, 'isso')
    assert 'aprovado' in result
    assert [r['status_pedido'] for r in orch.repo.list_requests()] == ['aprovado', 'reprovado', 'aprovado']
    assert s.automatic_reanalysis_source_id == 1 and s.last_request_id == 2
    assert orch.repo.get_client(s.cpf)['limite_credito'] == 7000
    assert s.active_agent == 'credito' and s.authenticated and not s.ended
    orch.handle(s, 'Qual é meu limite agora?')
    assert len(orch.repo.list_requests()) == 3
    check_protocol(orch, s)


@pytest.mark.parametrize('mode', ['native', 'structured'])
@pytest.mark.parametrize('question,confirmed,user_reply,updates,next_question', [
    ('Qual é seu tipo de emprego?', {'renda_mensal': 6000}, 'autonomo', {'tipo_emprego': 'autônomo'},
     'Qual é o valor das suas despesas fixas mensais?'),
    ('Qual é o valor das suas despesas fixas mensais?', {'renda_mensal': 6000, 'tipo_emprego': 'formal'},
     '2000', {'despesas_fixas': 2000}, 'Quantos dependentes você tem?'),
])
def test_interview_exchange_long_detour_and_generation_failure_restore_question_and_stage(
    repo_ana_620, mode, question, confirmed, user_reply, updates, next_question,
):
    orch = make(repo_ana_620, [message(question),
        action('transferir_para_agente', destino='credito'), action('transferir_para_agente', destino='cambio'),
        action('consultar_cotacao', moeda='USD'), message('A cotação do dólar é R$ 5,4321.'),
        action('consultar_cotacao', moeda='USD'), message('A cotação do dólar é R$ 5,4321.'),
        action('transferir_para_agente', destino='credito'), action('transferir_para_agente', destino='entrevista'),
        failure('json_validate_failed'), failure('json_validate_failed'),
        action('registrar_respostas_entrevista', **updates), message('Você confirma o dado informado?'),
        message(next_question)], mode)
    orch.fx.get_rate = lambda _: {'moeda': 'USD', 'valor_em_brl': 5.4321, 'fonte': 'API fictícia', 'atualizado_em': '2026-10-22'}
    s = logged_in(orch, agent='entrevista')
    FlowController.start_interview(s)
    s.flow.interview_answers = dict(confirmed)
    before = csv(orch.repo)
    orch.handle(s, 'Qual é a próxima etapa?')
    orch.handle(s, 'Antes quero a cotação do dólar')
    orch.handle(s, 'Consulte o dólar outra vez')
    orch.handle(s, 'Quero voltar à entrevista')
    assert s.active_agent == 'entrevista' and s.flow.interview_started
    assert s.flow.interview_answers == confirmed
    saved = copy.deepcopy(s.agent_questions['entrevista'])
    orch.handle(s, user_reply)
    request = orch.client.requests[-2]
    assert any(m['role'] == 'assistant' and m.get('content') == question for m in request['messages'])
    assert s.agent_questions['entrevista'] != saved
    normalized = {key: ('autonomo' if value == 'autônomo' else value) for key, value in updates.items()}
    assert s.flow.pending_interview_answers == normalized
    assert orch.handle(s, 'isso') == next_question
    assert s.flow.interview_answers == {**confirmed, **normalized}
    assert s.auth_failures == 0 and s.authenticated and not s.ended and csv(orch.repo) == before
    check_protocol(orch, s)


@pytest.mark.parametrize('mode', ['native', 'structured'])
def test_exchange_acknowledgement_does_not_confirm_interview_or_financial_proposal(repo_ana_620, mode):
    orch = make(repo_ana_620, [action('transferir_para_agente', destino='credito'),
        action('transferir_para_agente', destino='cambio'), message('Você quer consultar uma moeda?'),
        message('Qual moeda você gostaria de consultar?'), action('transferir_para_agente', destino='credito'),
        action('transferir_para_agente', destino='entrevista'), message('Você confirma a renda informada?'),
        message('Qual é seu tipo de emprego?')], mode)
    s = logged_in(orch, agent='entrevista')
    FlowController.start_interview(s)
    s.flow.pending_interview_answers = {'renda_mensal': 6000}
    s.credit_proposal = dict(valor=8000, novo_limite=8000, modalidade='limite_total', confirmada=False)
    proposal = copy.deepcopy(s.credit_proposal)
    before = csv(orch.repo)
    orch.handle(s, 'Antes quero saber do dólar')
    orch.handle(s, 'sim')
    assert s.active_agent == 'cambio' and not s.flow.interview_answers
    assert s.flow.pending_interview_answers == {'renda_mensal': 6000}
    assert s.credit_proposal == proposal
    orch.handle(s, 'Quero retomar a entrevista')
    assert s.active_agent == 'entrevista'
    orch.handle(s, 'sim')
    assert s.flow.interview_answers == {'renda_mensal': 6000} and not s.flow.pending_interview_answers
    assert s.credit_proposal == proposal and csv(orch.repo) == before
    check_protocol(orch, s)


def test_question_is_not_restored_after_stage_changes_and_fallback_does_not_replace_it():
    s = Session(active_agent='entrevista', authenticated=True)
    FlowController.start_interview(s)
    recorder = ReplyRecorder()
    recorder.say(s, 'Qual é a renda mensal?')
    saved = copy.deepcopy(s.agent_questions)
    recorder.say(s, 'Não consegui concluir a resposta. Pode aguardar?', track=False)
    assert s.agent_questions == saved
    s.history = [{'role': 'user', 'content': 'assunto recente'}]
    assert select_history('entrevista', s, s.history)[0]['content'] == 'Qual é a renda mensal?'
    s.flow.interview_answers = {'renda_mensal': 6000}
    assert select_history('entrevista', s, s.history) == s.history
    s.flow.interview_answers.clear()
    s.last_request_id, s.desired_limit = 9, 50000
    assert select_history('entrevista', s, s.history) == s.history  # Outro pedido de origem representa outra etapa de crédito.


@pytest.mark.parametrize('mode', ['native', 'structured'])
def test_model_requesting_credentials_during_interview_is_blocked_and_preserves_stage(repo_ana_620, mode):
    orch = make(repo_ana_620, [message('Informe seu CPF e sua data de nascimento para continuar.')], mode)
    s = logged_in(orch, agent='entrevista')
    FlowController.start_interview(s)
    s.flow.interview_answers = {'renda_mensal': 6000}
    before = csv(orch.repo)
    text = orch.handle(s, 'Vamos continuar a entrevista')
    assert 'identidade já está confirmada' in text
    assert 'Cliente autenticado' in systems(orch.client.requests[0])
    assert s.authenticated and s.auth_failures == 0 and s.active_agent == 'entrevista'
    assert s.flow.interview_answers == {'renda_mensal': 6000} and csv(orch.repo) == before
    assert s.last_failure_category == 'ungrounded_response'
    check_protocol(orch, s)


@pytest.mark.parametrize('agent', ['triagem', 'credito', 'entrevista', 'cambio'])
@pytest.mark.parametrize('fact_turn', [4, 5])
def test_current_or_unsettled_results_and_uncertain_operation_are_never_hidden(agent, fact_turn):
    s = Session(authenticated=True, active_agent=agent, turn=5)
    s.credit_operation = {'amount': 8000, 'receipt': {'cpf': 'private-identifier', 'request_id': 9}}
    s.tool_facts['solicitar_aumento_limite'] = {'turn': fact_turn, 'result': {'status_pedido': 'aprovado',
        'novo_limite_solicitado': 8000, 'limite_aplicado': False}}
    before = copy.deepcopy((s.tool_facts, s.credit_operation))
    context = AGENTS[agent].session_context(s)
    assert 'resultado incerto' in context and '8000' in context and '"limite_aplicado":false' in context
    assert 'financeiro pendente de conclusão' in context
    assert 'private-identifier' not in context
    assert (s.tool_facts, s.credit_operation) == before
