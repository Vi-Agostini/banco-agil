"""Retomada da interface e consumo limitado do Groq com HTTP e relógio simulados."""
import copy
import json
import logging
from types import SimpleNamespace as NS

import httpx2 as httpx
import pytest
from openai import OpenAI

from banco_agil import bootstrap
from banco_agil.flow import FlowController
from banco_agil.llm import call_llm
from banco_agil.agents import AGENTS
from banco_agil.rate_limit import GovernedClient, RateLimiter
from banco_agil.session import Session
from banco_agil.structured_output import record_result
from fakes import failure, RateLimited, Clock
from orchestrator_fakes import logged_in
from structured_fakes import action, message, make, check_protocol
from support import snapshot_csv as csv


@pytest.mark.parametrize("mode", ["native", "structured"])
def test_interview_mixed_failures_show_preserved_pending_data_naturally_and_resume(repo_ana_620, monkeypatch, mode):
    orch = make(repo_ana_620, [action("registrar_respostas_entrevista", tipo_emprego="autônomo"),
        RateLimited("4.905"), failure("json_validate_failed"), RateLimited("4.905"),
        message("Quais são suas despesas fixas mensais?")], mode)
    session = logged_in(orch, agent="entrevista")
    FlowController.start_interview(session)
    session.flow.interview_answers = {"renda_mensal": 90000}
    before = csv(orch.repo)
    waits = []
    monkeypatch.setattr("banco_agil.llm.time.sleep", waits.append)

    response = orch.handle(session, "autonomo")

    assert "alta demanda" in response and "- tipo de emprego: autônomo" in response
    assert "ainda aguardam sua confirmação" in response and "entrevista continua ativa" in response
    assert not any(value in response for value in ("emprego: autonomo", "HTTP", "JSON"))
    assert "Dados para confirmação:\n\n- tipo de emprego: autônomo" in response
    assert session.flow.interview_answers == {"renda_mensal": 90000}
    assert session.flow.pending_interview_answers == {"tipo_emprego": "autonomo"}
    assert session.active_agent == "entrevista" and not session.ended
    assert csv(orch.repo) == before and waits == [20.0] and len(orch.client.requests) == 4

    assert orch.handle(session, "isso") == "Quais são suas despesas fixas mensais?"
    assert session.flow.interview_answers == {"renda_mensal": 90000, "tipo_emprego": "autonomo"}
    assert not session.flow.pending_interview_answers and csv(orch.repo) == before
    check_protocol(orch, session)


@pytest.mark.parametrize("mode", ["native", "structured"])
def test_successful_tool_repetition_stops_without_further_generation_or_invalid_input_count(repo_ana_620, mode, caplog):
    orch = make(repo_ana_620, [action("registrar_entrada_invalida"), action("registrar_entrada_invalida")], mode)
    session = logged_in(orch, agent="cambio")
    before = csv(orch.repo)
    with caplog.at_level(logging.INFO, logger="banco_agil"):
        orch.handle(session, "texto incompreensível simulado")
    assert len(orch.client.requests) == 2 and session.flow.unclear_count == 1
    assert session.last_failure_category == "progress_stalled" and not session.ended
    assert csv(orch.repo) == before
    events = [json.loads(r.message) for r in caplog.records if r.message.startswith('{')]
    stop = next(e for e in events if e['event'] == 'pipeline_stop')
    assert stop['fallback_reason'] == 'repeated_tool_call' and stop['turn'] == session.turn
    assert 'texto incompreensível' not in caplog.text
    check_protocol(orch, session)


@pytest.mark.parametrize("mode", ["native", "structured"])
def test_equivalent_argument_order_does_not_restart_proposal_collection(repo_ana_620, mode):
    orch = make(repo_ana_620, [action("registrar_intencao_credito", valor=8000, modalidade="limite_total"),
                             action("registrar_intencao_credito", modalidade="limite_total", valor=8000)], mode)
    session = logged_in(orch, agent="credito")
    before = csv(orch.repo)
    response = orch.handle(session, "oito mil")
    assert session.last_failure_category == "progress_stalled" and len(orch.client.requests) == 2
    assert not session.credit_proposal['confirmada'] and 'me envie **confirmo** para continuar' in response
    assert csv(orch.repo) == before and not session.ended
    check_protocol(orch, session)


@pytest.mark.parametrize("mode", ["native", "structured"])
def test_static_prefix_survives_state_change_without_losing_trusted_context(repo_ana_620, mode):
    orch = make(repo_ana_620, [message("Podemos conversar sobre os dados que faltam."),
                             message("Vamos continuar pelos dados ainda ausentes.")], mode)
    session = logged_in(orch, agent="entrevista")
    FlowController.start_interview(session)
    session.flow.interview_answers = {'renda_mensal': 90000}
    orch.handle(session, 'Qual é a próxima etapa?')
    session.flow.interview_answers['tipo_emprego'] = 'autonomo'
    record_result(session, 'confirmacao_entrevista', {'ok': True, 'confirmacao_pendente': False,
        'respostas_confirmadas': dict(session.flow.interview_answers)})
    original_receipt = copy.deepcopy(session.history[-1])
    orch.handle(session, 'Gostaria de continuar.')
    first, second = (r['messages'] for r in orch.client.requests)
    assert first[0] == second[0] and first[0]['role'] == 'system'
    assert '90000' not in first[0]['content'] and 'autonomo' not in first[0]['content']
    trusted_first = '\n'.join(m['content'] for m in first[1:] if m['role'] == 'system')
    trusted_second = '\n'.join(m['content'] for m in second[1:] if m['role'] == 'system')
    assert '90000' in trusted_first and 'autonomo' in trusted_second
    assert trusted_second.count('90000') == 1
    assert 'confirmacao_entrevista' in trusted_second and 'confirmacao_pendente' in trusted_second
    assert original_receipt in session.history  # O contexto completo da sessão permanece intacto.
    assert any(m['role'] == 'user' and m['content'] == 'Qual é a próxima etapa?' for m in second)
    check_protocol(orch, session)


def test_state_compaction_keeps_financial_receipts_and_differing_proposals():
    from banco_agil.llm import _build_payload
    session = Session(active_agent='credito', authenticated=True)
    original = {'status_pedido': 'aprovado', 'limite_aplicado': True, 'novo_limite_solicitado': 8000}
    record_result(session, 'solicitar_aumento_limite', original)
    record_result(session, 'registrar_intencao_credito', {'proposta': {'valor': 6000}})
    session.credit_proposal = {'valor': 9000}
    before = copy.deepcopy(session.history)
    payload = _build_payload(AGENTS['credito'], session)
    assert payload[2:] == before and session.history == before


@pytest.mark.parametrize("mode", ["native", "structured"])
@pytest.mark.parametrize("model,effort,expected", [
    ("openai/gpt-oss-20b", "low", "low"), ("openai/gpt-oss-120b", "low", "low"),
    ("openai/gpt-oss-120b", "medium", "medium"), ("other-model", "low", None),
    ("openai/gpt-oss-120b", None, None),
])
def test_reasoning_setting_on_actual_sdk_wire_keeps_schema_budget_and_sanitized_metrics(mode, model, effort, expected, caplog):
    captured = []
    text = 'Podemos continuar.'
    def sink(request):
        body = json.loads(request.content)
        captured.append(body)
        content = json.dumps({'resposta': message(text)}) if mode == 'structured' else text
        return httpx.Response(200, json={'id': 'local', 'object': 'chat.completion', 'created': 0,
            'model': model, 'choices': [{'index': 0, 'finish_reason': 'stop',
                'message': {'role': 'assistant', 'content': content}}],
            'usage': {'prompt_tokens': 1200, 'completion_tokens': 33, 'total_tokens': 1233,
                      'prompt_tokens_details': {'cached_tokens': 1024}}})
    clock = Clock()
    limiter = RateLimiter(30, 30000, clock=clock, sleep=clock.sleep)
    with caplog.at_level(logging.INFO, logger='banco_agil'):
        with OpenAI(api_key='local', base_url='http://memory.invalid/v1', max_retries=0,
                    http_client=httpx.Client(transport=httpx.MockTransport(sink))) as sdk:
            client = GovernedClient(sdk, limiter, output_tokens=2048, reasoning_effort=effort)
            result = call_llm(client, model, AGENTS['triagem'], Session(), llm_mode=mode)
    assert result.content == text and len(captured) == 1 and not clock.waits
    assert captured[0].get('reasoning_effort') == expected
    assert captured[0]['max_completion_tokens'] == 2048
    if mode == 'structured':
        assert captured[0]['response_format']['json_schema']['strict'] is True and 'tools' not in captured[0]
    metrics = next(json.loads(r.message) for r in caplog.records if r.message.startswith('{'))
    assert (metrics['input_tokens'], metrics['output_tokens'], metrics['cached_input_tokens']) == (1200, 33, 1024)
    assert text not in caplog.text


@pytest.mark.parametrize('configured,expected', [(None, 'low'), ('medium', 'medium'), ('', None)])
def test_bootstrap_reasoning_default_and_operator_override(repo_ana_620, monkeypatch, configured, expected):
    captured = {}
    def orchestrator(repo, fx, client, model, **kwargs):
        captured['client'] = client
        return NS(credit=NS(reconcile=lambda: None))
    monkeypatch.setattr(bootstrap, 'OpenAI', lambda **kw: object())
    monkeypatch.setattr(bootstrap, 'load_dotenv', lambda *a: None)
    monkeypatch.setattr(bootstrap, 'setup_logging', lambda *a: None)
    monkeypatch.setattr(bootstrap, 'Orchestrator', orchestrator)
    monkeypatch.setenv('LLM_API_KEY', 'local')
    monkeypatch.setenv('LLM_BASE_URL', 'https://api.groq.com/openai/v1')
    monkeypatch.setenv('LLM_MODEL', 'openai/gpt-oss-120b')
    monkeypatch.setenv('DATA_DIR', str(repo_ana_620.dir))
    monkeypatch.setenv('GROQ_RATE_LIMITS_JSON', '{}')
    if configured is None:
        monkeypatch.delenv('GROQ_REASONING_EFFORT', raising=False)
    else:
        monkeypatch.setenv('GROQ_REASONING_EFFORT', configured)
    bootstrap.build_orchestrator()
    assert captured['client'].reasoning_effort == expected


def test_invalid_reasoning_setting_fails_before_http():
    with pytest.raises(ValueError, match='raciocínio'):
        GovernedClient(object(), RateLimiter(30, 8000), reasoning_effort='unsupported')
