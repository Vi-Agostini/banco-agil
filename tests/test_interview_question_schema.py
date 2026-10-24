"""Uma confirmação local que só precisa da próxima pergunta não oferece ações JSON."""
import copy
import json
from types import SimpleNamespace as NS

import httpx2 as httpx
from jsonschema import Draft202012Validator
from openai import OpenAI
import pytest

from banco_agil.agents import AGENTS
from banco_agil.exchange import ExchangeService
from banco_agil.flow import Awaiting, FlowController
from banco_agil.llm import _tool_schemas
from banco_agil.orchestrator import Orchestrator
from banco_agil.session import Session
from banco_agil.structured_output import StructuredOutputError, decode, response_format
from fakes import failure, Clock
from orchestrator_fakes import logged_in
from structured_fakes import action, check_protocol, make, message
from support import snapshot_csv as csv


ANSWERS = dict(renda_mensal=6000, tipo_emprego="formal", despesas_fixas=2000,
               num_dependentes=1, tem_dividas=False)


@pytest.mark.parametrize("count", [0, 1, 4, 5])
@pytest.mark.parametrize("mode", ["structured", "native"])
def test_question_phase_excludes_actions_only_when_a_question_is_the_remaining_work(mode, count):
    s = Session(active_agent="entrevista", authenticated=True, cpf="local-fixture", llm_mode=mode)
    FlowController.start_interview(s)
    s.flow.interview_answers = dict(list(ANSWERS.items())[:count])
    s.flow.interview_question_pending = True
    before = copy.deepcopy(s.flow)
    tools = _tool_schemas(AGENTS["entrevista"], s)
    if mode == "structured" and count < 5:
        assert tools == []
        fmt = response_format(tools)
        validator = Draft202012Validator(fmt["json_schema"]["schema"])
        assert validator.is_valid({"resposta": message("Qual dado deseja informar a seguir?")})
        # Schema e decoder proíbem qualquer ação nesta etapa, inclusive ferramentas
        # de controle que continuam disponíveis em outras etapas.
        for tool in AGENTS["entrevista"].tools:
            value = {"resposta": action(tool.name)}
            assert not validator.is_valid(value)
            with pytest.raises(StructuredOutputError):
                decode(NS(content=json.dumps(value)), tools, "stop")
        assert fmt["json_schema"]["strict"] is True
    else:
        names = {t["function"]["name"] for t in tools}
        assert {"encerrar_atendimento", "transferir_para_agente"} <= names
        assert ("registrar_entrevista_e_calcular_score" in names) is (count == 5)
    assert s.flow == before


def test_sdk_wire_after_acceptance_confirmation_new_answer_and_correction(repo_ana_620):
    script = [message("Qual é sua renda mensal?"),
        action("registrar_respostas_entrevista", renda_mensal=90000),
        message("Confirma a renda mensal de R$ 90.000,00?"),
        message("Qual é seu tipo de emprego?"),
        action("registrar_respostas_entrevista", tipo_emprego="formal"),
        message("Confirma o emprego formal?"),
        message("Quais são suas despesas fixas?"),
        action("registrar_respostas_entrevista", renda_mensal=80000, tem_dividas=False),
        message("Confirma renda de R$ 80.000,00 e ausência de dívidas?"),
        message("Quais são suas despesas fixas?")]
    requests = []

    def sink(request):
        wire = json.loads(request.content)
        requests.append(wire)
        assert wire["response_format"]["json_schema"]["strict"] is True and "tools" not in wire
        value = {"resposta": script.pop(0)}
        Draft202012Validator(wire["response_format"]["json_schema"]["schema"]).validate(value)
        return httpx.Response(200, json={"id": "local-question-schema", "object": "chat.completion",
            "created": 0, "model": "openai/gpt-oss-120b", "choices": [{"index": 0, "finish_reason": "stop",
            "message": {"role": "assistant", "content": json.dumps(value)}}]})

    with OpenAI(api_key="local-only", base_url="http://memory.invalid/v1", max_retries=0,
                http_client=httpx.Client(transport=httpx.MockTransport(sink), trust_env=False)) as client:
        orch = Orchestrator(repo_ana_620, ExchangeService(), client, "openai/gpt-oss-120b", llm_mode="structured")
        s = logged_in(orch, agent="credito")
        s.flow.awaiting = Awaiting.INTERVIEW_OFFER
        before = csv(orch.repo)
        assert orch.handle(s, "sim") == "Qual é sua renda mensal?"
        orch.handle(s, "90000")
        assert s.flow.pending_interview_answers == {"renda_mensal": 90000}
        assert orch.handle(s, "confirmo") == "Qual é seu tipo de emprego?"
        assert s.flow.interview_answers == {"renda_mensal": 90000}
        orch.handle(s, "formal")
        assert s.flow.pending_interview_answers == {"tipo_emprego": "formal"}
        assert orch.handle(s, "confirmo") == "Quais são suas despesas fixas?"
        orch.handle(s, "Na verdade, minha renda é oitenta mil e não tenho dívidas.")
        assert s.flow.pending_interview_answers == {"renda_mensal": 80000, "tem_dividas": False}
        assert orch.handle(s, "confirmo") == "Quais são suas despesas fixas?"
        assert s.flow.interview_answers == {"renda_mensal": 80000, "tipo_emprego": "formal", "tem_dividas": False}
        assert s.authenticated and s.active_agent == "entrevista" and not s.last_failure_category
        assert csv(orch.repo) == before and not script and len(requests) == 10
    for index, req in enumerate(requests):
        variants = req["response_format"]["json_schema"]["schema"]["properties"]["resposta"]["anyOf"]
        assert (len(variants) == 1) is (index in {0, 2, 3, 5, 6, 8, 9})
    # O histórico completo permanece na sessão; ações concluídas não viram
    # exemplos de nova saída. Os resultados locais preservam os fatos.
    assert any(m["role"] == "assistant" and "registrar_respostas_entrevista" in m["content"] for m in s.history)
    assert not any(m["role"] == "assistant" and "registrar_respostas_entrevista" in m["content"]
                   for m in requests[3]["messages"])
    assert any(m["role"] == "system" and "registrar_respostas_entrevista" in m["content"]
               for m in requests[3]["messages"])


@pytest.mark.parametrize("text,ends", [("quero encerrar todo o atendimento", True),
                                        ("quero cancelar somente a entrevista", False)])
def test_message_only_phase_does_not_prevent_explicit_local_closure(repo_ana_620, text, ends):
    orch = make(repo_ana_620, [message("Qual é seu tipo de emprego?")])
    s = logged_in(orch, agent="entrevista")
    FlowController.start_interview(s)
    s.flow.pending_interview_answers = {"renda_mensal": 90000}
    before = csv(orch.repo)
    orch.handle(s, "confirmo")
    orch.handle(s, text)
    assert len(orch.client.requests) == 1 and s.ended is ends
    assert (s.active_agent == "credito") is not ends
    assert csv(orch.repo) == before
    check_protocol(orch, s)


def test_exact_400_429_400_sequence_stays_bounded_and_next_answer_restores_collection(repo_ana_620, monkeypatch):
    clock = Clock()
    monkeypatch.setattr("banco_agil.llm.time.sleep", clock.sleep)
    # Até uma geração só de mensagem pode falhar. Não executa JSON rejeitado
    # nem usa retentativas ilimitadas para esconder erros repetidos.
    rejected = failure("json_validate_failed", failed_generation=json.dumps({"resposta":
        action("registrar_respostas_entrevista", renda_mensal=90000)}))
    throttle = type("ProviderError", (Exception,), {"status_code": 429,
        "headers": {"retry-after": "13.8825"}, "body": {"error": {"code": "rate_limit_exceeded"}}})("local-only")
    orch = make(repo_ana_620, [rejected, throttle, rejected,
        action("registrar_respostas_entrevista", tipo_emprego="formal"), message("Confirma o emprego formal?")])
    s = logged_in(orch, agent="entrevista")
    FlowController.start_interview(s)
    s.flow.pending_interview_answers = {"renda_mensal": 90000}
    s.credit_proposal = dict(valor=90000, modalidade="limite_total", novo_limite=90000, confirmada=False)
    proposal, before = copy.deepcopy(s.credit_proposal), csv(orch.repo)
    text = orch.handle(s, "confirmo")
    assert text.startswith("Seu emprego é formal, autônomo ou está desempregado?")
    assert "Não consegui processar essa resposta com segurança" not in text
    assert s.flow.interview_answers == {"renda_mensal": 90000} and not s.flow.pending_interview_answers
    assert len(orch.client.requests) == 3 and clock.waits == [20]
    for req in orch.client.requests:
        variants = req["response_format"]["json_schema"]["schema"]["properties"]["resposta"]["anyOf"]
        assert len(variants) == 1 and variants[0]["properties"]["acao"] == {"type": "null"}
    orch.handle(s, "formal")
    assert s.flow.pending_interview_answers == {"tipo_emprego": "formal"}
    assert s.flow.interview_answers == {"renda_mensal": 90000}
    assert s.credit_proposal == proposal and csv(orch.repo) == before and not orch.repo.list_requests()
    assert len(orch.client.requests) == 5 and clock.waits == [20]
    check_protocol(orch, s)
