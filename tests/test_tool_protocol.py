"""Regressões do protocolo Chat Completions após chamadas de ferramenta."""
from __future__ import annotations

import pytest

from banco_agil.exchange import ExchangeService
from banco_agil.orchestrator import Orchestrator
from fakes import FakeLLM, reply


def make(repo, script):
    return Orchestrator(repo, ExchangeService(), FakeLLM(script), "fake")


def logged_in(orch, agent="triagem"):
    session = orch.new_session()
    session.authenticated = True
    session.cpf = "11144477735"
    session.nome = "Ana Souza"
    session.active_agent = agent
    return session


def protocol_messages(request):
    return [message for message in request if message["role"] != "system"]


def test_tool_call_without_text_preserves_null_and_tool_name(repo):
    orch = make(repo, [
        reply(calls=[("transferir_para_agente", {"destino": "credito"})]),
        reply("Posso consultar seu limite."),
    ])
    session = logged_in(orch)

    # Um assunto explícito de crédito é encaminhado localmente. Aqui o
    # encaminhamento deve partir do LLM para exercitar o replay nativo.
    orch.handle(session, "continue o atendimento")

    replay = protocol_messages(orch.client.requests[1])
    assistant = next(message for message in replay if message.get("tool_calls"))
    tool = next(message for message in replay if message["role"] == "tool")
    assert assistant["content"] is None
    assert tool == {
        "role": "tool",
        "tool_call_id": assistant["tool_calls"][0]["id"],
        "name": "transferir_para_agente",
        "content": '{"ok": true, "instrucao": "Continue o atendimento normalmente, sem mencionar transição."}',
    }
    assert session.active_agent == "credito"
    orch.client.assert_exhausted()


def test_tool_call_with_text_preserves_text_in_replayed_history(repo):
    orch = make(repo, [
        reply("Vou consultar isso agora.", calls=[("transferir_para_agente", {"destino": "credito"})]),
        reply("Seu limite foi localizado."),
    ])
    session = logged_in(orch)

    orch.handle(session, "continue o atendimento")

    assistant = next(message for message in protocol_messages(orch.client.requests[1]) if message.get("tool_calls"))
    assert assistant["content"] == "Vou consultar isso agora."
    orch.client.assert_exhausted()


def test_each_tool_result_keeps_its_matching_id_and_name(repo):
    orch = make(repo, [
        reply(calls=[("consultar_limite", {}), ("registrar_entrada_invalida", {})]),
        reply("Concluído."),
    ])
    session = logged_in(orch, agent="credito")

    orch.handle(session, "verifique meu limite")

    replay = protocol_messages(orch.client.requests[1])
    assistant = next(message for message in replay if message.get("tool_calls"))
    results = [message for message in replay if message["role"] == "tool"]
    assert [result["tool_call_id"] for result in results] == [call["id"] for call in assistant["tool_calls"]]
    assert [result["name"] for result in results] == ["consultar_limite", "registrar_entrada_invalida"]
    orch.client.assert_exhausted()


@pytest.mark.parametrize(("source", "destination"), [
    ("triagem", "credito"),
    ("credito", "entrevista"),
    ("entrevista", "credito"),
    ("credito", "cambio"),
])
def test_transfer_replays_a_valid_tool_protocol_for_destination_agent(repo, source, destination):
    orch = make(repo, [
        reply(calls=[("transferir_para_agente", {"destino": destination})]),
        reply("Continuidade confirmada."),
    ])
    session = logged_in(orch, agent=source)

    orch.handle(session, "continue o atendimento")

    assert session.active_agent == destination
    replay = protocol_messages(orch.client.requests[1])
    tool = next(message for message in replay if message["role"] == "tool")
    assert tool["name"] == "transferir_para_agente"
    assert tool["tool_call_id"]
    orch.client.assert_exhausted()


def test_privacy_is_preserved_in_replayed_tool_history_after_transfer(repo):
    orch = make(repo, [
        reply(calls=[("transferir_para_agente", {"destino": "credito"})]),
        reply(calls=[("consultar_limite", {})]),
        reply("Seu limite foi informado."),
    ])
    session = orch.new_session()

    orch.handle(session, "CPF 11144477735, nasci em 12/05/1990 e quero continuar o atendimento")

    sent = str(orch.client.requests)
    assert "11144477735" not in sent
    assert "12/05/1990" not in sent
    assert session.active_agent == "credito"
    assert session.authenticated
    replay = protocol_messages(orch.client.requests[2])
    assert [message["name"] for message in replay if message["role"] == "tool"] == [
        "transferir_para_agente", "consultar_limite",
    ]
    orch.client.assert_exhausted()
