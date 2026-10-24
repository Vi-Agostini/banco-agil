"""Mensagens, clientes e verificações dos protocolos simulados."""
import copy
import json
from types import SimpleNamespace as NS
from banco_agil.agents import AGENTS
from banco_agil.exchange import ExchangeService
from banco_agil.orchestrator import Orchestrator
from fakes import assert_history_valid


def message(text):
    return {"tipo": "mensagem", "mensagem": text, "acao": None}


def action(name, **args):
    # O formato estrito exige todas as chaves opcionais da entrevista, aceitando null.
    tool = next(t for a in AGENTS.values() for t in a.tools if t.name == name)
    args = dict(args)
    for key in tool.parameters["properties"]:
        if key not in tool.parameters.get("required", ()):
            args.setdefault(key, None)
    return {"tipo": "acao", "mensagem": None, "acao": {"nome": name,
        "argumentos": args if tool.parameters["properties"] else None}}


def completion(value, *, mode="structured", finish="stop"):
    content, calls = json.dumps({"resposta": value}, ensure_ascii=False) if isinstance(value, dict) else value, None
    if mode == "strict_tool":
        finish, content, calls = "tool_calls", None, [NS(id="envelope-1", type="function", function=NS(
            name="banco_agil_resposta", arguments=content))]
    elif mode == "native":
        if value["tipo"] == "mensagem":
            content = value["mensagem"]
        else:
            proposal = value["acao"]
            args = {k: v for k, v in (proposal["argumentos"] or {}).items() if v is not None}
            content = None
            calls = [NS(id="native-1", function=NS(name=proposal["nome"], arguments=json.dumps(args)))]
    return NS(choices=[NS(finish_reason=finish, message=NS(content=content, tool_calls=calls))])


def strict_spec(request):
    """Confere o transporte estrito e retorna o schema completo do envelope."""
    if "response_format" in request:
        assert not {"tools", "tool_choice", "parallel_tool_calls"}.intersection(request)
        spec = request["response_format"]["json_schema"]
    else:
        assert len(request["tools"]) == 1
        function = request["tools"][0]["function"]
        assert function["name"] == "banco_agil_resposta"
        assert request["tool_choice"] == {"type": "function", "function": {"name": function["name"]}}
        assert request["parallel_tool_calls"] is False
        spec = {"name": function["name"], "strict": function["strict"], "schema": function["parameters"]}
    assert spec["strict"] is True
    return spec


class ScriptedClient:
    def __init__(self, script):
        self.script, self.requests = list(script), []
        self.chat = NS(completions=NS(create=self.create))

    def create(self, **request):
        self.requests.append(copy.deepcopy(request))
        value = self.script.pop(0)
        if isinstance(value, Exception):
            raise value
        if not isinstance(value, dict):
            return value
        mode = ("structured" if "response_format" in request else "strict_tool"
                if request.get("tool_choice", {}).get("function", {}).get("name") == "banco_agil_resposta" else "native")
        return completion(value, mode=mode)


def make(repo, script, mode="structured", **kwargs):
    return Orchestrator(repo, ExchangeService(), ScriptedClient(script), "openai/gpt-oss-20b",
                        llm_mode=mode, **kwargs)


def check_protocol(orch, session):
    assert not orch.client.script
    assert_history_valid(session.history)
    for req in orch.client.requests:
        if orch.llm_mode == "structured":
            strict_spec(req)
            assert all(m["role"] in {"system", "assistant", "user"} for m in req["messages"])
            assert not any(m.get("tool_calls") or "tool_call_id" in m for m in req["messages"])
        else:
            assert "response_format" not in req and "parallel_tool_calls" not in req
    if orch.llm_mode == "structured":
        assert not any(m["role"] == "tool" or m.get("tool_calls") for m in session.history)


CATALOG = {tool.name: tool for agent in AGENTS.values() for tool in agent.tools}


def sample(schema):
    if "anyOf" in schema: return sample(schema["anyOf"][0])
    if "enum" in schema: return schema["enum"][0]
    kind=schema["type"]
    if kind=="object": return {k:sample(v) for k,v in schema["properties"].items()}
    return {"null":None,"number":1,"integer":1,"string":"fixture","boolean":False}[kind]
