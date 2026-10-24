"""Respostas HTTP em memória e cópias do estado da conversa e das operações."""
import copy
from contextlib import ExitStack
import json
from types import SimpleNamespace as NS

import httpx2 as httpx
from openai import OpenAI
import pytest
import requests

from fakes import Clock


MODEL = "openai/gpt-oss-120b"
OTHER_MODEL = "openai/gpt-oss-20b"


def daily(*, retry_after=None, wait="9m6.48s", **overrides):
    detail = {
        "code": "rate_limit_exceeded", "type": "tokens", "param": "",
        "message": (f"Rate limit reached for model `{MODEL}` on tokens per day (TPD): "
                    f"Limit 200000, Used 199900, Requested 3000. Please try again in {wait}."),
    }
    detail.update(overrides)
    headers = {} if retry_after is None else {"Retry-After": retry_after}
    return 429, {"error": detail}, headers


def error(status, code, text="local-only", *, kind="tokens", headers=None, **fields):
    return status, {"error": {"code": code, "type": kind, "message": text, "param": "", **fields}}, headers or {}


@pytest.fixture
def local_http(monkeypatch):
    """Usa a serialização real do SDK, sem abrir conexões de rede."""
    clock = Clock()
    monkeypatch.setattr("banco_agil.llm.time.sleep", clock.sleep)
    with ExitStack() as stack:
        def create(script):
            pending, requests = list(script), []

            def sink(request):
                sent = json.loads(request.content)
                requests.append(sent)
                assert pending, "Unexpected HTTP attempt: retry or replay exceeded the local script"
                value = pending.pop(0)
                if isinstance(value, Exception):
                    raise value
                if isinstance(value, tuple):
                    status, body, headers = value
                    return httpx.Response(status, json=body, headers=headers)
                finish, calls = "stop", None
                if "_raw" in value:
                    content = value["_raw"]
                    finish = value.get("_finish", "stop")
                elif sent.get("tool_choice", {}).get("function", {}).get("name") == "banco_agil_resposta":
                    content, finish = None, "tool_calls"
                    calls = [{"id": "local-envelope-" + str(len(requests)), "type": "function",
                              "function": {"name": "banco_agil_resposta",
                                           "arguments": json.dumps({"resposta": value}, ensure_ascii=False)}}]
                elif "response_format" in sent:
                    content = json.dumps({"resposta": value}, ensure_ascii=False)
                elif value["tipo"] == "acao":
                    content, finish = None, "tool_calls"
                    proposal = value["acao"]
                    args = {k: v for k, v in (proposal["argumentos"] or {}).items() if v is not None}
                    calls = [{"id": "local-call-" + str(len(requests)), "type": "function",
                              "function": {"name": proposal["nome"], "arguments": json.dumps(args)}}]
                else:
                    content = value["mensagem"]
                body = {"id": "local-provider", "object": "chat.completion", "created": 0,
                        "model": sent["model"], "usage": {"prompt_tokens": 1200,
                        "completion_tokens": 100, "total_tokens": 1300},
                        "choices": [{"index": 0, "finish_reason": finish,
                        "message": {"role": "assistant", "content": content}}]}
                if calls:
                    body["choices"][0]["message"]["tool_calls"] = calls
                return httpx.Response(200, json=body)

            sdk = stack.enter_context(OpenAI(api_key="local-only", base_url="http://memory.invalid/v1",
                max_retries=0, http_client=httpx.Client(transport=httpx.MockTransport(sink), trust_env=False)))
            return NS(client=sdk, requests=requests, pending=pending)

        yield create, clock


def preserved_state(s):
    return copy.deepcopy({name: getattr(s, name) for name in (
        "history", "flow", "credit_proposal", "credit_operation", "agent_questions", "tool_facts",
        "correlation_id", "turn", "active_agent", "authenticated", "cpf", "nome", "desired_limit",
        "last_request_status", "last_request_id", "automatic_reanalysis_source_id")})


class Resp:
    def __init__(self, data=None, status=200, bad_json=False):
        self.data, self.status, self.bad_json = data, status, bad_json

    def raise_for_status(self):
        if self.status >= 400:
            raise requests.HTTPError(f"{self.status}")

    def json(self):
        if self.bad_json:
            raise ValueError("no json")
        return self.data
