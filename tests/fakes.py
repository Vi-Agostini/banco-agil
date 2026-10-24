import copy
import json
from types import SimpleNamespace as NS


def reply(text=None, calls=()):
    """Resposta simulada do LLM. `args` pode ser dict (vira JSON) ou str crua."""
    tcs = [
        NS(id=f"c{i}", function=NS(name=n, arguments=a if isinstance(a, str) else json.dumps(a)))
        for i, (n, a) in enumerate(calls)
    ] or None
    return NS(choices=[NS(message=NS(content=text, tool_calls=tcs))])


class FakeLLM:
    def __init__(self, script):
        self.script = list(script)
        self.chat = NS(completions=NS(create=self._create))
        self.requests = []  # Cópia das mensagens enviadas em cada chamada.

    def _create(self, **kw):
        self.requests.append(copy.deepcopy(kw["messages"]))
        return self.script.pop(0)

    def assert_exhausted(self):
        assert not self.script, f"LLM script has {len(self.script)} unused response(s)"


def assert_history_valid(history):
    """Toda tool_call do assistente tem exatamente uma resposta 'tool' logo em seguida."""
    i = 0
    while i < len(history):
        m = history[i]
        if m["role"] == "assistant":
            assert m.get("content") or m.get("tool_calls"), "mensagem vazia do assistente"
            ids = [c["id"] for c in m.get("tool_calls") or []]
            answered = [h["tool_call_id"] for h in history[i + 1 : i + 1 + len(ids)]]
            assert answered == ids, f"tool_calls sem resposta: {ids} vs {answered}"
            i += len(ids)
        i += 1


class Clock:
    def __init__(self):
        self.now = 0
        self.waits = []
    def __call__(self):
        return self.now
    def sleep(self, seconds):
        assert seconds > 0
        self.waits.append(seconds)
        self.now += seconds


class ScriptedClient:
    def __init__(self, script):
        self.script = list(script)
        self.requests = []
        self.chat = NS(completions=NS(create=self.create))

    def create(self, **kwargs):
        self.requests.append(copy.deepcopy(kwargs))
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class RateLimited(RuntimeError):
    status_code = 429

    def __init__(self, retry_after: str | None = None):
        super().__init__("provider detail must stay private")
        self.headers = {} if retry_after is None else {"Retry-After": retry_after}


class ToolUseFailed(RuntimeError):
    status_code = 400

    def __init__(self, code: str | None = "tool_use_failed"):
        super().__init__("provider detail must stay private")
        self.body = None if code is None else {
            "error": {"type": "invalid_request_error", "code": code, "param": None},
        }


def failure(code, **fields):
    return type('ProviderError',(Exception,),{'status_code':400,'body':{'error':{'code':code,**fields}}})('private-token')
