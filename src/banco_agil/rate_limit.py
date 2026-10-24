"""Contabilização compartilhada e cooldown após erros do provedor."""
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
import math
import re
from threading import Lock
import time


class CapacityError(RuntimeError):
    pass


def daily_token_limit(error) -> bool:
    """Reconhece o esgotamento explícito da cota diária de tokens."""
    body = getattr(error, "body", None)
    detail = body.get("error", body) if isinstance(body, Mapping) else None
    return bool(getattr(error, "status_code", None) == 429 and isinstance(detail, Mapping)
                and detail.get("code") == "rate_limit_exceeded" and detail.get("type") == "tokens"
                and detail.get("param") in (None, "")
                and isinstance(detail.get("message"), str)
                and re.search(r"\btokens\s+per\s+day\s*\(TPD\)", detail["message"][:8192], re.I))


def duration(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        parts = re.findall(r'(\d+(?:\.\d+)?)(ms|h|m|s)', str(value))
        if not parts or ''.join(n + unit for n, unit in parts) != str(value):
            return None
        number = sum(float(n) * {'ms':.001, 's':1, 'm':60, 'h':3600}[u] for n,u in parts)
    return number if math.isfinite(number) and number >= 0 else None


@dataclass(eq=False)
class Reservation:
    at: float
    tokens: int
    pending: bool = True


class RateLimiter:
    def __init__(self, rpm, tpm, *, clock=time.monotonic, sleep=time.sleep):
        if type(rpm) is not int or type(tpm) is not int or min(rpm, tpm) <= 0:
            raise ValueError('RPM/TPM devem ser inteiros positivos.')
        self.rpm, self.tpm = rpm, tpm
        self.clock, self.sleep = clock, sleep
        self.lock = Lock()
        self.entries = deque()
        self.blocked_until = 0.0
        self.remote = {}  # Dimensão -> [capacidade restante, instante de liberação no relógio monotônico].

    def defer(self, seconds):
        with self.lock:
            self.blocked_until = max(self.blocked_until, self.clock() + seconds)

    def acquire(self, tokens, deadline):
        if tokens > self.tpm:
            raise CapacityError('request_exceeds_tpm')
        while True:
            with self.lock:
                now = self.clock()
                if tokens > self.tpm:
                    raise CapacityError("request_exceeds_tpm")
                # Chamadas HTTP pendentes mantêm a reserva mesmo na virada da janela.
                self.entries = deque(e for e in self.entries if e.pending or e.at + 60 > now)
                # RPM, TPM e cabeçalhos de sucesso servem para contabilização.
                # Só um 429 observado abre um período de espera.
                for dimension, (_, reset) in list(self.remote.items()):
                    if reset <= now:
                        del self.remote[dimension]
                delay = max(0, self.blocked_until - now)
                if delay == 0:
                    entry = Reservation(now, tokens)
                    self.entries.append(entry)
                    for dimension, state in self.remote.items():
                        state[0] -= tokens if dimension == 'tokens' else 1
                    return entry
                if now + delay > deadline:
                    raise CapacityError('admission_deadline')
            self.sleep(delay)  # Libera a trava antes de esperar.

    def settle(self, entry, usage=None, headers=None):
        with self.lock:
            now = self.clock()
            entry.pending = False
            # Conta desde a conclusão para preservar reservas de chamadas lentas ou incertas.
            entry.at = now
            total = usage.get('total_tokens') if isinstance(usage, dict) else getattr(usage, 'total_tokens', None)
            if type(total) is int and total >= 0:
                entry.tokens = total
            headers = {k.lower():v for k,v in (headers or {}).items()}
            limit = duration(headers.get('x-ratelimit-limit-tokens'))
            if limit and limit < self.tpm:
                self.tpm = int(limit)
            for dimension in ('tokens', 'requests'):
                remaining = duration(headers.get('x-ratelimit-remaining-' + dimension))
                reset = duration(headers.get('x-ratelimit-reset-' + dimension))
                if remaining is None or reset is None:
                    continue
                # Cabeçalhos podem chegar fora de ordem; não devolve capacidade já consumida.
                # Desconta chamadas pendentes que podem não estar incluídas nesse cabeçalho.
                remaining -= sum(e.tokens if dimension == 'tokens' else 1 for e in self.entries if e.pending)
                old = self.remote.get(dimension)
                if old and old[1] > now:
                    remaining, reset = min(remaining, old[0]), max(reset, old[1] - now)
                self.remote[dimension] = [remaining, now + reset]


_registry = {}
_registry_lock = Lock()


def shared_limiter(organization, model, rpm, tpm):
    if type(rpm) is not int or type(tpm) is not int or min(rpm, tpm) <= 0:
        raise ValueError("RPM/TPM devem ser inteiros positivos.")
    with _registry_lock:
        key = (organization, model)
        if key not in _registry:
            _registry[key] = RateLimiter(rpm, tpm)
        limiter = _registry[key]
        with limiter.lock:
            # Outro cliente não pode aumentar os limites de um contador existente.
            limiter.rpm = min(limiter.rpm, rpm)
            limiter.tpm = min(limiter.tpm, tpm)
        return limiter


class GovernedClient:
    """Cliente com contabilização e cooldown reativo, sem retentativas próprias."""
    def __init__(self, client, limiter, max_wait=60, output_tokens=2048, *, reasoning_effort=None):
        if type(output_tokens) is not int or not 256 <= output_tokens <= 8192:
            raise ValueError("Reserva de saída deve estar entre 256 e 8192 tokens.")
        self.client, self.limiter, self.max_wait = client, limiter, max_wait
        self.output_tokens = output_tokens
        if reasoning_effort not in {None, "low", "medium", "high"}:
            raise ValueError("Esforço de raciocínio inválido.")
        self.reasoning_effort = reasoning_effort

    def create(self, request, estimated_input, deadline, *, timings=None):
        timings = timings if timings is not None else {}
        timings.update(admission_wait_ms=0, http_ms=0, http_attempted=False)
        output_reserve = self.output_tokens
        started = self.limiter.clock()
        try:
            entry = self.limiter.acquire(estimated_input + output_reserve, deadline)
        finally:
            timings["admission_wait_ms"] = round((self.limiter.clock() - started) * 1000)
        started = self.limiter.clock()
        try:
            # O teto de saída HTTP corresponde à reserva de tokens.
            request = dict(request, max_completion_tokens=output_reserve)
            if (self.reasoning_effort is not None
                    and request.get("model") in {"openai/gpt-oss-20b", "openai/gpt-oss-120b"}):
                request["reasoning_effort"] = self.reasoning_effort
            timings["http_attempted"] = True
            raw = self.client.chat.completions.with_raw_response.create(**request)
            response = raw.parse()
        except Exception as error:
            self.limiter.settle(entry, headers=getattr(getattr(error, 'response', None), 'headers', None))
            raise
        finally:
            timings["http_ms"] = round((self.limiter.clock() - started) * 1000)
        self.limiter.settle(entry, getattr(response, 'usage', None), raw.headers)
        return response
