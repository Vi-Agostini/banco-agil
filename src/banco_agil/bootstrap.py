from __future__ import annotations

import os
import json
import math
from urllib.parse import urlparse
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from .exchange import ExchangeService
from .rate_limit import GovernedClient, shared_limiter
from .llm import configured_rate_limit_wait_seconds
from .orchestrator import Orchestrator, setup_logging
from .repository import BankRepository, DataError
from .structured_output import validate_mode

ROOT = Path(__file__).resolve().parents[2]


class ConfigError(Exception):
    pass


def build_orchestrator() -> Orchestrator:
    load_dotenv(ROOT / ".env")
    try:
        llm_mode = validate_mode(os.getenv("LLM_OUTPUT_MODE", "native"))
    except ValueError as error:
        raise ConfigError(str(error)) from None
    key = os.getenv("LLM_API_KEY")
    if not key:
        raise ConfigError("Defina LLM_API_KEY no arquivo .env (veja .env.example).")
    setup_logging(ROOT / "logs")
    client = OpenAI(
        api_key=key,
        base_url=os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1"),
        timeout=float(os.getenv("LLM_TIMEOUT", "30")),
        max_retries=0,
    )
    if urlparse(os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")).hostname == "api.groq.com":
        try:
            organization = os.getenv("GROQ_ORGANIZATION", "default")
            model = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")
            limits = json.loads(os.getenv("GROQ_RATE_LIMITS_JSON", "{}"))
            settings = limits.get(organization, {}).get(model, {})
            rpm = settings.get("rpm", int(os.getenv("GROQ_RPM", "30")))
            tpm = settings.get("tpm", int(os.getenv("GROQ_TPM", "8000")))
            wait = float(os.getenv("GROQ_ADMISSION_TIMEOUT_SECONDS", "60"))
            if not math.isfinite(wait) or not 0 <= wait <= 120:
                raise ValueError("invalid admission timeout")
            output_tokens = settings.get("output_tokens", int(os.getenv("GROQ_MAX_COMPLETION_TOKENS", "2048")))
            effort = os.getenv("GROQ_REASONING_EFFORT", "low").strip() or None
            client = GovernedClient(client, shared_limiter(organization, model, rpm, tpm), wait, output_tokens,
                                    reasoning_effort=effort)
        except (ValueError, TypeError, AttributeError):
            raise ConfigError("Configuração inválida do limitador Groq.") from None
    data_dir = Path(os.getenv("DATA_DIR") or ROOT / "data")
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    repo = BankRepository(data_dir)
    orch = Orchestrator(
        repo, ExchangeService(), client, os.getenv("LLM_MODEL", "openai/gpt-oss-20b"),
        max_llm_retries=min(1, max(0, int(os.getenv("LLM_MAX_RETRIES", "0")))),
        test_id=os.getenv("BANCO_AGIL_TEST_ID"),
        max_rate_limit_wait_seconds=configured_rate_limit_wait_seconds(
            os.getenv("LLM_MAX_RATE_LIMIT_WAIT_SECONDS"),
        ),
        llm_mode=llm_mode,
    )
    try:
        orch.credit.reconcile()  # completa solicitações interrompidas por queda anterior
    except DataError:
        pass  # falha já registrada; o atendimento segue e tenta de novo por cliente
    return orch
