"""Inicialização com SDK simulado e CSVs isolados, sem ler segredos ou usar rede."""
import json
from types import SimpleNamespace as NS
from uuid import uuid4

import pytest

from banco_agil import bootstrap
from banco_agil.rate_limit import GovernedClient


PRIMARY = "openai/gpt-oss-120b"
OTHER_MODEL = "openai/gpt-oss-20b"


@pytest.fixture
def configured(monkeypatch, repo_ana_620):
    keys = ("LLM_MODEL", "LLM_BASE_URL", "LLM_TIMEOUT", "LLM_MAX_RETRIES", "LLM_OUTPUT_MODE",
            "LLM_MAX_RATE_LIMIT_WAIT_SECONDS", "GROQ_RATE_LIMITS_JSON",
            "GROQ_RPM", "GROQ_TPM", "GROQ_MAX_COMPLETION_TOKENS", "GROQ_REASONING_EFFORT",
            "GROQ_ADMISSION_TIMEOUT_SECONDS", "GROQ_TPD_FALLBACK_ENABLED", "BANCO_AGIL_TEST_ID")
    for key in keys:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("LLM_API_KEY", "local-fake-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
    monkeypatch.setenv("LLM_MODEL", PRIMARY)
    monkeypatch.setenv("DATA_DIR", str(repo_ana_620.dir))
    organization = "bootstrap-local-" + uuid4().hex
    monkeypatch.setenv("GROQ_ORGANIZATION", organization)
    monkeypatch.setattr(bootstrap, "load_dotenv", lambda *a, **kw: None)
    monkeypatch.setattr(bootstrap, "setup_logging", lambda *a, **kw: None)
    builds, limiter_models = [], []
    original_limiter = bootstrap.shared_limiter

    def configured_limiter(organization, model, rpm, tpm):
        limiter_models.append(model)
        return original_limiter(organization, model, rpm, tpm)

    monkeypatch.setattr(bootstrap, "shared_limiter", configured_limiter)

    def no_http(*args, **kwargs):
        raise AssertionError("Bootstrap must not send a provider request")

    def fake_sdk(**kwargs):
        sdk = NS(chat=NS(completions=NS(create=no_http, with_raw_response=NS(create=no_http))))
        builds.append((kwargs, sdk))
        return sdk

    monkeypatch.setattr(bootstrap, "OpenAI", fake_sdk)
    return NS(repo=repo_ana_620, builds=builds, organization=organization, limiter_models=limiter_models)


def snapshots(repo):
    return {path.name: path.read_bytes() for path in repo.dir.glob("*.csv")}


@pytest.mark.parametrize("mode", ["native", "structured"])
@pytest.mark.parametrize("model", [PRIMARY, OTHER_MODEL, "another-model"])
def test_bootstrap_builds_only_the_configured_model(configured, monkeypatch, mode, model):
    monkeypatch.setenv("LLM_OUTPUT_MODE", mode)
    monkeypatch.setenv("LLM_MODEL", model)
    before = snapshots(configured.repo)
    orch = bootstrap.build_orchestrator()
    assert orch.model == model and orch.llm_mode == mode
    assert not hasattr(orch, "model_fallback")
    assert isinstance(orch.client, GovernedClient)
    assert len(configured.builds) == 1 and configured.limiter_models == [model]
    kwargs, sdk = configured.builds[0]
    assert kwargs["max_retries"] == 0 and orch.client.client is sdk
    assert orch.client.output_tokens == 2048 and orch.client.reasoning_effort == "low"
    assert snapshots(configured.repo) == before


@pytest.mark.parametrize("legacy_flag", ["true", "1", "false", "invalid"])
def test_obsolete_environment_flag_cannot_create_another_model(configured, monkeypatch, legacy_flag):
    monkeypatch.setenv("GROQ_TPD_FALLBACK_ENABLED", legacy_flag)
    orch = bootstrap.build_orchestrator()
    assert orch.model == PRIMARY and not hasattr(orch, "model_fallback")
    assert len(configured.builds) == 1 and configured.limiter_models == [PRIMARY]


@pytest.mark.parametrize("url", [
    "https://api.openai.com/v1",
    "https://api.groq.com.example.invalid/v1",
    "http://127.0.0.1:12345/v1",
])
def test_other_provider_preserves_the_configured_model_and_single_sdk(configured, monkeypatch, url):
    monkeypatch.setenv("LLM_BASE_URL", url)
    orch = bootstrap.build_orchestrator()
    assert orch.model == PRIMARY and not hasattr(orch, "model_fallback")
    assert len(configured.builds) == 1 and configured.limiter_models == []
    assert orch.client is configured.builds[0][1]


def test_per_model_settings_apply_only_to_the_configured_model(configured, monkeypatch):
    limits = {configured.organization: {
        PRIMARY: {"rpm": 15, "tpm": 7000, "output_tokens": 1024},
        OTHER_MODEL: {"rpm": 25, "tpm": 6000, "output_tokens": 512},
    }, "another-organization": {PRIMARY: {"rpm": 1, "tpm": 1}}}
    monkeypatch.setenv("GROQ_RATE_LIMITS_JSON", json.dumps(limits))
    monkeypatch.setenv("GROQ_REASONING_EFFORT", "medium")
    monkeypatch.setenv("GROQ_ADMISSION_TIMEOUT_SECONDS", "45")
    monkeypatch.setenv("LLM_MAX_RATE_LIMIT_WAIT_SECONDS", "35")
    orch = bootstrap.build_orchestrator()
    client = orch.client
    assert (client.limiter.rpm, client.limiter.tpm, client.output_tokens) == (15, 7000, 1024)
    assert client.max_wait == 45 and client.reasoning_effort == "medium"
    assert orch.max_rate_limit_wait_seconds == 35
    assert configured.limiter_models == [PRIMARY]


def test_rebuild_reuses_accounting_without_resetting_error_cooldown(configured):
    first = bootstrap.build_orchestrator()
    limiter = first.client.limiter
    limiter.defer(45)
    until = limiter.blocked_until
    rebuilt = bootstrap.build_orchestrator()
    assert rebuilt.client.limiter is limiter and limiter.blocked_until == until
    assert first.model == rebuilt.model == PRIMARY
    assert configured.limiter_models == [PRIMARY, PRIMARY]


@pytest.mark.parametrize("settings", [{"tpm": 0}, {"rpm": -1}, {"output_tokens": 1}])
def test_bad_active_settings_fail_locally_without_http(configured, monkeypatch, settings):
    monkeypatch.setenv("GROQ_RATE_LIMITS_JSON", json.dumps({configured.organization: {PRIMARY: settings}}))
    with pytest.raises(bootstrap.ConfigError):
        bootstrap.build_orchestrator()
    assert len(configured.builds) == 1


def test_unused_model_settings_are_ignored_even_with_legacy_flag(configured, monkeypatch):
    monkeypatch.setenv("GROQ_TPD_FALLBACK_ENABLED", "true")
    monkeypatch.setenv("GROQ_RATE_LIMITS_JSON", json.dumps({configured.organization: {OTHER_MODEL: {"tpm": 0}}}))
    orch = bootstrap.build_orchestrator()
    assert not hasattr(orch, "model_fallback")
    assert orch.model == PRIMARY and orch.client.limiter.tpm == 8000
    assert configured.limiter_models == [PRIMARY]
