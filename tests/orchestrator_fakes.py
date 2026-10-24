"""Criação de orquestradores e sessões para os cenários de teste."""
from banco_agil.exchange import ExchangeService
from banco_agil.orchestrator import Orchestrator
from fakes import FakeLLM


def make(repo, script, **kw):
    return Orchestrator(repo, ExchangeService(), FakeLLM(script), "fake", **kw)


def logged_in(orch, cpf="11144477735", nome="Ana Souza", agent="triagem"):
    s = orch.new_session()
    s.authenticated, s.cpf, s.nome, s.active_agent = True, cpf, nome, agent
    return s
