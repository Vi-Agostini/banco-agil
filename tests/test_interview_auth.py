import pytest

from banco_agil import tools
from banco_agil.flow import Awaiting

ANA = "11144477735"


def interview(ctx, **over):
    args = dict(renda_mensal=5000, tipo_emprego="formal", despesas_fixas=2000, num_dependentes=0, tem_dividas="não")
    args.update(over)
    # Chamadas unitárias diretas representam respostas já coletadas localmente.
    ctx.session.flow.interview_answers = dict(args)
    return tools.registrar_entrevista_e_calcular_score(ctx, **args)


# Entrevista.
def test_interview_persists_score_and_returns_to_credit(auth_ctx_ana_620):
    auth_ctx_ana_620.session.active_agent, auth_ctx_ana_620.session.desired_limit = "entrevista", 9000
    r = interview(auth_ctx_ana_620)
    assert r["novo_score"] == 575 and r["score_anterior"] == 620 and r["valor_desejado_anteriormente"] == 9000
    assert auth_ctx_ana_620.repo.get_client(ANA)["score"] == 575
    assert auth_ctx_ana_620.session.active_agent == "credito"


def test_interview_debts_sim_and_nao(auth_ctx_ana_620):
    assert interview(auth_ctx_ana_620, tem_dividas="sim")["novo_score"] == 375
    assert interview(auth_ctx_ana_620, tem_dividas="não")["novo_score"] == 575
    assert interview(auth_ctx_ana_620, tem_dividas=True)["novo_score"] == 375
    assert interview(auth_ctx_ana_620, tem_dividas=False)["novo_score"] == 575


@pytest.mark.parametrize("over", [
    dict(tem_dividas="talvez"), dict(renda_mensal=-1), dict(despesas_fixas=-5), dict(num_dependentes=-1),
    dict(num_dependentes="muitos"), dict(tipo_emprego="rei"), dict(renda_mensal=float("nan")),
    dict(despesas_fixas=float("inf")),
])
def test_interview_invalid_input_changes_nothing(auth_ctx_ana_620, over):
    auth_ctx_ana_620.session.active_agent = "entrevista"
    assert "erro" in interview(auth_ctx_ana_620, **over)
    assert auth_ctx_ana_620.repo.get_client(ANA)["score"] == 620 and auth_ctx_ana_620.session.active_agent == "entrevista"


def test_interview_zero_expenses_and_score_bounds(auth_ctx_ana_620):
    assert interview(auth_ctx_ana_620, despesas_fixas=0, renda_mensal=10000)["novo_score"] == 1000
    assert interview(auth_ctx_ana_620, renda_mensal=0, tipo_emprego="desempregado", tem_dividas="sim", num_dependentes=3)["novo_score"] == 0


# Autenticação.
def test_auth_success_with_punctuated_cpf_and_br_date(ctx):
    r = tools.autenticar_cliente(ctx, "111.444.777-35", "12/05/1990")
    assert r["sucesso"] and ctx.session.authenticated and ctx.session.cpf == ANA


def test_auth_rejects_parenthesized_birth_date_argument(ctx):
    r = tools.autenticar_cliente(ctx, ANA, "(12/05/1990)")
    assert not r["sucesso"] and not ctx.session.authenticated


@pytest.mark.parametrize("value,code", [
    (0, "valor_nao_positivo"),
    ("R$ 0,00", "valor_nao_positivo"),
    ("zero", "valor_nao_positivo"),
    ("-10", "valor_negativo"),
    ("-1.000,00", "valor_negativo"),
])
def test_nonpositive_limit_returns_typed_error_without_tool_retry_state(auth_ctx, value, code):
    first = tools.solicitar_aumento_limite(auth_ctx, value)
    second = tools.solicitar_aumento_limite(auth_ctx, value)
    assert first["codigo"] == second["codigo"] == code
    assert "repetir" not in first and "repetir" not in second


def test_limit_tool_returns_code_without_mutating_flow_state(auth_ctx):
    flow = auth_ctx.session.flow
    flow.awaiting = Awaiting.LIMIT_AMOUNT
    flow.input_kind = "limit"
    flow.attempts = 2

    result = tools.solicitar_aumento_limite(auth_ctx, 0)

    assert result["codigo"] == "valor_nao_positivo"
    assert (flow.awaiting, flow.input_kind, flow.attempts) == (Awaiting.LIMIT_AMOUNT, "limit", 2)
    assert not hasattr(flow, "last_invalid_limit")
    assert not hasattr(flow, "invalid_limit_retries")


@pytest.mark.parametrize("cpf,data", [
    ("111 444 777 35", "12-05-1990"),
    ("11144477735", "1990-05-12"),
    ("111.444.777-35", "12.05.1990"),
])
def test_auth_accepts_common_user_variants(ctx, cpf, data):
    r = tools.autenticar_cliente(ctx, cpf, data)
    assert r["sucesso"] and ctx.session.authenticated and ctx.session.cpf == ANA


@pytest.mark.parametrize("cpf,data", [("123", "12/05/1990"), ("11144477735", "31/02/1990"), (123, 456)])
def test_auth_invalid_format_is_generic_and_counts_as_attempt(ctx, cpf, data):
    result = tools.autenticar_cliente(ctx, cpf, data)

    assert not result["sucesso"]
    assert result["motivo"] == "dados_nao_conferem"
    assert "qual dado falhou" in result["instrucao"]
    assert ctx.session.auth_failures == 1


@pytest.mark.parametrize("cpf,data", [("", ""), (None, None), ("11144477735", " "), (" ", "12/05/1990")])
def test_auth_missing_field_does_not_count_as_attempt(ctx, cpf, data):
    result = tools.autenticar_cliente(ctx, cpf, data)

    assert result == {
        "sucesso": False,
        "motivo": "dados_incompletos",
        "instrucao": "Peça ao cliente somente o dado de autenticação que estiver faltando.",
    }
    assert ctx.session.auth_failures == 0 and not ctx.session.ended


def test_auth_tool_accepts_omitted_fields_without_counting_attempt(ctx):
    result = tools.autenticar_cliente(ctx)

    assert result["motivo"] == "dados_incompletos"
    assert ctx.session.auth_failures == 0


def test_auth_unknown_cpf_counts_as_failure(ctx):
    r = tools.autenticar_cliente(ctx, "99999999999", "12/05/1990")
    assert r["motivo"] == "dados_nao_conferem" and ctx.session.auth_failures == 1


def test_failure_response_is_identical_for_wrong_cpf_or_wrong_date(ctx):
    wrong_date = tools.autenticar_cliente(ctx, ANA, "01/01/2000")
    ctx.session.auth_failures = 0
    unknown_cpf = tools.autenticar_cliente(ctx, "99999999999", "12/05/1990")
    assert wrong_date == unknown_cpf


def test_three_failures_then_blocked_even_with_correct_data(ctx):
    for _ in range(3):
        tools.autenticar_cliente(ctx, ANA, "01/01/2000")
    assert ctx.session.ended
    r = tools.autenticar_cliente(ctx, ANA, "12/05/1990")
    assert r["codigo"] == "sessao_encerrada" and not ctx.session.authenticated


def test_success_resets_failure_counter(ctx):
    tools.autenticar_cliente(ctx, ANA, "01/01/2000")
    tools.autenticar_cliente(ctx, ANA, "12/05/1990")
    assert ctx.session.auth_failures == 0


# Autorização nas chamadas diretas.
SENSITIVE = [
    lambda c: tools.consultar_limite(c),
    lambda c: tools.solicitar_aumento_limite(c, 6000),
    lambda c: tools.registrar_entrevista_e_calcular_score(c, 5000, "formal", 2000, 0, "não"),
    lambda c: tools.consultar_cotacao(c, "USD"),
    lambda c: tools.transferir_para_agente(c, "credito"),
]


@pytest.mark.parametrize("call", SENSITIVE)
def test_unauthenticated_direct_calls_denied(ctx_ana_620, call):
    assert call(ctx_ana_620)["codigo"] == "nao_autenticado"
    assert ctx_ana_620.repo.list_requests() == [] and ctx_ana_620.repo.get_client(ANA)["score"] == 620


@pytest.mark.parametrize("call", SENSITIVE)
def test_flag_without_identity_is_denied(ctx, call):
    ctx.session.authenticated = True  # sem CPF na sessão não há identidade
    assert call(ctx)["codigo"] == "nao_autenticado"


@pytest.mark.parametrize("call", SENSITIVE)
def test_ended_session_denies_everything(auth_ctx, call):
    auth_ctx.session.ended = True
    assert call(auth_ctx)["codigo"] == "sessao_encerrada"
    assert auth_ctx.repo.list_requests() == []


def test_tools_have_no_cpf_parameter():
    import inspect
    for fn in (tools.consultar_limite, tools.solicitar_aumento_limite, tools.registrar_entrevista_e_calcular_score,
               tools.consultar_cotacao):
        assert "cpf" not in inspect.signature(fn).parameters


# Isolamento entre clientes e autorização.
BRUNO = "52998224725"


def test_session_isolation_different_customers(auth_ctx_ana_620, repo_ana_620):
    """Duas sessões com CPFs diferentes são totalmente isoladas."""
    # Sessão de Ana (já autenticada via fixture)
    ana_limit = tools.consultar_limite(auth_ctx_ana_620)
    assert ana_limit["limite_credito_atual"] == 5000.0

    # Cria sessão separada para Bruno no mesmo repositório
    from banco_agil.exchange import ExchangeService
    from banco_agil.session import Session
    from banco_agil.tools import ToolContext

    bruno_session = Session()
    bruno_session.authenticated = True
    bruno_session.cpf = BRUNO
    bruno_session.nome = "Bruno Lima"
    bruno_ctx = ToolContext(bruno_session, repo_ana_620, ExchangeService())

    bruno_limit = tools.consultar_limite(bruno_ctx)
    assert bruno_limit["limite_credito_atual"] == 2000.0
    assert bruno_limit["limite_credito_atual"] != ana_limit["limite_credito_atual"]

    # Operação de Ana (aprovada) não afeta dados de Bruno
    r_ana = tools.solicitar_aumento_limite(auth_ctx_ana_620, 6000)
    assert r_ana["status_pedido"] == "aprovado"  # Ana score 620, teto 8000
    assert repo_ana_620.get_client(BRUNO)["limite_credito"] == 2000.0

    # Operação de Bruno (reprovada) não afeta dados de Ana
    bruno_ctx.session.active_agent = "credito"
    r_bruno = tools.solicitar_aumento_limite(bruno_ctx, 5000)
    assert r_bruno["status_pedido"] == "reprovado"  # Bruno score 350, teto 3000
    assert repo_ana_620.get_client(ANA)["limite_credito"] == 6000.0


def test_authenticated_session_cannot_reauthenticate_as_third_party(auth_ctx_ana_620):
    """Sessão já autenticada como Ana não pode ser reautenticada como Bruno."""
    assert auth_ctx_ana_620.session.authenticated
    assert auth_ctx_ana_620.session.cpf == ANA

    # Tentativa de autenticar como Bruno com dados válidos de Bruno
    r = tools.autenticar_cliente(auth_ctx_ana_620, BRUNO, "30/11/1985")
    # Já autenticado: retorna sucesso sem alterar a identidade da sessão
    assert r["sucesso"] is True
    assert auth_ctx_ana_620.session.cpf == ANA  # identidade preservada
    assert auth_ctx_ana_620.session.nome == "Ana Souza"


def test_third_party_cpf_not_used_in_authenticated_tools(auth_ctx_ana_620):
    """Ferramentas autenticadas usam só o CPF da sessão, nunca o de terceiros."""
    assert auth_ctx_ana_620.session.authenticated
    assert auth_ctx_ana_620.session.cpf == ANA

    # solicitar_aumento_limite não aceita CPF como argumento
    r = tools.solicitar_aumento_limite(auth_ctx_ana_620, 6000)
    assert r["status_pedido"] == "aprovado"  # Ana score 620, teto 8000
    assert auth_ctx_ana_620.repo.get_client(ANA)["limite_credito"] == 6000.0
    # Bruno não foi afetado
    assert auth_ctx_ana_620.repo.get_client(BRUNO)["limite_credito"] == 2000.0

    # consultar_limite retorna o limite de Ana, não de Bruno
    limit = tools.consultar_limite(auth_ctx_ana_620)
    assert limit["limite_credito_atual"] == 6000.0


def test_other_customer_data_not_exposed_in_response(auth_ctx_ana_620):
    """Resposta de ferramentas não contém dados de clientes de terceiros."""
    r = tools.solicitar_aumento_limite(auth_ctx_ana_620, 6000)
    response_str = str(r)
    assert BRUNO not in response_str
    assert "Bruno" not in response_str
    assert "52998224725" not in response_str
