import pytest

from banco_agil.flow import Awaiting, LIMIT_RETRY_PROMPT, apply_tool_effects
from banco_agil.session import Session
from banco_agil.statuses import STATUS_APROVADO, STATUS_REPROVADO


@pytest.mark.parametrize(("name", "result", "awaiting", "kind", "attempts", "prompt", "pending_cpf"), [
    ("solicitar_aumento_limite", {"status_pedido": STATUS_APROVADO}, Awaiting.NONE, None, 0, None, "11144477735"),
    ("solicitar_aumento_limite", {"status_pedido": STATUS_REPROVADO}, Awaiting.INTERVIEW_OFFER, None, 0, None, "11144477735"),
    ("solicitar_aumento_limite", {"erro": "inválido", "codigo": "valor_invalido"}, Awaiting.LIMIT_AMOUNT, "limit", 3, LIMIT_RETRY_PROMPT, "11144477735"),
    ("registrar_entrevista_e_calcular_score", {"novo_score": 570}, Awaiting.NONE, None, 0, None, "11144477735"),
    ("consultar_cotacao", {"cotacao": 5.0}, Awaiting.LIMIT_AMOUNT, "limit", 2, "Informe o limite", "11144477735"),
    ("transferir_para_agente", {"ok": True}, Awaiting.NONE, None, 0, None, None),
    ("transferir_para_agente", {"ok": False}, Awaiting.LIMIT_AMOUNT, "limit", 2, "Informe o limite", "11144477735"),
    ("encerrar_atendimento", {"ok": True}, Awaiting.NONE, None, 0, None, None),
    ("autenticar_cliente", {"sucesso": True}, Awaiting.NONE, None, 2, "Informe nascimento", None),
    ("autenticar_cliente", {"sucesso": False, "motivo": "dados_incompletos"}, Awaiting.BIRTH_DATE, None, 2, "Informe nascimento", "11144477735"),
])
def test_apply_tool_effects_transition_table(name, result, awaiting, kind, attempts, prompt, pending_cpf):
    session = Session()
    session.flow.awaiting = Awaiting.BIRTH_DATE if name == "autenticar_cliente" else Awaiting.LIMIT_AMOUNT
    session.flow.input_kind = "limit" if session.flow.awaiting is Awaiting.LIMIT_AMOUNT else None
    session.flow.input_prompt = "Informe o limite" if session.flow.awaiting is Awaiting.LIMIT_AMOUNT else "Informe nascimento"
    session.flow.attempts = 2
    session.flow.pending_cpf = "11144477735"

    apply_tool_effects(session, name, result)

    assert session.flow.awaiting is awaiting
    assert session.flow.input_kind == kind
    assert session.flow.attempts == attempts
    assert session.flow.input_prompt == prompt
    assert session.flow.pending_cpf == pending_cpf
