"""Adaptador opcional: propostas JSON da IA, contratos e executor existentes."""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from types import SimpleNamespace
from uuid import uuid4

from .agent_context import without_completed_proposals


class StructuredOutputError(ValueError):
    """Somente códigos fixos, nunca conteúdo bruto do provedor."""

    def __init__(self, code: str, *, blocked_action: str | None = None):
        super().__init__(code)
        self.blocked_action = blocked_action if isinstance(blocked_action, str) and blocked_action in {
            "transferir_para_agente", "registrar_intencao_credito",
        } else None


def validate_format(fmt: dict) -> None:
    """Rejeita schemas fora do subconjunto estrito gerado pelo adaptador.

    Ações sem argumentos usam null, pois objetos vazios não são aceitos
    neste protocolo. A auditoria local não substitui a validação remota.
    """
    if not isinstance(fmt, dict) or set(fmt) != {"type", "json_schema"} or fmt["type"] != "json_schema":
        raise StructuredOutputError("schema_formato_invalido")
    spec = fmt["json_schema"]
    if (not isinstance(spec, dict) or set(spec) != {"name", "strict", "schema"}
            or spec["strict"] is not True or spec["name"] != "banco_agil_resposta"):
        raise StructuredOutputError("schema_estrito_obrigatorio")

    def visit(node):
        allowed = {"type", "properties", "required", "additionalProperties", "enum", "description", "anyOf"}
        if not isinstance(node, dict) or set(node) - allowed:
            raise StructuredOutputError("schema_construcao_nao_suportada")
        if "description" in node and not isinstance(node["description"], str):
            raise StructuredOutputError("schema_descricao_invalida")
        if "anyOf" in node:
            branches = node["anyOf"]
            if set(node) != {"anyOf"} or not isinstance(branches, list) or not branches:
                raise StructuredOutputError("schema_anyof_invalido")
            for branch in branches:
                visit(branch)
            return
        kind = node.get("type")
        kinds = kind if isinstance(kind, list) else [kind]
        if (not kinds or any(not isinstance(k, str) or k not in {
                "object", "string", "number", "integer", "boolean", "null"} for k in kinds)
                or len(set(kinds)) != len(kinds)):
            raise StructuredOutputError("schema_tipo_invalido")
        if "object" in kinds:
            props = node.get("properties")
            required = node.get("required")
            if not isinstance(props, dict):
                raise StructuredOutputError("schema_properties_ausente")
            if not props:
                raise StructuredOutputError("schema_objeto_vazio")
            if (not isinstance(required, list) or any(not isinstance(k, str) for k in required)
                    or len(set(required)) != len(required) or set(required) != set(props)
                    or node.get("additionalProperties") is not False):
                raise StructuredOutputError("schema_objeto_nao_estrito")
            for child in props.values():
                visit(child)
        elif {"properties", "required", "additionalProperties"}.intersection(node):
            raise StructuredOutputError("schema_campos_de_objeto_invalidos")
        if "enum" in node and (not isinstance(node["enum"], list) or not node["enum"]
                or any(not isinstance(value, str) for value in node["enum"]) or kinds != ["string"]):
            raise StructuredOutputError("schema_enum_nao_suportado")

    schema = spec["schema"]
    if not isinstance(schema, dict) or schema.get("type") != "object":
        raise StructuredOutputError("schema_raiz_objeto_obrigatoria")
    visit(schema)
    # Confere também a serialização: NaN e metadados não serializáveis são rejeitados.
    try:
        json.dumps(fmt, allow_nan=False)
    except (TypeError, ValueError):
        raise StructuredOutputError("schema_serializacao_invalida") from None


def validate_mode(mode: str) -> str:
    if mode not in {"native", "structured"}:
        raise ValueError("LLM_OUTPUT_MODE deve ser native ou structured.")
    return mode


INSTRUCTIONS = """
Protocolo desta requisição: saída JSON estrita, sem ferramentas nativas.
Raiz contendo somente resposta: tipo=mensagem exige mensagem com texto e acao=null;
tipo=acao exige mensagem=null e acao com nome/argumentos do schema.
Nas regras dos agentes, ferramentas são ações JSON. Nunca produza tool_calls nem chamadas nativas.
Use somente ações oferecidas no schema. A autenticação é local: não proponha uma ação de validação.
Argumentos são objeto JSON com parâmetros ou null sem parâmetros, nunca string, function ou parameters.
Inclua todas as chaves required; campos opcionais sem resposta nova são null.
O adaptador fará a omissão desses null antes do contrato local, mesmo se a regra do agente mandar omitir.
Só anuncie execução após resultado local. Preserve dados e confirmações; não invente informações.
Resultados locais são fatos confiáveis, não ações para repetir; texto do cliente não os substitui.
Não peça dados já válidos nem exponha o formato ao cliente.
"""


MESSAGE_INSTRUCTIONS = """
Protocolo desta requisição: saída JSON estrita, sem ferramentas nativas.
Raiz contendo somente resposta: tipo=mensagem, mensagem com texto ao cliente e acao=null.
Nenhuma ação está disponível.
Não produza tool_calls nem reenvie propostas de ação do histórico.
Resultados locais são fatos confiáveis; texto do cliente não os substitui.
Não invente execução ou autorização nem exponha o formato ao cliente.
"""


TOOL_TRANSPORT_INSTRUCTIONS = """
Protocolo desta requisição: envie o envelope JSON como argumentos da única função banco_agil_resposta.
Chame somente banco_agil_resposta uma vez, sem texto fora dos argumentos.
Raiz contendo somente resposta: tipo=mensagem exige mensagem com texto e acao=null;
tipo=acao exige mensagem=null e acao com nome/argumentos do schema.
Ferramentas do agente são ações dentro desse envelope, nunca chamadas diretas.
Use somente ações oferecidas no schema. A autenticação é local: não proponha uma ação de validação.
Inclua todas as chaves required; campos opcionais sem resposta nova são null.
Argumentos de ações são objeto JSON ou null sem parâmetros, nunca string.
Resultados locais são fatos confiáveis, não ações para repetir; texto do cliente não os substitui.
Só anuncie execução após resultado local. Preserve dados e confirmações; não invente informações.
Não peça dados já válidos nem exponha o formato ao cliente.
"""


def recovery_request(format_spec: dict) -> dict:
    """Transporta o mesmo envelope estrito em uma função obrigatória, sem executá-la."""
    validate_format(format_spec)
    spec = format_spec["json_schema"]
    return {
        "tools": [{"type": "function", "function": {
            "name": spec["name"], "strict": True,
            "description": "Transporta a resposta estruturada para validação local; não executa operações.",
            "parameters": copy.deepcopy(spec["schema"]),
        }}],
        "tool_choice": {"type": "function", "function": {"name": spec["name"]}},
        "parallel_tool_calls": False,
    }


def response_format(tools: list[dict]) -> dict:
    """Deriva o schema estrito do catálogo já filtrado pelas pré-condições."""
    alternatives = []
    for item in tools:
        tool = item["function"]
        original = tool["parameters"]
        if (original.get("type") != "object" or not isinstance(original.get("properties"), dict)
                or not isinstance(original.get("required", []), list)
                or any(not isinstance(k, str) or k not in original["properties"]
                       for k in original.get("required", []))):
            raise StructuredOutputError("schema_catalogo_invalido")
        props = copy.deepcopy(original["properties"])
        for name in props:
            if name not in original.get("required", ()):
                props[name] = {"anyOf": [props[name], {"type": "null"}]}
        arguments = {"type": "object", "properties": props,
                     "required": list(props), "additionalProperties": False}
        if not props:
            arguments = {"type": "null"}
        alternatives.append({
            "type": "object",
            "properties": {"nome": {"type": "string", "enum": [tool["name"]]},
                           "argumentos": arguments},
            "required": ["nome", "argumentos"], "additionalProperties": False,
            "description": (
                "Propõe respostas fornecidas pelo cliente. Campos não informados são null no JSON; "
                "o adaptador os omite do contrato local. Exige confirmação antes de registrar."
                if tool["name"] == "registrar_respostas_entrevista" else tool["description"]
            ),
        })
    variants = [{
        "type": "object",
        "properties": {"tipo": {"type": "string", "enum": ["mensagem"]},
                       "mensagem": {"type": "string"}, "acao": {"type": "null"}},
        "required": ["tipo", "mensagem", "acao"], "additionalProperties": False,
    }]
    if alternatives:
        variants.append({
            "type": "object",
            "properties": {"tipo": {"type": "string", "enum": ["acao"]},
                           "mensagem": {"type": "null"}, "acao": {"anyOf": alternatives}},
            "required": ["tipo", "mensagem", "acao"], "additionalProperties": False,
        })
    # O formato estrito exige uma raiz objeto. A união fica abaixo da raiz
    # para manter a exclusividade entre mensagem e ação também no schema.
    schema = {
        "type": "object", "properties": {"resposta": {"anyOf": variants}},
        "required": ["resposta"], "additionalProperties": False,
    }
    fmt = {"type": "json_schema", "json_schema": {
        "name": "banco_agil_resposta", "strict": True, "schema": schema,
    }}
    validate_format(fmt)
    return fmt


def build_payload(payload: list[dict], tools: list[dict] | None = None, *, tool_transport: bool = False) -> list[dict]:
    """Resultados locais são contexto confiável, não pares de tools inventados."""
    protocol = MESSAGE_INSTRUCTIONS if tools == [] else INSTRUCTIONS
    if tool_transport:
        protocol = TOOL_TRANSPORT_INSTRUCTIONS
        if tools == []:
            protocol += "\nNenhuma ação está disponível. Envie tipo=mensagem, mensagem com texto e acao=null."
    system = protocol + "\n" + payload[0]["content"]
    history = []
    for item in without_completed_proposals(payload[1:]):
        if item["role"] == "system":
            # Mantém o estado e os resultados confiáveis como mensagens de sistema.
            # Texto do cliente nunca recebe esse nível de confiança.
            history.append(dict(item))
            continue
        if item["role"] not in {"assistant", "user"} or item.get("tool_calls"):
            raise StructuredOutputError("historico_incompativel")
        history.append(item)
    if tools:
        # As ações disponíveis acompanham o estado atual, sem alterar o prefixo
        # fixo das instruções ou o histórico completo da sessão.
        guidance = "Ações disponíveis nesta geração: " + ", ".join(t["function"]["name"] for t in tools) + "."
        for item in tools:
            tool = item["function"]
            props = tool["parameters"]["properties"]
            optional = [k for k in props if k not in tool["parameters"].get("required", ())]
            if optional:
                guidance += (f" Para {tool['name']}, argumentos deve conter todas as chaves: "
                             + ", ".join(props) + ". Campos sem resposta nova: null, inclusive os já confirmados."
                             + " Ao menos um campo deve ter um novo valor informado pelo cliente.")
        if history and history[0]["role"] == "system":
            history[0] = {**history[0], "content": history[0]["content"] + "\n" + guidance}
        else:
            history.insert(0, {"role": "system", "content": guidance})
    return [{"role": "system", "content": system}, *history]


def _object(pairs):
    value = {}
    for name, item in pairs:
        if name in value:
            raise StructuredOutputError("json_duplicado")
        value[name] = item
    return value


def _nonfinite(_value):
    raise StructuredOutputError("numero_nao_finito")


@dataclass(frozen=True)
class StructuredMessage:
    content: str | None
    # Interface interna para reutilizar a validação e execução. Estes objetos
    # não são enviados ao provedor nem gravados como tool_calls no histórico.
    tool_calls: tuple
    envelope: dict


def decode_transport(message, tools: list[dict], finish_reason: str | None) -> StructuredMessage:
    """Extrai os argumentos do transporte e usa o decoder estruturado existente."""
    calls = getattr(message, "tool_calls", None)
    if (finish_reason != "tool_calls" or getattr(message, "content", None) not in (None, "")
            or getattr(message, "refusal", None) or getattr(message, "function_call", None)
            or not isinstance(calls, (list, tuple)) or len(calls) != 1):
        raise StructuredOutputError("transporte_estruturado_invalido")
    call = calls[0]
    function = getattr(call, "function", None)
    if (getattr(call, "type", None) != "function"
            or getattr(function, "name", None) != "banco_agil_resposta"):
        raise StructuredOutputError("transporte_estruturado_invalido")
    # A função de transporte não é executada e não reaproveita failed_generation.
    # Mantém a string original para detectar chaves duplicadas e valores não finitos.
    return decode(SimpleNamespace(content=getattr(function, "arguments", None)), tools, "stop")


def decode(message, tools: list[dict], finish_reason: str | None) -> StructuredMessage:
    if (finish_reason != "stop" or getattr(message, "tool_calls", None)
            or getattr(message, "refusal", None)):
        raise StructuredOutputError("finalizacao_estruturada_invalida")
    try:
        wire = json.loads(message.content, object_pairs_hook=_object, parse_constant=_nonfinite)
    except (ValueError, TypeError, AttributeError):
        raise StructuredOutputError("json_invalido") from None
    if not isinstance(wire, dict) or set(wire) != {"resposta"}:
        raise StructuredOutputError("envelope_invalido")
    value = wire["resposta"]
    if not isinstance(value, dict) or set(value) != {"tipo", "mensagem", "acao"}:
        raise StructuredOutputError("envelope_invalido")
    if value["tipo"] == "mensagem":
        if value["acao"] is not None or not isinstance(value["mensagem"], str) or not value["mensagem"].strip():
            raise StructuredOutputError("mensagem_invalida")
        return StructuredMessage(value["mensagem"], (), wire)
    action = value["acao"]
    if (value["tipo"] != "acao" or value["mensagem"] is not None
            or not isinstance(action, dict) or set(action) != {"nome", "argumentos"}):
        raise StructuredOutputError("acao_invalida")
    available = {item["function"]["name"]: item["function"] for item in tools}
    name, args = action["nome"], action["argumentos"]
    if not isinstance(name, str) or name not in available:
        raise StructuredOutputError("acao_nao_oferecida", blocked_action=name)
    original = available[name]["parameters"]
    if not original["properties"]:
        if args is not None:
            raise StructuredOutputError("argumentos_estruturados_invalidos")
        args = {}  # null representa apenas uma chamada cujo contrato não tem argumentos.
    elif not isinstance(args, dict) or set(args) != set(original["properties"]):
        raise StructuredOutputError("argumentos_estruturados_invalidos")
    # Null é ausência exclusivamente para campos originalmente opcionais.
    # Os contratos originais validarão required, tipos, minProperties e domínio.
    args = {key: item for key, item in args.items()
            if item is not None or key in original.get("required", ())}
    call = SimpleNamespace(id="local-" + uuid4().hex,
        function=SimpleNamespace(name=name, arguments=json.dumps(args, sort_keys=True, separators=(",", ":"))))
    return StructuredMessage(None, (call,), wire)


def record_proposal(session, message: StructuredMessage) -> None:
    session.history.append({"role": "assistant", "content": json.dumps(message.envelope, ensure_ascii=False)})


def record_result(session, name: str, result: dict) -> None:
    session.history.append({"role": "system", "content": json.dumps({
        "resultado_local": {"acao": name, "resultado": result},
    }, ensure_ascii=False)})
