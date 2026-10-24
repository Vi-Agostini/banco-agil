"""Diagnóstico sem conteúdo contra o schema enviado na requisição.

A análise cobre o subconjunto do adaptador e não substitui as validações de
execução. Não retorna nomes ou valores gerados pelo provedor.
"""
import hashlib
import json
import math
from collections.abc import Mapping

from .structured_output import _object, _nonfinite, StructuredOutputError


def _issues(value, schema, path='$', depth=0):
    if depth > 32:
        return [path + ':depth_limit']
    if 'anyOf' in schema:
        branches = schema['anyOf']
        # Usa o discriminador declarado para localizar o erro na alternativa correta,
        # sem atribuir toda falha de ação à alternativa de mensagem.
        if isinstance(value, dict):
            for discriminator in ('tipo', 'nome'):
                matching = [b for b in branches if value.get(discriminator) in
                            b.get('properties', {}).get(discriminator, {}).get('enum', [])]
                if matching:
                    branches = matching
                    break
        results = [_issues(value, branch, path, depth + 1) for branch in branches]
        if any(not errors for errors in results):
            return []
        return min(results, key=len) if results else [path + ':anyOf']
    types = schema.get('type', [])
    types = types if isinstance(types, list) else [types]
    matches = {
        'object': isinstance(value, dict), 'string': isinstance(value, str),
        'null': value is None, 'boolean': type(value) is bool,
        'number': type(value) in (int, float) and (not isinstance(value, float) or math.isfinite(value)),
        'integer': type(value) in (int, float) and (type(value) is int or math.isfinite(value) and value.is_integer()),
    }
    if not any(matches.get(kind, False) for kind in types):
        return [path + ':type']
    if 'enum' in schema and value not in schema['enum']:
        return [path + ':enum']
    errors = []
    if isinstance(value, dict):
        properties = schema['properties']
        for name in schema['required']:
            if name not in value:
                errors.append(path + '.' + name + ':required')
        if schema.get('additionalProperties') is False and set(value) - set(properties):
            errors.append(path + ':additional_properties')
        for name, node in properties.items():
            if name in value:
                errors.extend(_issues(value[name], node, path + '.' + name, depth + 1))
    return errors[:8]


def generation_report(error, format_spec):
    """Retorna apenas o estado da geração, caminhos do schema e seu hash."""
    body = getattr(error, 'body', None)
    detail = body.get('error', body) if isinstance(body, Mapping) else None
    generation = detail.get('failed_generation') if isinstance(detail, Mapping) else None
    if generation is None and isinstance(body, Mapping):
        generation = body.get('failed_generation')
    report = {'generation_status': 'not_provided', 'schema_issues': []}
    if format_spec is not None:
        schema = format_spec['json_schema']['schema']
        report['schema_sha256'] = hashlib.sha256(json.dumps(schema, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if generation is None:
        return report
    if not isinstance(generation, str):
        report['generation_status'] = 'unsupported_format'
        return report
    if len(generation) > 100_000:
        report['generation_status'] = 'size_limit'
        return report
    try:
        value = json.loads(generation, object_pairs_hook=_object, parse_constant=_nonfinite)
    except StructuredOutputError as error:
        report['generation_status'] = 'duplicate_key' if str(error) == 'json_duplicado' else 'nonfinite_number'
        return report
    except (ValueError, RecursionError):
        report['generation_status'] = 'invalid_json'
        return report
    if format_spec is None:
        report['generation_status'] = 'json_without_schema'
        return report
    errors = _issues(value, schema)
    report['schema_issues'] = errors
    report['generation_status'] = 'schema_mismatch' if errors else 'schema_valid'
    return report
