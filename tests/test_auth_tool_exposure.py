"""A autenticação aceita apenas credenciais coletadas localmente."""
import json
from types import SimpleNamespace

import pytest

from banco_agil.agents import TRIAGEM
from banco_agil.flow import Awaiting
from banco_agil.llm import call_llm
from orchestrator_fakes import make
from fakes import reply


@pytest.mark.parametrize('state', ['empty', 'cpf', 'birth', 'authenticated'])
def test_actual_llm_request_never_exposes_local_authentication(repo, state):
    orch = make(repo, [reply('Informe o próximo dado necessário.')])
    s = orch.new_session()
    if state == 'cpf':
        s.flow.pending_cpf = '11144477735'
        s.flow.awaiting = Awaiting.BIRTH_DATE
    elif state == 'birth':
        s.history.append({'role': 'user', 'content': '[DATA_NASCIMENTO]'})
    elif state == 'authenticated':
        s.authenticated = True
    requests = []
    def create(**request):
        requests.append(request)
        return reply('Informe o próximo dado necessário.')
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    call_llm(client, 'fake', TRIAGEM, s)
    names = [t['function']['name'] for t in requests[0]['tools']]
    assert 'autenticar_cliente' not in names
    assert 'encerrar_atendimento' in names
    schema = TRIAGEM.get_tool('autenticar_cliente').schema()['function']['parameters']
    assert schema['required'] == ['cpf', 'data_nascimento']


@pytest.mark.parametrize('args', [{}, {'cpf': '11144477735'},
    {'data_nascimento': '12/05/1990'}, {'cpf': None, 'data_nascimento': 123},
    {'cpf': '11144477735', 'data_nascimento': '12/05/1990'}])
def test_noncompliant_provider_cannot_authenticate_with_invented_data(repo, args, monkeypatch):
    orch = make(repo, [reply(calls=[('autenticar_cliente', args)])])
    s = orch.new_session()
    before = [p.read_bytes() for p in (repo.clients_path, repo.score_path, repo.requests_path)]
    monkeypatch.setattr(repo, 'authenticate', lambda *a: pytest.fail('authentication must not run'))
    response = orch.handle(s, 'quero aumentar meu limite para 9000')
    assert 'CPF' in response
    assert not s.authenticated and s.auth_failures == 0
    results = [json.loads(m['content']) for m in s.history if m['role'] == 'tool']
    assert results == []
    assert s.last_failure_category == 'invalid_tool_response'
    assert before == [p.read_bytes() for p in (repo.clients_path, repo.score_path, repo.requests_path)]
    assert s.active_agent == 'triagem'


def test_compatible_provider_reproduces_schema_rejection_before_and_safe_collection_after(repo):
    class Provider:
        def __init__(self):
            self.chat = SimpleNamespace(completions=self)
            self.rejected = 0
        def create(self, **request):
            if any(t['function']['name'] == 'autenticar_cliente' for t in request.get('tools', [])):
                self.rejected += 1
                error = RuntimeError('tool_use_failed: missing required properties')
                error.status_code = 400
                error.body = {'error': {'type': 'invalid_request_error', 'code': 'tool_use_failed'}}
                raise error
            return reply('Por favor, informe seu CPF.')
    provider = Provider()
    # O catálogo completo não filtra credenciais; o envio real deve usar o catálogo seguro.
    with pytest.raises(RuntimeError):
        provider.create(tools=[tool.schema() for tool in TRIAGEM.tools])
    orch = make(repo, [])
    orch.client = provider
    s = orch.new_session()
    assert 'CPF' in orch.handle(s, 'quero aumentar meu limite para 9000')
    assert provider.rejected == 1 and s.last_failure_category is None
    assert not s.authenticated and s.auth_failures == 0


@pytest.mark.parametrize('credentials,success,failures', [
    ('11144477735 12/05/1990', True, 0),
    ('11144477735 01/01/2000', False, 1),
    ('11144477735 31/02/1990', False, 1),
])
def test_local_complete_credentials_keep_existing_validation(repo, credentials, success, failures):
    orch = make(repo, [reply('Como posso ajudar?')] if success else [])
    s = orch.new_session()
    orch.handle(s, credentials)
    assert s.authenticated is success and s.auth_failures == failures


def test_missing_birth_is_collected_without_llm_authentication(repo):
    orch = make(repo, [])
    s = orch.new_session()
    assert 'nascimento' in orch.handle(s, '11144477735')
    assert not s.authenticated and s.auth_failures == 0
    orch.handle(s, '12/05/1990')
    assert s.authenticated and not orch.client.requests


def test_loopback_http_provider_rejects_old_exposure_and_accepts_corrected_flow(repo):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from threading import Thread
    import urllib.request
    from urllib.error import HTTPError

    records = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            names = [t['function']['name'] for t in body.get('tools', [])]
            rejected = 'autenticar_cliente' in names
            records.append({'tools': names, 'status': 400 if rejected else 200})
            payload = ({'error': {'type': 'invalid_request_error', 'code': 'tool_use_failed',
                       'message': 'missing properties cpf, data_nascimento'}} if rejected else
                       {'id': 'local', 'object': 'chat.completion', 'created': 0, 'model': 'fake',
                        'choices': [{'index': 0, 'finish_reason': 'stop',
                                     'message': {'role': 'assistant', 'content': 'Por favor, informe seu CPF.'}}]})
            data = json.dumps(payload).encode()
            self.send_response(400 if rejected else 200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = Thread(target=server.serve_forever)
    thread.start()
    try:
        class Client:
            def __init__(self):
                self.chat = SimpleNamespace(completions=self)
                self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            def create(self, **payload):
                request = urllib.request.Request(
                    f'http://127.0.0.1:{server.server_port}/v1/chat/completions',
                    data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
                with self.opener.open(request, timeout=5) as response:
                    body = json.load(response)
                return reply(body['choices'][0]['message']['content'])
        client = Client()
        with pytest.raises(HTTPError) as rejected:
            client.create(model='fake', messages=[{'role': 'user', 'content': 'quero aumentar meu limite para 9000'}], tools=[tool.schema() for tool in TRIAGEM.tools])
        assert json.load(rejected.value)['error']['code'] == 'tool_use_failed'
        orch = make(repo, [])
        orch.client = client
        s = orch.new_session()
        assert 'CPF' in orch.handle(s, 'quero aumentar meu limite para 9000')
        assert not s.authenticated and s.auth_failures == 0
        assert [r['status'] for r in records] == [400, 200]
        assert 'autenticar_cliente' not in records[-1]['tools']
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()


@pytest.mark.parametrize('args', [{}, {'cpf': '11144477735'}, {'data_nascimento': '12/05/1990'}])
def test_local_incomplete_credentials_do_not_consume_attempt(repo, args):
    orch = make(repo, [])
    s = orch.new_session()
    result = orch.tool_executor.execute(TRIAGEM, s, 'autenticar_cliente', json.dumps(args))
    assert result['motivo'] == 'dados_incompletos'
    assert not s.authenticated and s.auth_failures == 0


def test_birth_without_cpf_does_not_authenticate(repo):
    orch = make(repo, [reply('Por favor, informe seu CPF.')])
    s = orch.new_session()
    assert 'CPF' in orch.handle(s, 'nasci em 12/05/1990')
    assert not s.authenticated and s.auth_failures == 0
    assert not any(m.get('tool_calls') for m in s.history)
