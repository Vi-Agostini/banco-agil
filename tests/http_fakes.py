"""Servidor HTTP de loopback para conferir a serialização do SDK."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import pytest
from openai import OpenAI


@pytest.fixture
def provider():
    """SDK real contra HTTP loopback; nunca lê credenciais ou usa outro destino."""
    requests, responses = [], []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            status, body = responses.pop(0) if responses else (500, {"error": {"message": "Unexpected local call"}})
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = OpenAI(api_key="local-test-only", base_url=f"http://127.0.0.1:{server.server_port}/v1",
                    max_retries=0, timeout=2)
    try:
        yield client, requests, responses
    finally:
        client.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        assert not thread.is_alive()
