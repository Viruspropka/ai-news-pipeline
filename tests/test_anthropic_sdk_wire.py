"""Real `anthropic` SDK against a local fake Messages endpoint: checks the
request the SDK actually puts on the wire. No network, no key, no cost.
Skipped when the optional SDK is not installed."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

anthropic = pytest.importorskip("anthropic")

from ai_news_pipeline.llm import QuotaExhausted  # noqa: E402
from ai_news_pipeline.llm.anthropic_backend import AnthropicBackend  # noqa: E402
from ai_news_pipeline.stages.verify import VERIFY_SCHEMA  # noqa: E402


class _Handler(BaseHTTPRequestHandler):
    status = 200
    seen: list = []

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).seen.append((self.path, body))
        if self.status == 200:
            payload = {
                "id": "msg_test", "type": "message", "role": "assistant", "model": body["model"],
                "content": [{"type": "text", "text": json.dumps(
                    {"verdict": "pass", "flags": ["open-source-repo"], "confidence": 0.9})}],
                "stop_reason": "end_turn", "stop_sequence": None,
                "usage": {"input_tokens": 50, "output_tokens": 12},
            }
        else:
            payload = {"type": "error", "error": {"type": "billing_error", "message": "credit balance is too low"}}
        data = json.dumps(payload).encode()
        self.send_response(self.status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


@pytest.fixture
def server():
    _Handler.seen = []
    _Handler.status = 200
    httpd = HTTPServer(("127.0.0.1", 0), _Handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield httpd
    httpd.shutdown()


def _client(httpd):
    return anthropic.Anthropic(api_key="test-key-not-real", max_retries=0,
                               base_url=f"http://127.0.0.1:{httpd.server_port}")


def test_sdk_sends_structured_output_request(server):
    b = AnthropicBackend(client=_client(server))
    out = b.complete_json(purpose="verify", system="SYS", user="USER", schema=VERIFY_SCHEMA)
    assert out["verdict"] == "pass"
    path, body = _Handler.seen[0]
    assert path == "/v1/messages"
    assert body["model"] == "claude-sonnet-5-5"
    assert body["output_config"]["format"]["type"] == "json_schema"
    assert body["output_config"]["effort"] == "medium"
    assert body["system"] == "SYS" and body["messages"] == [{"role": "user", "content": "USER"}]
    assert b.usage["verify"]["input_tokens"] == 50


def test_sdk_billing_error_stops_the_run(server):
    _Handler.status = 402
    b = AnthropicBackend(client=_client(server))
    with pytest.raises(QuotaExhausted):
        b.complete_json(purpose="verify", system="S", user="U", schema=VERIFY_SCHEMA)
