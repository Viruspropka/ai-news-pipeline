"""The Anthropic backend with a fake client: no network, no key, no SDK needed."""
import json
from types import SimpleNamespace

import pytest

from ai_news_pipeline.llm import LLMError, QuotaExhausted, SchemaError
from ai_news_pipeline.llm.anthropic_backend import AnthropicBackend, api_schema
from ai_news_pipeline.stages.score import score_schema

SCHEMA = score_schema(["models", "other"])
GOOD = {"relevance": 7, "novelty": 5, "actionability": 3, "category": "models", "one_line": "A fine summary"}


class FakeMessages:
    def __init__(self, outcome):
        self.outcome = outcome
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def _resp(obj, stop="end_turn"):
    text = obj if isinstance(obj, str) else json.dumps(obj)
    return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)], stop_reason=stop,
                           usage=SimpleNamespace(input_tokens=120, output_tokens=30))


def _backend(outcome):
    msgs = FakeMessages(outcome)
    return AnthropicBackend(client=SimpleNamespace(messages=msgs)), msgs


def test_structured_output_request_shape_and_usage():
    b, msgs = _backend(_resp(GOOD))
    assert b.complete_json(purpose="score", system="S", user="U", schema=SCHEMA) == GOOD
    call = msgs.calls[0]
    assert call["model"] == "claude-haiku-5-5"
    fmt = call["output_config"]["format"]
    assert fmt["type"] == "json_schema"
    assert fmt["schema"]["additionalProperties"] is False
    assert "maximum" not in json.dumps(fmt["schema"])  # bounds are checked locally
    assert "tools" not in call and "tool_choice" not in call
    assert b.usage["score"] == {"calls": 1, "input_tokens": 120, "output_tokens": 30}


def test_local_validation_still_applies():
    b, _ = _backend(_resp({**GOOD, "relevance": 42}))
    with pytest.raises(SchemaError, match="maximum"):
        b.complete_json(purpose="score", system="S", user="U", schema=SCHEMA)


def test_refusal_and_truncation_are_item_errors():
    b, _ = _backend(_resp(GOOD, stop="refusal"))
    with pytest.raises(LLMError, match="declined"):
        b.complete_json(purpose="score", system="S", user="U", schema=SCHEMA)
    b, _ = _backend(_resp("{", stop="max_tokens"))
    with pytest.raises(LLMError, match="max_tokens"):
        b.complete_json(purpose="score", system="S", user="U", schema=SCHEMA)


class FakeAPIError(Exception):
    def __init__(self, status, type_, headers=None):
        super().__init__(f"{status} {type_}")
        self.status_code = status
        self.type = type_
        self.response = SimpleNamespace(headers=headers or {})


def test_billing_and_rate_limit_stop_the_run():
    b, _ = _backend(FakeAPIError(402, "billing_error"))
    with pytest.raises(QuotaExhausted):
        b.complete_json(purpose="verify", system="S", user="U", schema=SCHEMA)
    b, _ = _backend(FakeAPIError(429, "rate_limit_error", {"retry-after": "30"}))
    with pytest.raises(QuotaExhausted) as info:
        b.complete_json(purpose="verify", system="S", user="U", schema=SCHEMA)
    assert info.value.retry_after_s == 30


def test_other_api_errors_fail_only_the_item():
    b, _ = _backend(FakeAPIError(400, "invalid_request_error"))
    with pytest.raises(LLMError) as info:
        b.complete_json(purpose="brief", system="S", user="U", schema=SCHEMA)
    assert not isinstance(info.value, QuotaExhausted)


def test_api_schema_does_not_mutate_input():
    before = json.dumps(SCHEMA, sort_keys=True)
    api_schema(SCHEMA)
    assert json.dumps(SCHEMA, sort_keys=True) == before
