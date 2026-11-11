import pytest

from ai_news_pipeline.llm import SchemaError, make_backend, validate
from ai_news_pipeline.llm.base import extract_json_object
from ai_news_pipeline.stages.score import score_schema
from ai_news_pipeline.stages.verify import VERIFY_SCHEMA

SCHEMA = score_schema(["models", "other"])
GOOD = {"relevance": 7, "novelty": 5, "actionability": 3, "category": "models", "one_line": "A fine summary"}


def test_validate_accepts_good_object():
    validate(GOOD, SCHEMA)


@pytest.mark.parametrize("patch,needle", [
    ({"relevance": 11}, "maximum"),
    ({"relevance": -1}, "minimum"),
    ({"relevance": "7"}, "expected integer"),
    ({"relevance": True}, "boolean"),
    ({"category": "sports"}, "not in"),
    ({"one_line": "abc"}, "shorter"),
    ({"extra": 1}, "unexpected keys"),
])
def test_validate_rejects(patch, needle):
    with pytest.raises(SchemaError, match=needle):
        validate({**GOOD, **patch}, SCHEMA)


def test_validate_missing_key_and_nested_array():
    bad = dict(GOOD)
    del bad["novelty"]
    with pytest.raises(SchemaError, match="missing required key 'novelty'"):
        validate(bad, SCHEMA)
    with pytest.raises(SchemaError, match=r"\$\.flags\[1\]"):
        validate({"verdict": "pass", "flags": ["ok", ""], "confidence": 0.5}, VERIFY_SCHEMA)


def test_extract_json_object_tolerates_prose_and_fences():
    raw = 'Sure! ```json\n{"a": {"b": [1, 2]}, "c": "}"}\n``` done'
    assert extract_json_object(raw) == {"a": {"b": [1, 2]}, "c": "}"}
    with pytest.raises(SchemaError):
        extract_json_object("no json here")
    with pytest.raises(SchemaError):
        extract_json_object("")


def test_make_backend():
    assert make_backend("mock").name == "mock"
    assert make_backend("command", {"command": "cat"}).name == "command"
    with pytest.raises(ValueError):
        make_backend("gpt-whatever")
