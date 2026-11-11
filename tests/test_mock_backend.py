import json

from ai_news_pipeline.llm.mock import MockBackend


def _score(title, summary="", weight=1.0):
    b = MockBackend()
    return b.complete_json(purpose="score", system="", schema={"type": "object"},
                           user=json.dumps({"article": {"title": title, "summary": summary,
                                                         "source_weight": weight}}))


def _verify(title, summary="", weight=1.0):
    b = MockBackend()
    return b.complete_json(purpose="verify", system="", schema={"type": "object"},
                           user=json.dumps({"article": {"title": title, "summary": summary,
                                                         "source_weight": weight}}))


def test_scores_are_deterministic_and_in_range():
    a = _score("New open-weight LLM released", "Weights on GitHub, pip install it today.")
    assert a == _score("New open-weight LLM released", "Weights on GitHub, pip install it today.")
    for k in ("relevance", "novelty", "actionability"):
        assert 0 <= a[k] <= 10
    assert a["relevance"] >= 6 and a["actionability"] >= 6


def test_off_topic_is_low_relevance_and_other():
    s = _score("Spring gardening", "Tomatoes and basil.")
    assert s["relevance"] == 2 and s["category"] == "other"


def test_whole_word_matching():
    # "clients" must not count as the artefact "cli"
    v = _verify("Model maker partners with consultancy", "Bringing LLM agents to clients.")
    assert v["verdict"] == "thin"


def test_verdicts():
    assert _verify("Revolutionary model changes everything")["verdict"] == "hype"
    assert _verify("Revolutionary agent memory", "open-source repo with a benchmark")["verdict"] == "pass"
    assert _verify("Leaked: secret model")["verdict"] == "dubious"
    assert _verify("Vendor update", "Short note", weight=1.5)["verdict"] == "pass"


def test_usage_is_counted():
    b = MockBackend()
    b.complete_json(purpose="score", system="", schema={"type": "object"},
                    user=json.dumps({"article": {"title": "x"}}))
    assert b.usage["score"]["calls"] == 1
