"""Score stage: relevance, novelty, actionability, category, one-line summary.

One model call per unscored article, typed JSON only. Every failure is
logged with its cause and skips that one article; only QuotaExhausted
propagates, because then every further call would fail too.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from ..llm import LLMBackend, LLMError, QuotaExhausted
from ..store import Store
from .common import UNTRUSTED_NOTE, StageReport

log = logging.getLogger(__name__)

DEFAULT_WEIGHTS = {"relevance": 0.5, "actionability": 0.3, "novelty": 0.2}

SCORE_SYSTEM = f"""\
You are a senior AI-news triage editor. The reader is a working ML and
automation engineer who wants only actionable, novel, signal-rich items.
Reject hype, re-syndication and generic listicles.

You get ONE article as JSON. Score each axis from 0 to 10:
- relevance: AI agents, LLM advances, automation, developer tooling around
  them. Unrelated ML topics 2-4, direct hits 8-10, off-topic 0-2.
- novelty: new information (8-10) or a re-package of older news (0-3).
- actionability: can the reader try, install, clone or use it today?
  A pip-installable library, a new API or code 8-10; an opinion essay 0-3.
Pick exactly one category code from the list you are given.
one_line: a 12-25 word English summary of the unique value, no filler
such as "this article discusses".

{UNTRUSTED_NOTE}"""


def score_schema(categories: list[str]) -> dict[str, Any]:
    axis = {"type": "integer", "minimum": 0, "maximum": 10}
    return {
        "type": "object",
        "properties": {
            "relevance": axis,
            "novelty": axis,
            "actionability": axis,
            "category": {"type": "string", "enum": categories},
            "one_line": {"type": "string", "minLength": 5, "maxLength": 200},
        },
        "required": ["relevance", "novelty", "actionability", "category", "one_line"],
        "additionalProperties": False,
    }


def compute_composite(relevance: int, novelty: int, actionability: int,
                      source_weight: float, weights: dict[str, float] | None = None) -> float:
    """Weighted blend of the three axes times the source weight. The single formula.

    Relevance dominates because the brief promises "no off-topic items";
    actionability is the brief's edge over generic AI news.
    """
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    base = w["relevance"] * relevance + w["actionability"] * actionability + w["novelty"] * novelty
    return round(base * source_weight, 4)


def build_input(row: Any, categories: dict[str, str]) -> str:
    return json.dumps({
        "categories": categories,
        "article": {
            "source": row["source_name"],
            "source_weight": row["source_weight"],
            "title": row["title"],
            "summary": (row["summary"] or "")[:1500],
        },
    }, ensure_ascii=False)


def score_pending(store: Store, backend: LLMBackend, *, categories: dict[str, str],
                  weights: dict[str, float] | None = None, limit: int = 200) -> StageReport:
    report = StageReport()
    schema = score_schema(list(categories))
    for article_id in store.unscored_ids(limit):
        row = store.get_article(article_id)
        if row is None:
            report.skip(f"score {article_id}: article vanished")
            continue
        try:
            out = backend.complete_json(purpose="score", system=SCORE_SYSTEM,
                                        user=build_input(row, categories), schema=schema)
        except QuotaExhausted:
            raise
        except LLMError as exc:
            log.warning("score skip article_id=%s title=%r: %s", article_id, row["title"][:80], exc)
            report.skip(f"score {article_id}: {exc}")
            continue
        composite = compute_composite(out["relevance"], out["novelty"], out["actionability"],
                                      float(row["source_weight"]), weights)
        store.save_score(article_id, relevance=out["relevance"], novelty=out["novelty"],
                         actionability=out["actionability"], composite=composite,
                         category=out["category"], one_line=out["one_line"],
                         model=backend.model_for("score"))
        report.done += 1
    log.info("score: done=%d skipped=%d", report.done, report.skipped)
    return report
