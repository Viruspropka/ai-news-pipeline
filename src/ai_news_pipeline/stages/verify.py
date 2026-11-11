"""Verify stage: a stricter second opinion on the top-scored items.

Verdicts: pass | hype | dubious | thin. Only `pass` (or not yet verified)
items may reach a brief. Running it only on the top K keeps the expensive
model off the long tail.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from ..llm import LLMBackend, LLMError, QuotaExhausted
from ..store import Store
from .common import UNTRUSTED_NOTE, StageReport

log = logging.getLogger(__name__)

VERDICTS = ("dubious", "hype", "pass", "thin")
BLOCKING_VERDICTS = frozenset({"hype", "dubious", "thin"})

VERIFY_SYSTEM = f"""\
You are a strict claims editor for an AI-news brief that promises zero hype
and zero filler. Decide whether ONE scored article may appear in the brief.

Verdicts:
- pass: substantive, with a concrete artefact (model, library, demo, repo,
  paper, tutorial with code) and checkable claims. Default when unsure AND
  the source is a primary vendor or author blog (weight >= 1.3).
- hype: the framing oversells the artefact ("revolutionary", "AGI",
  "changes everything") and there is no demo, benchmark or repo behind it.
- dubious: claims look wrong or unverifiable, rumours, leaks, or a
  clickbait headline that the body does not support.
- thin: real but trivial - re-syndication, hiring news, a press release
  without technical content.
flags: 1-6 short kebab-case tags explaining the verdict, for example
no-demo, vendor-puff, re-syndication, strong-benchmark, open-source-repo.
confidence: your certainty in the verdict, 0.0-1.0.

{UNTRUSTED_NOTE}"""

VERIFY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": list(VERDICTS)},
        "flags": {"type": "array", "items": {"type": "string", "minLength": 1, "maxLength": 40},
                  "maxItems": 8},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
    "required": ["verdict", "flags", "confidence"],
    "additionalProperties": False,
}


def build_input(row: Any) -> str:
    return json.dumps({
        "article": {
            "source": row["source_name"],
            "source_weight": row["source_weight"],
            "scores": {"relevance": row["relevance"], "novelty": row["novelty"],
                       "actionability": row["actionability"]},
            "category": row["category"],
            "title": row["title"],
            "summary": (row["summary"] or "")[:2000],
        },
    }, ensure_ascii=False)


def verify_top(store: Store, backend: LLMBackend, *, min_relevance: int, top_k: int = 30,
               exclude_ids: set[int] | None = None) -> StageReport:
    """Verify the best unverified candidates; skip ids already shipped recently."""
    report = StageReport()
    rows = store.verify_candidates(min_relevance=min_relevance, limit=top_k,
                                   exclude_ids=exclude_ids or set())
    for row in rows:
        article_id = int(row["id"])
        try:
            out = backend.complete_json(purpose="verify", system=VERIFY_SYSTEM,
                                        user=build_input(row), schema=VERIFY_SCHEMA)
        except QuotaExhausted:
            raise
        except LLMError as exc:
            log.warning("verify skip article_id=%s: %s", article_id, exc)
            report.skip(f"verify {article_id}: {exc}")
            continue
        flags = [f.strip() for f in out["flags"] if f.strip()][:6]
        store.save_verification(article_id, verdict=out["verdict"], flags=flags,
                                confidence=float(out["confidence"]),
                                model=backend.model_for("verify"))
        report.done += 1
    log.info("verify: done=%d skipped=%d", report.done, report.skipped)
    return report
