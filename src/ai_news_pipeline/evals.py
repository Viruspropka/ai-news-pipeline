"""Small, honest eval for the score and verify stages.

Runs the labelled cases in data/eval_cases.yaml through the SAME prompts,
schemas and input builders the pipeline uses, against any backend, and
reports:
- verdict accuracy, plus precision/recall per verdict
- false-pass rate: share of non-`pass` cases that the gate would let through
  (the number that matters most for a "no hype" brief)
- relevance-band accuracy (high >= threshold vs low)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

from .llm import LLMBackend, LLMError, QuotaExhausted
from .stages.score import SCORE_SYSTEM, score_schema
from .stages.score import build_input as score_input
from .stages.verify import VERDICTS, VERIFY_SCHEMA, VERIFY_SYSTEM
from .stages.verify import build_input as verify_input


@dataclass
class CaseResult:
    id: str
    expected_verdict: str
    got_verdict: str
    expected_relevance: str
    got_relevance: int | None
    error: str = ""


@dataclass
class EvalReport:
    backend: str
    results: list[CaseResult] = field(default_factory=list)
    relevance_threshold: int = 6

    @property
    def scored(self) -> list[CaseResult]:
        return [r for r in self.results if not r.error]

    def verdict_accuracy(self) -> float:
        rs = self.scored
        return sum(r.expected_verdict == r.got_verdict for r in rs) / len(rs) if rs else 0.0

    def false_pass_rate(self) -> float:
        neg = [r for r in self.scored if r.expected_verdict != "pass"]
        return sum(r.got_verdict == "pass" for r in neg) / len(neg) if neg else 0.0

    def relevance_accuracy(self) -> float:
        rs = [r for r in self.scored if r.got_relevance is not None]
        ok = sum((r.got_relevance >= self.relevance_threshold) == (r.expected_relevance == "high") for r in rs)
        return ok / len(rs) if rs else 0.0

    def per_verdict(self) -> dict[str, dict[str, float]]:
        out = {}
        for v in VERDICTS:
            tp = sum(r.expected_verdict == v and r.got_verdict == v for r in self.scored)
            pred = sum(r.got_verdict == v for r in self.scored)
            actual = sum(r.expected_verdict == v for r in self.scored)
            out[v] = {"precision": tp / pred if pred else 0.0,
                      "recall": tp / actual if actual else 0.0, "support": actual}
        return out

    def as_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "cases": len(self.results),
            "errors": sum(bool(r.error) for r in self.results),
            "verdict_accuracy": round(self.verdict_accuracy(), 3),
            "false_pass_rate": round(self.false_pass_rate(), 3),
            "relevance_accuracy": round(self.relevance_accuracy(), 3),
            "per_verdict": {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in self.per_verdict().items()},
            "misses": [r.__dict__ for r in self.results
                       if r.error or r.expected_verdict != r.got_verdict],
        }

    def render(self) -> str:
        d = self.as_dict()
        lines = [
            f"eval backend={d['backend']} cases={d['cases']} errors={d['errors']}",
            f"  verdict accuracy   {d['verdict_accuracy']:.0%}",
            f"  false-pass rate    {d['false_pass_rate']:.0%}  (non-pass items the gate lets through)",
            f"  relevance accuracy {d['relevance_accuracy']:.0%}  (band high >= {self.relevance_threshold})",
            "  per verdict (precision / recall / support):",
        ]
        for v, m in d["per_verdict"].items():
            lines.append(f"    {v:<8} {m['precision']:.2f} / {m['recall']:.2f} / {int(m['support'])}")
        for miss in d["misses"]:
            why = miss["error"] or f"expected {miss['expected_verdict']}, got {miss['got_verdict']}"
            lines.append(f"  miss: {miss['id']}: {why}")
        return "\n".join(lines)


def load_cases(path: str | Path | None = None) -> list[dict[str, Any]]:
    if path:
        text = Path(path).read_text(encoding="utf-8")
    else:
        text = resources.files("ai_news_pipeline.data").joinpath("eval_cases.yaml").read_text(encoding="utf-8")
    cases = (yaml.safe_load(text) or {}).get("cases") or []
    for c in cases:
        if c.get("verdict") not in VERDICTS or c.get("relevance") not in ("high", "low"):
            raise ValueError(f"eval case {c.get('id')!r} has a bad label")
    return cases


def run_eval(backend: LLMBackend, cases: list[dict[str, Any]], categories: dict[str, str],
             relevance_threshold: int = 6) -> EvalReport:
    report = EvalReport(backend=backend.name, relevance_threshold=relevance_threshold)
    schema = score_schema(list(categories))
    for c in cases:
        row = {"source_name": "eval", "source_weight": c.get("source_weight", 1.0),
               "title": c["title"], "summary": c.get("summary", "")}
        res = CaseResult(c["id"], c["verdict"], "", c["relevance"], None)
        try:
            s = backend.complete_json(purpose="score", system=SCORE_SYSTEM,
                                      user=score_input(row, categories), schema=schema)
            res.got_relevance = s["relevance"]
            vrow = {**row, "relevance": s["relevance"], "novelty": s["novelty"],
                    "actionability": s["actionability"], "category": s["category"]}
            v = backend.complete_json(purpose="verify", system=VERIFY_SYSTEM,
                                      user=verify_input(vrow), schema=VERIFY_SCHEMA)
            res.got_verdict = v["verdict"]
        except QuotaExhausted:
            raise
        except LLMError as exc:
            res.error = str(exc)
        report.results.append(res)
    return report
