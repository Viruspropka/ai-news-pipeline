"""Deterministic offline backend.

It reads the same JSON input the real models get and answers with keyword
heuristics. That is enough to run the whole pipeline, the demo and the tests
without network or keys, and it gives the eval a transparent baseline to
compare a real model against. It does not translate: briefs come out in the
input language whatever `output_language` says.
"""
from __future__ import annotations

import json
import re
from typing import Any

from .base import LLMBackend, LLMError

TOPIC = (
    "llm", "llms", "language model", "agent", "agents", "agentic", "model", "models", "transformer",
    "fine-tune", "fine-tuning", "fine-tuned",
    "rag", "retrieval", "inference", "embedding", "embeddings", "prompt", "prompts", "mcp", "tool use",
    "benchmark", "open-weight", "reasoning", "multimodal", "diffusion", "tokenizer",
)
ARTEFACT = (
    "github", "repo", "repos", "pip install", "open-source", "open source", "api", "sdk",
    "weights", "paper", "arxiv", "benchmark", "benchmarks", "demo", "code", "dataset",
    "datasets", "eval", "evals", "notebook", "cli", "release notes", "changelog",
)
FRESH = ("introducing", "launch", "launches", "released", "releases", "release", "announces",
         "announced", "new", "today", "now available")
STALE = ("top 10", "top ten", "roundup", "recap", "best of", "you need to know", "weekly", "ultimate guide")
HYPE = (
    "revolutionary", "changes everything", "agi", "game-changer", "game changer",
    "mind-blowing", "will replace", "the end of", "unprecedented", "insane",
)
THIN = (
    "hires", "appoints", "joins as", "partnership", "partners with", "steps down",
    "new office", "webinar", "conference", "event recap", "named as",
)
DUBIOUS = (
    "you won't believe", "leaked", "rumor", "rumour", "insiders say", "allegedly",
    "shocking", "secret",
)
CATEGORY_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("startup", ("raises", "funding", "series a", "series b", "seed round", "valuation", "acquires")),
    ("research", ("paper", "arxiv", "we propose", "study", "researchers")),
    ("tutorial", ("tutorial", "guide", "how to", "walkthrough", "step by step")),
    ("agents", ("agent", "agentic", "mcp", "tool use", "automation", "workflow")),
    ("models", ("open-weight", "weights", "parameters", "language model", "reasoning model",
                "embedding model")),
    ("devtools", ("sdk", "cli", "library", "framework", "ide", "api", "pip install", "server")),
    ("content", ("video", "image generation", "text-to-speech", "audio", "music")),
)


def _hits(text: str, words: tuple[str, ...]) -> list[str]:
    """Whole-word / whole-phrase matches only ("cli" must not match "clients")."""
    return [w for w in words if re.search(rf"(?<![\w-]){re.escape(w)}(?![\w-])", text)]


def _clamp(n: float) -> int:
    return max(0, min(10, int(round(n))))


def _first_sentence(text: str, max_words: int = 25) -> str:
    sentence = re.split(r"(?<=[.!?])\s+", text.strip(), maxsplit=1)[0]
    words = sentence.split()
    return " ".join(words[:max_words]).rstrip(",;:")


class MockBackend(LLMBackend):
    name = "mock"

    def model_for(self, purpose: str) -> str:
        return "mock-heuristic-v1"

    def _complete_json(self, *, purpose: str, system: str, user: str,
                       schema: dict[str, Any]) -> dict[str, Any]:
        try:
            payload = json.loads(user)
        except json.JSONDecodeError as exc:
            raise LLMError(f"mock backend expects JSON input: {exc}") from exc
        self.record(purpose)
        if purpose == "score":
            return self.score(payload["article"])
        if purpose == "verify":
            return self.verify(payload["article"])
        if purpose == "brief":
            return self.brief(payload)
        raise LLMError(f"mock backend has no rule for purpose {purpose!r}")

    @staticmethod
    def score(article: dict[str, Any]) -> dict[str, Any]:
        text = f"{article.get('title', '')} {article.get('summary') or ''}".lower()
        topic = _hits(text, TOPIC)
        relevance = _clamp(2 + 2 * len(topic)) if topic else 2
        actionability = _clamp(2 + 2 * len(_hits(text, ARTEFACT)))
        novelty = _clamp(5 + 2 * len(_hits(text, FRESH)) - 3 * len(_hits(text, STALE)))
        category = "other"
        if topic:
            for code, words in CATEGORY_RULES:
                if _hits(text, words):
                    category = code
                    break
        one_line = _first_sentence(article.get("summary") or article.get("title") or "")
        if len(one_line) < 5:
            one_line = (article.get("title") or "untitled item")[:200]
        return {
            "relevance": relevance,
            "novelty": novelty,
            "actionability": actionability,
            "category": category,
            "one_line": one_line[:200],
        }

    @staticmethod
    def verify(article: dict[str, Any]) -> dict[str, Any]:
        text = f"{article.get('title', '')} {article.get('summary') or ''}".lower()
        artefacts = _hits(text, ARTEFACT)
        hype, thin, dubious = _hits(text, HYPE), _hits(text, THIN), _hits(text, DUBIOUS)
        primary = float(article.get("source_weight", 1.0)) >= 1.3
        if dubious:
            verdict, flags = "dubious", ["unverifiable-claim", *dubious[:2]]
        elif hype and not artefacts:
            verdict, flags = "hype", ["no-artefact", *hype[:2]]
        elif thin and not artefacts:
            verdict, flags = "thin", ["no-technical-content", *thin[:2]]
        elif artefacts or primary:
            verdict = "pass"
            flags = [f"has-{a.replace(' ', '-')}" for a in artefacts[:3]] or ["primary-source"]
        else:
            verdict, flags = "thin", ["no-concrete-artefact"]
        evidence = len(artefacts) + len(hype) + len(thin) + len(dubious)
        return {
            "verdict": verdict,
            "flags": [f[:40] for f in flags][:6],
            "confidence": round(min(0.95, 0.55 + 0.1 * evidence), 2),
        }

    @staticmethod
    def brief(payload: dict[str, Any]) -> dict[str, Any]:
        items = []
        counts: dict[str, int] = {}
        for c in payload.get("candidates", []):
            items.append({
                "id": int(c["id"]),
                "headline": (c.get("title") or "")[:120] or "Untitled",
                "blurb": c.get("one_line") or c.get("title") or "No summary.",
            })
            counts[c.get("category", "other")] = counts.get(c.get("category", "other"), 0) + 1
        labels = payload.get("categories", {})
        top = max(counts, key=lambda k: counts[k]) if counts else "other"
        takeaway = f"Most of today's signal is in {labels.get(top, top)}."
        return {"items": items, "takeaway": takeaway}
