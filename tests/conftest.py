from __future__ import annotations

from datetime import datetime, timezone
from importlib import resources
from pathlib import Path
from typing import Any

import pytest

from ai_news_pipeline.config import Settings
from ai_news_pipeline.llm.base import LLMBackend
from ai_news_pipeline.models import RawArticle, Source
from ai_news_pipeline.sources.base import canonicalise_url, hash_content
from ai_news_pipeline.store import Store

NOW = datetime(2026, 3, 11, 8, 0, tzinfo=timezone.utc)


@pytest.fixture
def data_dir() -> Path:
    return Path(str(resources.files("ai_news_pipeline.data")))


@pytest.fixture
def store() -> Store:
    s = Store(":memory:")
    yield s
    s.close()


@pytest.fixture
def settings(tmp_path: Path, data_dir: Path) -> Settings:
    return Settings(
        sources_file=data_dir / "demo_sources.yaml",
        db_path=tmp_path / "news.sqlite",
        state_dir=tmp_path / "state",
        base_dir=tmp_path,
        sinks=[{"type": "file", "dir": "briefs"}],
    )


def make_article(url: str, title: str, summary: str = "", slug: str = "src-a",
                 published: str | None = "2026-03-10T10:00:00Z") -> RawArticle:
    return RawArticle(url=url, url_canonical=canonicalise_url(url), title=title, summary=summary,
                      author=None, content_hash=hash_content(title, summary),
                      published_at=published, source_slug=slug)


def add_source(store: Store, slug: str = "src-a", weight: float = 1.0, name: str | None = None) -> Source:
    s = Source(slug=slug, name=name or slug.upper(), type="rss", weight=weight)
    store.upsert_source(s)
    return s


class ScriptedBackend(LLMBackend):
    """Returns queued answers (or raises queued exceptions) per purpose."""

    name = "scripted"

    def __init__(self, answers: dict[str, list[Any]]) -> None:
        super().__init__()
        self.answers = {k: list(v) for k, v in answers.items()}
        self.prompts: list[tuple[str, str, str]] = []

    def _complete_json(self, *, purpose: str, system: str, user: str, schema: dict[str, Any]) -> dict[str, Any]:
        self.prompts.append((purpose, system, user))
        self.record(purpose)
        queue = self.answers.get(purpose) or []
        if not queue:
            raise AssertionError(f"no scripted answer left for {purpose}")
        answer = queue.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer
