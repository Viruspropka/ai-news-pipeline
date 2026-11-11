"""RSS / Atom adapter built on `feedparser`.

Config keys (exactly one of them):
    url:  an http(s) feed URL
    path: a local feed file; relative paths resolve against the directory of
          the sources file (used by the offline demo and the tests)
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import feedparser

from ..models import RawArticle
from .base import SourceAdapter, canonicalise_url, hash_content, is_publishable_url, strip_html

log = logging.getLogger(__name__)

USER_AGENT = "ai-news-pipeline/0.1 (+https://github.com/)"
SUMMARY_CAP = 1500


class RSSAdapter(SourceAdapter):
    type_ = "rss"

    def _target(self) -> str | None:
        if self.config.get("path"):
            path = Path(self.config["path"])
            base = self.config.get("_base_dir")
            if not path.is_absolute() and base:
                path = Path(base) / path
            return str(path)
        return self.config.get("url")

    def fetch(self, limit: int) -> Iterator[RawArticle]:
        target = self._target()
        if not target:
            log.warning("rss source %s has neither `url` nor `path`", self.slug)
            return
        if self.config.get("path") and not Path(target).is_file():
            raise FileNotFoundError(f"feed file not found: {Path(target).name}")

        parsed = feedparser.parse(target, request_headers={"User-Agent": USER_AGENT})
        entries = parsed.get("entries") or []
        if getattr(parsed, "bozo", 0) and not entries:
            # A fatal parse or transport error: surface it so the registry
            # counts the source as failed instead of silently empty.
            raise ValueError(f"feed parse failed: {getattr(parsed, 'bozo_exception', 'unknown')}")

        feed_title = ""
        try:
            feed_title = (parsed.feed.get("title") or "").strip()
        except (AttributeError, TypeError):
            pass

        count = 0
        for entry in entries:
            if count >= limit:
                break
            article = entry_to_article(entry, feed_title, self.slug)
            if article is None:
                continue
            count += 1
            yield article


def entry_to_article(entry: Any, feed_title: str, source_slug: str) -> RawArticle | None:
    """Map one feedparser entry to a `RawArticle`; None when the title is missing
    or the link is missing or not a plain http(s) URL."""
    link = (entry.get("link") or "").strip()
    title = strip_html((entry.get("title") or "").strip())
    if not link or not title or not is_publishable_url(link):
        return None

    raw_summary = entry.get("summary") or entry.get("description") or ""
    if not raw_summary:
        content = entry.get("content")
        if isinstance(content, list) and content and isinstance(content[0], dict):
            raw_summary = content[0].get("value") or ""
    summary = strip_html(raw_summary)[:SUMMARY_CAP] or None

    author = (entry.get("author") or feed_title or "").strip() or None

    return RawArticle(
        url=link,
        url_canonical=canonicalise_url(link),
        title=title,
        summary=summary,
        author=author,
        content_hash=hash_content(title, summary),
        published_at=_published(entry),
        source_slug=source_slug,
    )


def _published(entry: Any) -> str | None:
    parsed_time = entry.get("published_parsed") or entry.get("updated_parsed")
    if not parsed_time:
        return None
    try:
        dt = datetime(*parsed_time[:6], tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
