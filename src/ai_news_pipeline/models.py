"""Plain data objects shared by every stage."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Source:
    """One entry of the sources file."""

    slug: str
    name: str
    type: str
    weight: float = 1.0
    enabled: bool = True
    config: dict[str, Any] = field(default_factory=dict)


@dataclass
class RawArticle:
    """One article as fetched, before it is stored.

    `url_canonical` is the cross-run dedupe key (UNIQUE in the store);
    `content_hash` catches re-syndicated copies that live under another URL.
    """

    url: str
    url_canonical: str
    title: str
    summary: str | None
    author: str | None
    content_hash: str
    published_at: str | None  # ISO 8601 UTC ("...Z") or None
    source_slug: str = ""


@dataclass(frozen=True)
class BriefItem:
    article_id: int
    category: str
    headline: str
    blurb: str
    url: str
    source_name: str


@dataclass
class Brief:
    """A composed brief, ready for the sinks."""

    title: str
    language: str
    created_at: str
    items: list[BriefItem]
    takeaway: str
    footer: str
    text: str = ""  # rendered plain text, filled by the brief stage
