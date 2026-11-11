"""In-batch dedupe. Pure function: no store, no LLM, no I/O.

Cross-run dedupe is the store's job (UNIQUE url_canonical plus a content-hash
lookup). Inside one ingest pass the same announcement often arrives through
two feeds at once; this drops every later copy that shares either key.
"""
from __future__ import annotations

from typing import Iterable, Iterator

from ..models import RawArticle


def dedupe_in_batch(articles: Iterable[RawArticle]) -> Iterator[RawArticle]:
    """Yield the first article for each content hash and canonical URL.

    Order matters: callers pass the highest-weight source first. Empty keys
    are never treated as equal - refusing to dedupe is safer than merging
    two unrelated articles.
    """
    seen_hashes: set[str] = set()
    seen_urls: set[str] = set()
    for art in articles:
        h, u = art.content_hash or "", art.url_canonical or ""
        if (h and h in seen_hashes) or (u and u in seen_urls):
            continue
        if h:
            seen_hashes.add(h)
        if u:
            seen_urls.add(u)
        yield art
