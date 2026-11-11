"""Source adapter base class, URL canonicalisation and content hashing."""
from __future__ import annotations

import hashlib
import re
import string
from abc import ABC, abstractmethod
from html.parser import HTMLParser
from typing import Any, Iterator
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from ..models import RawArticle

# Tracking parameters stripped from every URL before it becomes a dedupe key.
_TRACKING_PARAMS = {
    "fbclid", "gclid", "gbraid", "wbraid", "msclkid",
    "mc_cid", "mc_eid", "mkt_tok",
    "ref", "ref_src", "ref_url", "referrer",
    "icid", "_ga", "_gl",
}


# RFC 3986 unreserved, reserved and "%", minus "[" and "]": anything else
# (spaces, quotes, <>, backticks, control or bidi characters) would need
# escaping wherever the link is rendered, and without "]" a feed link cannot
# close a Markdown link in the .md file. Such a link is not ingested at all.
# (Brackets are only legal in IPv6 hosts, which news feeds do not use.)
_URL_CHARS = frozenset(string.ascii_letters + string.digits + "-._~:/?#@!$&'()*+,;=%")


def is_publishable_url(raw_url: str) -> bool:
    """True for a plain http(s) URL with a host and no user:password part.

    Feed links end up in the published brief, so a feed cannot smuggle in a
    javascript:, data: or credential-carrying link."""
    try:
        parts = urlparse((raw_url or "").strip())
    except ValueError:
        return False
    raw = (raw_url or "").strip()
    return (parts.scheme.lower() in ("http", "https") and bool(parts.hostname)
            and "@" not in parts.netloc and all(c in _URL_CHARS for c in raw))


def canonicalise_url(raw_url: str) -> str:
    """Produce a stable dedupe key from a feed URL. Idempotent.

    Lower-cases scheme and host, drops "www.", the fragment, tracking
    parameters (utm_* and the list above), sorts the remaining query
    parameters and drops a trailing slash on non-root paths.
    """
    if not raw_url or not raw_url.strip():
        return ""
    parts = urlparse(raw_url.strip())
    scheme = (parts.scheme or "https").lower()
    host = (parts.netloc or "").lower()
    if host.startswith("www."):
        host = host[4:]
    path = parts.path or "/"
    if len(path) > 1 and path.endswith("/"):
        path = path[:-1]
    pairs = [
        (k, v)
        for k, v in parse_qsl(parts.query, keep_blank_values=False)
        if k.lower() not in _TRACKING_PARAMS and not k.lower().startswith("utm_")
    ]
    pairs.sort()
    return urlunparse((scheme, host, path, "", urlencode(pairs, doseq=True), ""))


_WS = re.compile(r"\s+", re.UNICODE)
_TAGS = re.compile(r"<[^>]+>")


def _norm_text(text: str) -> str:
    return _WS.sub(" ", _TAGS.sub(" ", text)).strip().lower()


def hash_content(title: str, summary: str | None) -> str:
    """SHA-256 of the normalised title and the first 500 summary chars.

    Two feeds that carry the same announcement under different URLs produce
    the same hash, so the copy is dropped.
    """
    h = hashlib.sha256()
    h.update(_norm_text(title)[:300].encode("utf-8"))
    h.update(b"\x1f")
    h.update(_norm_text(summary or "")[:500].encode("utf-8"))
    return h.hexdigest()


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        if tag in ("script", "style"):
            self._skip += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style") and self._skip:
            self._skip -= 1

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self.parts.append(data)


def strip_html(text: str) -> str:
    """Plain text from an HTML fragment (stdlib only), whitespace collapsed."""
    if not text:
        return ""
    parser = _TextExtractor()
    parser.feed(text)
    parser.close()
    return " ".join(" ".join(parser.parts).split())


class SourceAdapter(ABC):
    """Turns one configured source into a stream of `RawArticle`.

    Adapters never touch the store: persistence, dedupe and per-source caps
    live in the registry, in one place.
    """

    type_: str = ""

    def __init__(self, slug: str, name: str, config: dict[str, Any]) -> None:
        self.slug = slug
        self.name = name
        self.config = config

    @abstractmethod
    def fetch(self, limit: int) -> Iterator[RawArticle]:
        """Yield up to `limit` articles, newest first where the feed allows."""

    def __repr__(self) -> str:
        return f"<{type(self).__name__} slug={self.slug!r}>"
