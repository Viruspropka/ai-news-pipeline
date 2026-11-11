"""Declarative source list (YAML) and the ingest step.

Sources file schema, one entry per feed:

    sources:
      - slug: hf-blog          # unique, kebab-case, used as the store key
        name: Hugging Face Blog
        type: rss              # adapter type, see ADAPTERS
        weight: 1.4            # multiplies the composite score, 0.1..3.0
        enabled: true
        config: {url: https://huggingface.co/blog/feed.xml}
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..models import RawArticle, Source
from ..stages.dedupe import dedupe_in_batch
from ..store import Store
from .base import SourceAdapter
from .rss import RSSAdapter

log = logging.getLogger(__name__)

ADAPTERS: dict[str, type[SourceAdapter]] = {"rss": RSSAdapter}

_SLUG = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
WEIGHT_MIN, WEIGHT_MAX = 0.1, 3.0


class SourcesError(ValueError):
    """The sources file is malformed."""


def load_sources(path: str | Path) -> list[Source]:
    """Parse and validate the sources file. Raises SourcesError on any problem.

    Validation is strict on purpose: a typo in a weight or a duplicate slug
    should fail loudly at start, not skew rankings silently for weeks.
    """
    path = Path(path)
    try:
        doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise SourcesError(f"cannot read sources file {path}: {exc}") from exc
    entries = doc.get("sources") if isinstance(doc, dict) else None
    if not isinstance(entries, list):
        raise SourcesError(f"{path}: top-level `sources:` list is missing")

    out: list[Source] = []
    seen: set[str] = set()
    for i, raw in enumerate(entries):
        where = f"{path.name} entry #{i + 1}"
        if not isinstance(raw, dict):
            raise SourcesError(f"{where}: expected a mapping")
        slug = str(raw.get("slug", "")).strip()
        if not _SLUG.match(slug):
            raise SourcesError(f"{where}: invalid slug {slug!r} (kebab-case, a-z0-9-)")
        if slug in seen:
            raise SourcesError(f"{where}: duplicate slug {slug!r}")
        seen.add(slug)
        type_ = str(raw.get("type", "")).strip()
        if type_ not in ADAPTERS:
            raise SourcesError(f"{where} ({slug}): unknown type {type_!r}; known: {sorted(ADAPTERS)}")
        try:
            weight = float(raw.get("weight", 1.0))
        except (TypeError, ValueError) as exc:
            raise SourcesError(f"{where} ({slug}): weight must be a number") from exc
        if not WEIGHT_MIN <= weight <= WEIGHT_MAX:
            raise SourcesError(f"{where} ({slug}): weight {weight} outside {WEIGHT_MIN}..{WEIGHT_MAX}")
        config = raw.get("config") or {}
        if not isinstance(config, dict):
            raise SourcesError(f"{where} ({slug}): config must be a mapping")
        if type_ == "rss" and bool(config.get("url")) == bool(config.get("path")):
            raise SourcesError(f"{where} ({slug}): rss needs exactly one of config.url / config.path")
        config = {**config, "_base_dir": str(path.parent.resolve())}
        out.append(Source(
            slug=slug,
            name=str(raw.get("name") or slug),
            type=type_,
            weight=weight,
            enabled=bool(raw.get("enabled", True)),
            config=config,
        ))
    return out


def adapter_for(source: Source) -> SourceAdapter:
    return ADAPTERS[source.type](source.slug, source.name, source.config)


@dataclass
class IngestReport:
    fetched: int = 0
    dropped_in_batch: int = 0
    duplicates_in_store: int = 0
    stored: int = 0
    failed_sources: dict[str, str] = field(default_factory=dict)


def run_ingest(
    sources: list[Source],
    store: Store,
    *,
    per_source_limit: int = 30,
    only_slug: str | None = None,
) -> IngestReport:
    """Fetch every enabled source, dedupe the batch, store new articles.

    Sources are fetched in weight order (highest first) so that, when two
    sources carry the same story, the higher-weight one wins the dedupe.
    One failing source is logged and counted; it never stops the others.
    """
    report = IngestReport()
    active = [s for s in sources if s.enabled and (only_slug is None or s.slug == only_slug)]
    active.sort(key=lambda s: s.weight, reverse=True)

    batch: list[RawArticle] = []
    by_slug = {s.slug: s for s in active}
    for source in active:
        store.upsert_source(source)
        try:
            items = list(adapter_for(source).fetch(per_source_limit))
        except Exception as exc:  # noqa: BLE001 - one bad feed must not stop the run
            report.failed_sources[source.slug] = f"{type(exc).__name__}: {exc}"
            log.warning("ingest: source %s failed: %s", source.slug, exc)
            continue
        report.fetched += len(items)
        batch.extend(items)

    unique = list(dedupe_in_batch(batch))
    report.dropped_in_batch = len(batch) - len(unique)
    for art in unique:
        if store.insert_article(art, by_slug[art.source_slug]):
            report.stored += 1
        else:
            report.duplicates_in_store += 1
    log.info(
        "ingest: fetched=%d in-batch-dups=%d store-dups=%d stored=%d failed=%d",
        report.fetched, report.dropped_in_batch, report.duplicates_in_store,
        report.stored, len(report.failed_sources),
    )
    return report
