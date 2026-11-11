"""Run the stages in order: ingest -> score -> verify -> brief -> deliver."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .config import Settings
from .llm import LLMBackend, QuotaExhausted, make_backend
from .models import Brief
from .quota import QuotaGuard, Stop
from .sinks import Sink, deliver, make_sinks
from .sources.registry import IngestReport, load_sources, run_ingest
from .stages.brief import ComposeReport, SelectionReport, compose_brief
from .stages.common import StageReport
from .stages.score import score_pending
from .stages.verify import verify_top
from .store import Store

log = logging.getLogger(__name__)

STAGES = ("ingest", "score", "verify", "brief")


@dataclass
class RunReport:
    ingest: IngestReport | None = None
    score: StageReport | None = None
    verify: StageReport | None = None
    selection: SelectionReport | None = None
    compose: ComposeReport | None = None
    brief: Brief | None = None
    brief_id: int | None = None
    delivery: dict[str, str] = field(default_factory=dict)
    stopped: Stop | None = None
    usage: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.stopped is None


def run(settings: Settings, *, stages: tuple[str, ...] = STAGES, now: datetime | None = None,
        backend: LLMBackend | None = None, sinks: list[Sink] | None = None,
        store: Store | None = None) -> RunReport:
    """One pipeline pass. Never raises for per-item or per-source failures.

    A used-up model plan (QuotaExhausted) trips the quota guard: the rest of
    the run is skipped, and later runs exit early until the stop time passes.
    """
    now = now or datetime.now(timezone.utc)
    report = RunReport()
    guard = QuotaGuard(settings.state_dir, settings.quota_cooldown_min)
    report.stopped = guard.active(now)
    if report.stopped:
        log.warning(report.stopped.describe())
        return report

    own_store = store is None
    store = store or Store(settings.db_path)
    backend = backend or make_backend(settings.llm_backend, settings.llm_options)
    try:
        if "ingest" in stages:
            report.ingest = run_ingest(load_sources(settings.sources_file), store,
                                       per_source_limit=settings.per_source_limit)
        if "score" in stages:
            report.score = score_pending(store, backend, categories=settings.brief.categories,
                                         weights=settings.weights, limit=settings.score_limit)
        if "verify" in stages:
            recent = store.recently_briefed_ids(settings.brief.repost_cooldown_hours, now=now)
            report.verify = verify_top(store, backend, min_relevance=settings.brief.min_relevance,
                                       top_k=settings.verify_top_k, exclude_ids=recent)
        if "brief" in stages:
            brief, report.selection, report.compose = compose_brief(store, backend, settings.brief, now)
            if brief is not None:
                report.brief = brief
                targets = sinks if sinks is not None else make_sinks(settings.sinks, base_dir=str(settings.base_dir))
                report.delivery = deliver(brief, targets)
                report.brief_id = store.save_brief(
                    created_at=brief.created_at, language=brief.language, text=brief.text,
                    article_ids=[i.article_id for i in brief.items], delivery=report.delivery)
    except QuotaExhausted as exc:
        report.stopped = guard.trip(backend.name, str(exc), exc.retry_after_s, now=now)
        log.warning(report.stopped.describe())
    finally:
        report.usage = backend.usage
        if own_store:
            store.close()
    return report
