"""Command line: run | demo | eval | sources."""
from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from importlib import resources
from pathlib import Path
from typing import Any

from . import __version__
from .config import ConfigError, Settings, load_settings
from .evals import load_cases, run_eval
from .llm import LLMBackend, QuotaExhausted, make_backend
from .llm.mock import MockBackend
from .pipeline import STAGES, RunReport, run
from .quota import EXIT_STOPPED
from .sources.registry import SourcesError, load_sources

DEMO_NOW = datetime(2026, 3, 11, 8, 0, tzinfo=timezone.utc)


def format_report(r: RunReport) -> str:
    """Stage-by-stage summary, one line per stage."""
    if r.stopped and r.ingest is None and r.score is None:
        return f"  {r.stopped.describe()} (run skipped)"
    lines = []
    if r.ingest:
        i = r.ingest
        lines.append(f"  ingest : fetched={i.fetched} in-batch duplicates={i.dropped_in_batch} "
                     f"already stored={i.duplicates_in_store} new={i.stored}")
        for slug, why in i.failed_sources.items():
            lines.append(f"           source {slug} failed (others continued): {why}")
    if r.score:
        lines.append(f"  score  : scored={r.score.done} skipped={r.score.skipped}")
    if r.verify:
        lines.append(f"  verify : verified={r.verify.done} skipped={r.verify.skipped}")
    if r.selection:
        s = r.selection
        lines.append(f"  select : in window={s.considered} blocked by verdict={len(s.blocked_by_verdict)} "
                     f"not verified={len(s.not_verified)} briefed recently={len(s.recently_briefed)} cut by caps={s.cut_by_diversity} "
                     f"selected={len(s.selected)}")
    if r.compose and (r.compose.error or r.compose.dropped_banned or r.compose.dropped_unknown_ids):
        c = r.compose
        lines.append(f"  compose: dropped unknown ids={c.dropped_unknown_ids} dropped for banned "
                     f"phrases={c.dropped_banned} note={c.error or '-'}")
    if r.brief:
        n = len(r.brief.items)
        lines.append(f"  brief  : {n} item{'s' if n != 1 else ''} in {r.brief.language}, stored as brief #{r.brief_id}")
        for sink, result in r.delivery.items():
            lines.append(f"  deliver: {sink}: {result}")
    elif r.selection is not None:
        lines.append("  brief  : nothing new to publish")
    if r.usage:
        calls = ", ".join(f"{k}={v['calls']}" for k, v in sorted(r.usage.items()))
        lines.append(f"  model calls: {calls}")
    if r.stopped:
        lines.append(f"  {r.stopped.describe()}")
    return "\n".join(lines)


def _settings(args: argparse.Namespace) -> Settings:
    return load_settings(getattr(args, "config", None))


def cmd_run(args: argparse.Namespace) -> int:
    settings = _settings(args)
    stages = tuple(s.strip() for s in args.stages.split(",")) if args.stages else STAGES
    unknown = set(stages) - set(STAGES)
    if unknown:
        print(f"unknown stage(s): {sorted(unknown)}; known: {', '.join(STAGES)}", file=sys.stderr)
        return 2
    report = run(settings, stages=stages)
    print(format_report(report))
    return EXIT_STOPPED if report.stopped else 0


class _PlanUsedUp(MockBackend):
    """Demo-only backend that behaves like a model plan that has run out."""

    name = "demo-exhausted"

    def _complete_json(self, **kwargs: Any) -> dict[str, Any]:
        raise QuotaExhausted("usage limit reached, resets in 45 minutes", retry_after_s=45 * 60)


def cmd_demo(args: argparse.Namespace) -> int:
    out = Path(args.out) if args.out else Path(tempfile.mkdtemp(prefix="ai-news-demo-"))
    out.mkdir(parents=True, exist_ok=True)
    data = resources.files("ai_news_pipeline.data")
    with resources.as_file(data) as data_dir:
        base = Settings(
            sources_file=Path(data_dir) / "demo_sources.yaml",
            db_path=out / "news.sqlite",
            state_dir=out / "state",
            base_dir=out,
            sinks=[{"type": "console"}, {"type": "file", "dir": "briefs"}],
        )
        base.brief = replace(base.brief, language=args.language)
        print(f"ai-news-pipeline {__version__} offline demo: synthetic feeds, mock model, output in {out}\n")
        print(f"Run 1 at {DEMO_NOW:%Y-%m-%d %H:%M} UTC: three synthetic feeds and one broken source")
        print("-" * 72)
        r1 = run(base, now=DEMO_NOW)
        print("-" * 72)
        print(format_report(r1))

        print("\nRun 2 one hour later, same feeds: nothing new is stored; only the item the")
        print("per-source cap held back in run 1 is still unbriefed")
        r2 = run(base, now=DEMO_NOW + timedelta(hours=1))
        print(format_report(r2))

        print("\nRun 3 on a fresh database with a model plan that has run out")
        r3 = run(replace(base, db_path=out / "quota-demo.sqlite"), now=DEMO_NOW + timedelta(hours=2),
                 backend=_PlanUsedUp())
        print(format_report(r3))
        print("\nRun 4 ten minutes later: the guard skips the run instead of calling the model again")
        r4 = run(replace(base, db_path=out / "quota-demo.sqlite"), now=DEMO_NOW + timedelta(hours=2, minutes=10))
        print(format_report(r4))
    ok = (bool(r1.brief) and r2.ingest is not None and r2.ingest.stored == 0
          and r3.stopped is not None and r4.stopped is not None and r4.ingest is None)
    print(f"\ndemo {'OK' if ok else 'FAILED'}: briefs written to {out / 'briefs'}")
    return 0 if ok else 1


def cmd_eval(args: argparse.Namespace) -> int:
    settings = _settings(args)
    backend: LLMBackend = make_backend(args.backend or settings.llm_backend, settings.llm_options)
    cases = load_cases(args.cases)
    try:
        report = run_eval(backend, cases, settings.brief.categories,
                          relevance_threshold=settings.brief.min_relevance)
    except QuotaExhausted as exc:
        print(f"eval stopped: model plan exhausted: {exc}", file=sys.stderr)
        return EXIT_STOPPED
    print(report.render())
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report.as_dict(), indent=2), encoding="utf-8")
    failed = report.verdict_accuracy() < args.min_accuracy or report.false_pass_rate() > args.max_false_pass
    return 1 if failed else 0


def cmd_sources(args: argparse.Namespace) -> int:
    settings = _settings(args)
    for s in load_sources(settings.sources_file):
        target = s.config.get("url") or s.config.get("path")
        print(f"{'on ' if s.enabled else 'off'} {s.slug:<22} w={s.weight:<4} {s.type:<4} {target}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ai-news-pipeline",
                                description="LLM-scored AI news: ingest, dedupe, score, verify, brief.")
    p.add_argument("--version", action="version", version=__version__)
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="one pipeline pass with the configured feeds, model and sinks")
    r.add_argument("--config", help="YAML config (default: $ANP_CONFIG or built-in defaults)")
    r.add_argument("--stages", help=f"comma list, default {','.join(STAGES)}")
    r.set_defaults(func=cmd_run)

    d = sub.add_parser("demo", help="offline demo on synthetic feeds with the mock model")
    d.add_argument("--out", help="directory for the demo database and briefs (default: a temp dir)")
    d.add_argument("--language", default="English", help="output language passed to the model")
    d.set_defaults(func=cmd_demo)

    e = sub.add_parser("eval", help="score+verify the labelled cases and report accuracy")
    e.add_argument("--config")
    e.add_argument("--backend", choices=["mock", "anthropic", "command"])
    e.add_argument("--cases", help="YAML with labelled cases (default: bundled set)")
    e.add_argument("--json-out", help="also write the metrics as JSON")
    e.add_argument("--min-accuracy", type=float, default=0.0, help="exit 1 below this verdict accuracy")
    e.add_argument("--max-false-pass", type=float, default=1.0, help="exit 1 above this false-pass rate")
    e.set_defaults(func=cmd_eval)

    s = sub.add_parser("sources", help="validate and list the sources file")
    s.add_argument("--config")
    s.set_defaults(func=cmd_sources)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.ERROR,
                        format="%(levelname)s %(name)s: %(message)s")
    try:
        return int(args.func(args))
    except (ConfigError, SourcesError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
