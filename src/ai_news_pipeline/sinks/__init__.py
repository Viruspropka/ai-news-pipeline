"""Delivery sinks: where a finished brief goes."""
from __future__ import annotations

import logging
from typing import Any, Callable

from ..models import Brief
from .base import Sink, SinkError
from .console import ConsoleSink
from .file import FileSink
from .telegram import TelegramSink

log = logging.getLogger(__name__)

__all__ = ["Sink", "SinkError", "ConsoleSink", "FileSink", "TelegramSink", "make_sinks", "deliver"]


def make_sinks(specs: list[dict[str, Any]], *, base_dir: str = ".",
               env: dict[str, str] | None = None) -> list[Sink]:
    """Build sinks from config entries like {type: file, dir: data/briefs}."""
    builders: dict[str, Callable[[dict[str, Any]], Sink]] = {
        "console": lambda s: ConsoleSink(),
        "file": lambda s: FileSink(s.get("dir", "data/briefs"), base_dir=base_dir),
        "telegram": lambda s: TelegramSink.from_env(env),
    }
    sinks: list[Sink] = []
    for spec in specs:
        kind = str(spec.get("type", "")).strip()
        if kind not in builders:
            raise ValueError(f"unknown sink type {kind!r}; expected one of {sorted(builders)}")
        sinks.append(builders[kind](spec))
    return sinks


def deliver(brief: Brief, sinks: list[Sink]) -> dict[str, str]:
    """Send to every sink. One failing sink never blocks the others.

    Returns {sink name: "ok: <detail>" | "error: <cause>"}.
    """
    results: dict[str, str] = {}
    for sink in sinks:
        try:
            results[sink.name] = f"ok: {sink.deliver(brief)}"
        except Exception as exc:  # noqa: BLE001 - recorded per sink
            log.error("sink %s failed: %s", sink.name, exc)
            results[sink.name] = f"error: {type(exc).__name__}: {exc}"
    return results
