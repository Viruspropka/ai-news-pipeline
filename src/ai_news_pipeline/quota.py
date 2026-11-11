"""Stop-on-quota guard: a used-up model plan stops the run until it resets.

Without it a scheduled job whose plan is exhausted calls the model again on
every tick, fails every item, and fills the logs with noise. With it, the
first QuotaExhausted writes a small state file; later runs see it, exit at
once with a clear message and exit code 75 (EX_TEMPFAIL), and resume by
themselves after the stop time.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

EXIT_STOPPED = 75
MAX_STOP = timedelta(hours=24)


@dataclass(frozen=True)
class Stop:
    until: datetime
    backend: str
    reason: str

    def describe(self) -> str:
        return (f"stopped: model plan exhausted ({self.backend}) until "
                f"{self.until.strftime('%Y-%m-%d %H:%M UTC')} - {self.reason}")


class QuotaGuard:
    def __init__(self, state_dir: str | Path, default_cooldown_min: float = 60.0) -> None:
        self.path = Path(state_dir) / "quota_stop.json"
        self.default_cooldown = timedelta(minutes=default_cooldown_min)

    def active(self, now: datetime | None = None) -> Stop | None:
        """The current stop, or None. An expired or unreadable file means no stop."""
        now = now or datetime.now(timezone.utc)
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            stop = Stop(datetime.fromisoformat(data["until"]), data["backend"], data["reason"])
        except (OSError, ValueError, KeyError, TypeError):
            return None
        return stop if stop.until > now else None

    def trip(self, backend: str, reason: str, retry_after_s: float | None = None,
             now: datetime | None = None) -> Stop:
        now = now or datetime.now(timezone.utc)
        wait = timedelta(seconds=retry_after_s) if retry_after_s else self.default_cooldown
        # A reset time comes from provider or command text; cap it so one odd
        # message cannot stop the pipeline for days.
        wait = min(wait, MAX_STOP)
        stop = Stop(now + wait, backend, reason[:300])
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({
            "until": stop.until.isoformat(), "backend": backend, "reason": stop.reason,
        }), encoding="utf-8")
        return stop

    def clear(self) -> None:
        self.path.unlink(missing_ok=True)
