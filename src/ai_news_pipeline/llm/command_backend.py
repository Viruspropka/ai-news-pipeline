"""Any command-line model client as a backend.

The prompt goes to the command on stdin; the command must print a JSON object
on stdout (prose around it is tolerated). This keeps the pipeline vendor
neutral: a subscription CLI, a local model runner or a shell script all work.

    ANP_LLM_BACKEND=command
    ANP_LLM_COMMAND="my-model-cli --json"

When the command fails with exit code 75, or fails and its last stderr lines
report a used-up plan or rate limit (see QUOTA_MARKERS), the backend raises
QuotaExhausted so the run stops instead of hammering a
plan that will refuse every call until it resets.

The command gets a minimal environment (PATH, HOME, locale and the like)
plus only the variables listed in llm.command_env, so pipeline secrets such as
TELEGRAM_BOT_TOKEN never reach it. Run it without tool or network permissions
where the client allows it: it reads untrusted article text.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import signal
import subprocess
from typing import Any

from .base import LLMBackend, LLMError, QuotaExhausted, extract_json_object

QUOTA_MARKERS = (
    "usage limit", "rate limit", "quota exceeded", "quota exhausted", "limit reached",
    "credit balance is too low", "insufficient_quota", "out of credits",
)
_RESET = re.compile(r"(?:retry after|resets? in)\s+(\d+)\s*(s|sec|seconds|m|min|minutes|h|hours)\b", re.I)


def _retry_after(text: str) -> float | None:
    m = _RESET.search(text)
    if not m:
        return None
    n, unit = int(m.group(1)), m.group(2).lower()
    return n * (3600 if unit.startswith("h") else 60 if unit.startswith("m") else 1)


# The command reads untrusted article text, so treat it as a process that could
# be talked into leaking what it can see: it gets an allowlisted environment.
# Variables the client really needs (its own login token, a proxy) are named
# by the operator in llm.command_env / ANP_LLM_COMMAND_ENV.
BASE_ENV = ("PATH", "HOME", "USER", "LOGNAME", "SHELL", "LANG", "TERM", "TMPDIR", "TZ")
BASE_ENV_PREFIXES = ("LC_", "XDG_")
# A command may also report a used-up plan with this exit code (EX_TEMPFAIL).
QUOTA_EXIT_CODE = 75


def child_env(extra: tuple[str, ...] = ()) -> dict[str, str]:
    keep = set(BASE_ENV) | set(extra)
    return {k: v for k, v in os.environ.items() if k in keep or k.startswith(BASE_ENV_PREFIXES)}


def diagnose_returncode(rc: int) -> str:
    """Human cause for a non-zero exit (143 = SIGTERM, 137 = SIGKILL/OOM, ...)."""
    if rc < 0:
        try:
            return f"killed by {signal.Signals(-rc).name}"
        except ValueError:
            return f"killed by signal {-rc}"
    return {
        124: "timed out (rc=124)",
        127: "command not found (rc=127)",
        137: "killed, probably out of memory (rc=137)",
        143: "terminated by SIGTERM (rc=143)",
    }.get(rc, f"exited with rc={rc}")


class CommandBackend(LLMBackend):
    name = "command"

    def __init__(self, command: str, timeout_s: float = 300.0, env: tuple[str, ...] = ()) -> None:
        super().__init__()
        self.env_names = tuple(env)
        if not command.strip():
            raise LLMError("command backend needs ANP_LLM_COMMAND (or llm.command in config)")
        self.argv = shlex.split(command)
        self.timeout_s = timeout_s

    def model_for(self, purpose: str) -> str:
        return f"command:{os.path.basename(self.argv[0])}"

    def _complete_json(self, *, purpose: str, system: str, user: str,
                       schema: dict[str, Any]) -> dict[str, Any]:
        prompt = (
            f"{system}\n\nReturn ONLY one JSON object that matches this JSON schema, "
            f"no prose:\n{json.dumps(schema)}\n\nINPUT:\n{user}\n"
        )
        try:
            proc = subprocess.run(
                self.argv, input=prompt, capture_output=True, text=True,
                timeout=self.timeout_s, check=False, env=child_env(self.env_names),
            )
        except subprocess.TimeoutExpired as exc:
            raise LLMError(f"{purpose}: command timed out after {self.timeout_s:.0f}s") from exc
        except OSError as exc:
            raise LLMError(f"{purpose}: cannot start {self.argv[0]!r}: {exc}") from exc

        if proc.returncode != 0:
            combined = f"{proc.stdout}\n{proc.stderr}"
            # Quota counts only on a failed call, and only from the exit code or
            # the last lines of stderr: stdout can hold model output that quotes
            # article text such as "rate limit", and that must not stop the run.
            err_tail = "\n".join(proc.stderr.strip().splitlines()[-5:])
            if proc.returncode == QUOTA_EXIT_CODE or any(m in err_tail.lower() for m in QUOTA_MARKERS):
                raise QuotaExhausted(
                    f"{purpose}: model plan exhausted: {err_tail[-300:] or diagnose_returncode(proc.returncode)}",
                    retry_after_s=_retry_after(err_tail),
                )
            tail = combined.strip()[-300:] or "(no output)"
            raise LLMError(f"{purpose}: command {diagnose_returncode(proc.returncode)}: {tail}")
        self.record(purpose)
        return extract_json_object(proc.stdout)
