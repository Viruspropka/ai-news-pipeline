"""Settings: defaults, then an optional YAML file, then environment variables.

Relative paths in the file resolve against the file's directory, so a config
works from any working directory. Environment overrides (all optional):

    ANP_CONFIG           path of the YAML config file
    ANP_SOURCES          sources file
    ANP_DB_PATH          SQLite database path
    ANP_STATE_DIR        directory for the quota-stop state file
    ANP_OUTPUT_LANGUAGE  language of the brief, e.g. "German"
    ANP_LLM_BACKEND      mock | anthropic | command
    ANP_LLM_COMMAND      command line for the `command` backend
    ANP_LLM_COMMAND_ENV  comma list of env var names passed to that command
    ANP_SINKS            comma list, e.g. "console,file,telegram"
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from .stages.brief import DEFAULT_BANNED, BriefConfig
from .stages.common import DEFAULT_CATEGORIES
from .stages.score import DEFAULT_WEIGHTS



class ConfigError(ValueError):
    pass


@dataclass
class Settings:
    sources_file: Path = Path("sources.yaml")
    db_path: Path = Path("data/news.sqlite")
    state_dir: Path = Path("data/state")
    base_dir: Path = Path(".")
    llm_backend: str = "mock"
    llm_options: dict[str, Any] = field(default_factory=dict)
    per_source_limit: int = 30
    score_limit: int = 200
    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    verify_top_k: int = 30
    quota_cooldown_min: float = 60.0
    brief: BriefConfig = field(default_factory=BriefConfig)
    sinks: list[dict[str, Any]] = field(default_factory=lambda: [{"type": "console"}])


def _section(doc: dict[str, Any], name: str) -> dict[str, Any]:
    value = doc.get(name) or {}
    if not isinstance(value, dict):
        raise ConfigError(f"config section `{name}` must be a mapping")
    return value


def _int(value: Any, name: str, lo: int, hi: int) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name} must be an integer") from exc
    if not lo <= n <= hi:
        raise ConfigError(f"{name}={n} outside {lo}..{hi}")
    return n


def _float(value: Any, name: str, lo: float, hi: float) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name} must be a number") from exc
    if not lo <= x <= hi:
        raise ConfigError(f"{name}={x} outside {lo}..{hi}")
    return x


def load_settings(path: str | Path | None = None, env: Mapping[str, str] | None = None) -> Settings:
    env = os.environ if env is None else env
    path = path or env.get("ANP_CONFIG") or None
    doc: dict[str, Any] = {}
    base = Path.cwd()
    if path:
        p = Path(path)
        try:
            doc = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError) as exc:
            raise ConfigError(f"cannot read config {p}: {exc}") from exc
        if not isinstance(doc, dict):
            raise ConfigError(f"{p}: top level must be a mapping")
        base = p.resolve().parent

    def resolve(value: Any) -> Path:
        q = Path(str(value)).expanduser()
        return q if q.is_absolute() else base / q

    s = Settings(base_dir=base)
    s.sources_file = resolve(env.get("ANP_SOURCES") or doc.get("sources_file", "sources.yaml"))
    s.db_path = resolve(env.get("ANP_DB_PATH") or doc.get("db_path", "data/news.sqlite"))
    s.state_dir = resolve(env.get("ANP_STATE_DIR") or doc.get("state_dir", "data/state"))

    llm = _section(doc, "llm")
    s.llm_backend = str(env.get("ANP_LLM_BACKEND") or llm.get("backend", "mock")).lower()
    if s.llm_backend not in ("mock", "anthropic", "command"):
        raise ConfigError(f"llm.backend {s.llm_backend!r} must be mock | anthropic | command")
    s.llm_options = {
        "models": dict(llm.get("models") or {}),
        "effort": dict(llm.get("effort") or {}),
        "command": env.get("ANP_LLM_COMMAND") or llm.get("command", ""),
        "timeout_s": llm.get("timeout_s", 300),
        "command_env": _names(env.get("ANP_LLM_COMMAND_ENV"), llm.get("command_env")),
    }
    s.quota_cooldown_min = _float(llm.get("quota_cooldown_minutes", 60), "llm.quota_cooldown_minutes", 1, 10080)

    ingest = _section(doc, "ingest")
    s.per_source_limit = _int(ingest.get("per_source_limit", 30), "ingest.per_source_limit", 1, 500)

    score = _section(doc, "score")
    s.score_limit = _int(score.get("batch_limit", 200), "score.batch_limit", 1, 5000)
    weights = {**DEFAULT_WEIGHTS, **(score.get("weights") or {})}
    s.weights = {k: _float(v, f"score.weights.{k}", 0, 1) for k, v in weights.items()}

    verify = _section(doc, "verify")
    s.verify_top_k = _int(verify.get("top_k", 30), "verify.top_k", 0, 1000)

    b = _section(doc, "brief")
    categories = doc.get("categories") or DEFAULT_CATEGORIES
    if not isinstance(categories, dict) or "other" not in categories:
        raise ConfigError("categories must be a mapping that includes `other`")
    s.brief = BriefConfig(
        language=str(env.get("ANP_OUTPUT_LANGUAGE") or doc.get("output_language", "English")),
        categories={str(k): str(v) for k, v in categories.items()},
        window_hours=_float(b.get("window_hours", 24), "brief.window_hours", 1, 24 * 30),
        min_relevance=_int(b.get("min_relevance", 6), "brief.min_relevance", 0, 10),
        max_items=_int(b.get("max_items", 8), "brief.max_items", 1, 50),
        max_per_category=_int(b.get("max_per_category", 3), "brief.max_per_category", 1, 50),
        max_per_source=_int(b.get("max_per_source", 2), "brief.max_per_source", 1, 50),
        repost_cooldown_hours=_float(b.get("repost_cooldown_hours", 36), "brief.repost_cooldown_hours", 0, 24 * 30),
        title=str(b.get("title", "AI news brief - {date}")),
        disclosure=str(b.get("disclosure", BriefConfig().disclosure)),
        require_verified=_bool(b.get("require_verified", True), "brief.require_verified"),
        banned_phrases=tuple(str(p).lower() for p in (b.get("banned_phrases") or DEFAULT_BANNED)),
    )

    sinks = doc.get("sinks") or [{"type": "console"}]
    if env.get("ANP_SINKS"):
        sinks = [{"type": t.strip()} for t in env["ANP_SINKS"].split(",") if t.strip()]
        file_spec = next((x for x in (doc.get("sinks") or []) if x.get("type") == "file"), None)
        sinks = [file_spec if (x["type"] == "file" and file_spec) else x for x in sinks]
    if not isinstance(sinks, list) or not all(isinstance(x, dict) for x in sinks):
        raise ConfigError("sinks must be a list of mappings like {type: console}")
    s.sinks = sinks
    return s


def _names(env_value: str | None, file_value: Any) -> list[str]:
    """Env var names for the command backend, from a comma string or a YAML list."""
    raw = env_value if env_value is not None else file_value
    if raw is None:
        return []
    items = raw.split(",") if isinstance(raw, str) else raw
    if not isinstance(items, (list, tuple)):
        raise ConfigError("llm.command_env must be a list of variable names")
    names = [str(n).strip() for n in items if str(n).strip()]
    bad = [n for n in names if not n.replace("_", "").isalnum()]
    if bad:
        raise ConfigError(f"llm.command_env has invalid names: {bad}")
    return names


def _bool(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    raise ConfigError(f"{name} must be true or false, got {value!r}")
