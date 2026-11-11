"""One adapter for every model call.

Stages only ever see `LLMBackend.complete_json(...)`. Which model answers is
a config choice: `mock` (offline, deterministic), `anthropic` (SDK) or
`command` (any CLI that reads a prompt on stdin and prints JSON).
"""
from __future__ import annotations

from typing import Any

from .base import LLMBackend, LLMError, QuotaExhausted, SchemaError, validate

__all__ = ["LLMBackend", "LLMError", "QuotaExhausted", "SchemaError", "validate", "make_backend"]


def make_backend(name: str, options: dict[str, Any] | None = None) -> LLMBackend:
    """Build the configured backend. Imports are lazy so `mock` needs no extras."""
    options = options or {}
    name = (name or "mock").strip().lower()
    if name == "mock":
        from .mock import MockBackend
        return MockBackend()
    if name == "anthropic":
        from .anthropic_backend import AnthropicBackend
        return AnthropicBackend(models=options.get("models") or {}, effort=options.get("effort") or {})
    if name == "command":
        from .command_backend import CommandBackend
        return CommandBackend(command=options.get("command") or "", timeout_s=float(options.get("timeout_s", 300)),
                              env=tuple(options.get("command_env") or ()))
    raise ValueError(f"unknown llm backend {name!r}; expected mock | anthropic | command")
