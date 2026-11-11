"""Claude via the official `anthropic` SDK (optional extra: `pip install .[anthropic]`).

Typed JSON comes from structured outputs (`output_config.format` with a JSON
schema), not from parsing prose. The schema sent to the API keeps only the
structure (types, enums, required keys); numeric and length bounds are
checked locally by `validate`, so a model that answers 11 on a 0..10 axis is
rejected the same way with every backend.

Credentials follow the SDK's own resolution (ANTHROPIC_API_KEY and friends).
"""
from __future__ import annotations

import copy
import json
from typing import Any

from .base import LLMBackend, LLMError, QuotaExhausted, SchemaError

DEFAULT_MODELS = {
    "score": "claude-haiku-5-5",
    "verify": "claude-sonnet-5-5",
    "brief": "claude-sonnet-5-5",
}
DEFAULT_EFFORT = {"score": "low", "verify": "medium", "brief": "medium"}

_LOCAL_ONLY_KEYS = ("minimum", "maximum", "minLength", "maxLength", "minItems", "maxItems")


def api_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Structure-only copy of `schema` with additionalProperties=false on every object."""
    out = copy.deepcopy(schema)

    def walk(node: Any) -> None:
        if not isinstance(node, dict):
            return
        for key in _LOCAL_ONLY_KEYS:
            node.pop(key, None)
        if node.get("type") == "object":
            node["additionalProperties"] = False
            node.setdefault("required", list(node.get("properties", {})))
        for sub in node.get("properties", {}).values():
            walk(sub)
        walk(node.get("items"))

    walk(out)
    return out


class AnthropicBackend(LLMBackend):
    name = "anthropic"

    def __init__(self, models: dict[str, str] | None = None, effort: dict[str, str] | None = None,
                 client: Any = None, max_tokens: int = 4000) -> None:
        super().__init__()
        self.models = {**DEFAULT_MODELS, **(models or {})}
        self.effort = {**DEFAULT_EFFORT, **(effort or {})}
        self.max_tokens = max_tokens
        self._client = client

    def model_for(self, purpose: str) -> str:
        return self.models.get(purpose, self.models["verify"])

    def client(self) -> Any:
        if self._client is None:
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover - depends on the extra
                raise LLMError("the anthropic backend needs `pip install .[anthropic]`") from exc
            self._client = anthropic.Anthropic()  # SDK retries 429/5xx itself (max_retries=2)
        return self._client

    def _complete_json(self, *, purpose: str, system: str, user: str,
                       schema: dict[str, Any]) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model_for(purpose),
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "output_config": {
                "format": {"type": "json_schema", "schema": api_schema(schema)},
                "effort": self.effort.get(purpose, "medium"),
            },
        }
        try:
            resp = self.client().messages.create(**kwargs)
        except Exception as exc:  # noqa: BLE001 - classified below, never swallowed
            raise _classify(exc) from exc

        usage = getattr(resp, "usage", None)
        self.record(purpose, getattr(usage, "input_tokens", 0), getattr(usage, "output_tokens", 0))
        stop = getattr(resp, "stop_reason", None)
        if stop == "refusal":
            raise LLMError(f"{purpose}: model declined the request")
        if stop == "max_tokens":
            raise LLMError(f"{purpose}: output hit max_tokens={self.max_tokens}")
        text = next((b.text for b in resp.content if getattr(b, "type", "") == "text"), "")
        try:
            obj = json.loads(text)
        except json.JSONDecodeError as exc:
            raise SchemaError(f"{purpose}: structured output is not JSON: {text[:200]!r}") from exc
        if not isinstance(obj, dict):
            raise SchemaError(f"{purpose}: structured output is not an object")
        return obj


def _classify(exc: Exception) -> LLMError:
    """Map SDK errors: used-up credit or a rate limit that survived the SDK retries
    stops the run (QuotaExhausted); everything else fails only this item."""
    status = getattr(exc, "status_code", None)
    err_type = getattr(exc, "type", None)
    if status == 402 or err_type == "billing_error":
        return QuotaExhausted(f"billing: {exc}")
    if status == 429 or err_type == "rate_limit_error":
        retry_after = None
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", None) or {}
        try:
            retry_after = float(headers.get("retry-after")) if headers.get("retry-after") else None
        except (TypeError, ValueError):
            retry_after = None
        return QuotaExhausted(f"rate limit after retries: {exc}", retry_after_s=retry_after)
    return LLMError(f"{type(exc).__name__}: {exc}")
