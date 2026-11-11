"""Backend contract, error types, JSON extraction and a small schema validator."""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any


class LLMError(RuntimeError):
    """A model call failed in a way that only affects this one item."""


class QuotaExhausted(LLMError):
    """The model plan or credit is used up. The run must stop, not retry.

    `retry_after_s` is the provider's hint when it gives one.
    """

    def __init__(self, message: str, retry_after_s: float | None = None) -> None:
        super().__init__(message)
        self.retry_after_s = retry_after_s


class SchemaError(LLMError):
    """The model answered, but the JSON does not match the schema."""


class LLMBackend(ABC):
    """Every backend returns a JSON object that already passed `validate`."""

    name: str = "base"

    def __init__(self) -> None:
        self.usage: dict[str, dict[str, int]] = {}

    def model_for(self, purpose: str) -> str:
        return self.name

    def record(self, purpose: str, input_tokens: int = 0, output_tokens: int = 0) -> None:
        u = self.usage.setdefault(purpose, {"calls": 0, "input_tokens": 0, "output_tokens": 0})
        u["calls"] += 1
        u["input_tokens"] += int(input_tokens or 0)
        u["output_tokens"] += int(output_tokens or 0)

    def complete_json(self, *, purpose: str, system: str, user: str,
                      schema: dict[str, Any]) -> dict[str, Any]:
        obj = self._complete_json(purpose=purpose, system=system, user=user, schema=schema)
        validate(obj, schema)
        return obj

    @abstractmethod
    def _complete_json(self, *, purpose: str, system: str, user: str,
                       schema: dict[str, Any]) -> dict[str, Any]:
        """Backend-specific call; must return a dict (validation happens above)."""


def extract_json_object(raw: str) -> dict[str, Any]:
    """First complete top-level JSON object in `raw` (tolerates prose or code fences around it)."""
    if not raw or not raw.strip():
        raise SchemaError("empty model output")
    decoder = json.JSONDecoder()
    for i, ch in enumerate(raw):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(raw[i:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    raise SchemaError(f"no JSON object in model output: {raw[:200]!r}")


_TYPES: dict[str, tuple[type, ...]] = {
    "object": (dict,), "array": (list,), "string": (str,),
    "integer": (int,), "number": (int, float), "boolean": (bool,),
}


def validate(value: Any, schema: dict[str, Any], path: str = "$") -> None:
    """Validate the JSON-Schema subset the stages use; raise SchemaError on the first problem.

    Supported: type, properties, required, additionalProperties=false, enum,
    minimum, maximum, minLength, maxLength, items, minItems, maxItems.
    """
    t = schema.get("type")
    if t:
        expected = _TYPES[t]
        if isinstance(value, bool) and t in ("integer", "number"):
            raise SchemaError(f"{path}: expected {t}, got boolean")
        if not isinstance(value, expected):
            raise SchemaError(f"{path}: expected {t}, got {type(value).__name__}")
    if "enum" in schema and value not in schema["enum"]:
        raise SchemaError(f"{path}: {value!r} not in {schema['enum']}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            raise SchemaError(f"{path}: {value} < minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            raise SchemaError(f"{path}: {value} > maximum {schema['maximum']}")
    if isinstance(value, str):
        if len(value) < schema.get("minLength", 0):
            raise SchemaError(f"{path}: string shorter than {schema['minLength']}")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            raise SchemaError(f"{path}: string longer than {schema['maxLength']}")
    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0):
            raise SchemaError(f"{path}: fewer than {schema['minItems']} items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            raise SchemaError(f"{path}: more than {schema['maxItems']} items")
        if "items" in schema:
            for i, item in enumerate(value):
                validate(item, schema["items"], f"{path}[{i}]")
    if isinstance(value, dict):
        props = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in value:
                raise SchemaError(f"{path}: missing required key {key!r}")
        if schema.get("additionalProperties") is False:
            extra = set(value) - set(props)
            if extra:
                raise SchemaError(f"{path}: unexpected keys {sorted(extra)}")
        for key, sub in props.items():
            if key in value:
                validate(value[key], sub, f"{path}.{key}")
