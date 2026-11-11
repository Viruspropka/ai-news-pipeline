"""Sink interface."""
from __future__ import annotations

from abc import ABC, abstractmethod

from ..models import Brief


class SinkError(RuntimeError):
    pass


class Sink(ABC):
    name: str = "sink"

    @abstractmethod
    def deliver(self, brief: Brief) -> str:
        """Deliver the brief; return a short detail (path, message ids). Raise on failure."""
