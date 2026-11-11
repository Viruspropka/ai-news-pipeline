"""Print the brief to stdout."""
from __future__ import annotations

import sys
from typing import TextIO

from ..models import Brief
from .base import Sink


class ConsoleSink(Sink):
    name = "console"

    def __init__(self, stream: TextIO | None = None) -> None:
        self.stream = stream

    def deliver(self, brief: Brief) -> str:
        out = self.stream or sys.stdout
        out.write(brief.text)
        out.flush()
        return f"{len(brief.text)} chars"
