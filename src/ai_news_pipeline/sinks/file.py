"""Write each brief as a Markdown file plus one JSON line in an index."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from ..models import Brief
from .base import Sink


class FileSink(Sink):
    name = "file"

    def __init__(self, directory: str | Path, base_dir: str | Path = ".") -> None:
        d = Path(directory)
        self.dir = d if d.is_absolute() else Path(base_dir) / d

    def deliver(self, brief: Brief) -> str:
        self.dir.mkdir(parents=True, exist_ok=True)
        stamp = brief.created_at.replace(":", "").replace("-", "")
        path = self.dir / f"brief-{stamp}.md"
        n = 2
        while path.exists():
            path = self.dir / f"brief-{stamp}_{n}.md"
            n += 1
        path.write_text(brief.text, encoding="utf-8")
        record = {k: v for k, v in asdict(brief).items() if k != "text"}
        record["file"] = path.name
        with (self.dir / "briefs.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        return str(path)
