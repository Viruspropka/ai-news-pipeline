"""SQLite store: sources, articles, scores, verifications, briefs.

One file, stdlib `sqlite3`, WAL mode. Every query is parameterised.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

from .models import RawArticle, Source

SCHEMA = """
CREATE TABLE IF NOT EXISTS sources (
    slug        TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    type        TEXT NOT NULL,
    weight      REAL NOT NULL DEFAULT 1.0
);
CREATE TABLE IF NOT EXISTS articles (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    source_slug   TEXT NOT NULL REFERENCES sources(slug),
    url           TEXT NOT NULL,
    url_canonical TEXT NOT NULL UNIQUE,
    title         TEXT NOT NULL,
    summary       TEXT,
    author        TEXT,
    content_hash  TEXT NOT NULL,
    published_at  TEXT,
    ingested_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_articles_hash ON articles(content_hash);
CREATE TABLE IF NOT EXISTS scores (
    article_id    INTEGER PRIMARY KEY REFERENCES articles(id) ON DELETE CASCADE,
    relevance     INTEGER NOT NULL,
    novelty       INTEGER NOT NULL,
    actionability INTEGER NOT NULL,
    composite     REAL NOT NULL,
    category      TEXT NOT NULL,
    one_line      TEXT NOT NULL,
    model         TEXT NOT NULL,
    scored_at     TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS verifications (
    article_id  INTEGER PRIMARY KEY REFERENCES articles(id) ON DELETE CASCADE,
    verdict     TEXT NOT NULL,
    flags_json  TEXT NOT NULL,
    confidence  REAL NOT NULL,
    model       TEXT NOT NULL,
    verified_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS briefs (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at       TEXT NOT NULL,
    language         TEXT NOT NULL,
    text             TEXT NOT NULL,
    article_ids_json TEXT NOT NULL,
    delivery_json    TEXT NOT NULL
);
"""


def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Store:
    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        if self.path != ":memory:":
            self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    # sources / articles -------------------------------------------------

    def upsert_source(self, s: Source) -> None:
        self.conn.execute(
            "INSERT INTO sources(slug, name, type, weight) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(slug) DO UPDATE SET name=excluded.name, type=excluded.type, "
            "weight=excluded.weight",
            (s.slug, s.name, s.type, s.weight),
        )
        self.conn.commit()

    def insert_article(self, art: RawArticle, source: Source, now: datetime | None = None) -> bool:
        """Store a new article. False when its URL or content hash is already stored."""
        if self.conn.execute(
            "SELECT 1 FROM articles WHERE content_hash = ? LIMIT 1", (art.content_hash,)
        ).fetchone():
            return False
        ingested = iso(now or utcnow())
        # A feed date in the future would keep the item inside every later
        # brief window, so it would come back after each repost cooldown.
        # Both strings use the same fixed UTC format, so they compare as text.
        published = art.published_at
        if published and published > ingested:
            published = ingested
        try:
            self.conn.execute(
                "INSERT INTO articles(source_slug, url, url_canonical, title, summary, author, "
                "content_hash, published_at, ingested_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (source.slug, art.url, art.url_canonical, art.title, art.summary, art.author,
                 art.content_hash, published, ingested),
            )
        except sqlite3.IntegrityError:
            return False
        self.conn.commit()
        return True

    def article_count(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0])

    def get_article(self, article_id: int) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT a.*, s.name AS source_name, s.weight AS source_weight "
            "FROM articles a JOIN sources s ON s.slug = a.source_slug WHERE a.id = ?",
            (article_id,),
        ).fetchone()

    # scores -------------------------------------------------------------

    def unscored_ids(self, limit: int) -> list[int]:
        rows = self.conn.execute(
            "SELECT a.id FROM articles a LEFT JOIN scores s ON s.article_id = a.id "
            "WHERE s.article_id IS NULL ORDER BY a.id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [int(r[0]) for r in rows]

    def save_score(self, article_id: int, *, relevance: int, novelty: int, actionability: int,
                   composite: float, category: str, one_line: str, model: str) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO scores(article_id, relevance, novelty, actionability, "
            "composite, category, one_line, model, scored_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (article_id, relevance, novelty, actionability, composite, category, one_line,
             model, iso(utcnow())),
        )
        self.conn.commit()

    # verifications ------------------------------------------------------

    def verify_candidates(self, *, min_relevance: int, limit: int,
                          exclude_ids: Iterable[int] = ()) -> list[sqlite3.Row]:
        """Scored, not yet verified, relevant enough; best composite first."""
        excluded = set(exclude_ids)
        rows = self.conn.execute(
            "SELECT a.id, a.title, a.summary, a.url, s.relevance, s.novelty, s.actionability, "
            "s.composite, s.category, src.name AS source_name, src.weight AS source_weight "
            "FROM articles a JOIN scores s ON s.article_id = a.id "
            "JOIN sources src ON src.slug = a.source_slug "
            "LEFT JOIN verifications v ON v.article_id = a.id "
            "WHERE v.article_id IS NULL AND s.relevance >= ? "
            "ORDER BY s.composite DESC, a.id DESC",
            (min_relevance,),
        ).fetchall()
        return [r for r in rows if int(r["id"]) not in excluded][:limit]

    def save_verification(self, article_id: int, *, verdict: str, flags: list[str],
                          confidence: float, model: str) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO verifications(article_id, verdict, flags_json, confidence, "
            "model, verified_at) VALUES (?, ?, ?, ?, ?, ?)",
            (article_id, verdict, json.dumps(flags), confidence, model, iso(utcnow())),
        )
        self.conn.commit()

    def verification(self, article_id: int) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT verdict, flags_json, confidence FROM verifications WHERE article_id = ?",
            (article_id,),
        ).fetchone()
        if row is None:
            return None
        return {"verdict": row["verdict"], "flags": json.loads(row["flags_json"]),
                "confidence": row["confidence"]}

    # briefs -------------------------------------------------------------

    def brief_candidates(self, *, since_iso: str, min_relevance: int, limit: int) -> list[sqlite3.Row]:
        """Scored articles inside the window, with their verdict (NULL = unverified).

        `url` is the canonical URL: tracking parameters never reach a reader.

        An article without a published date counts by its ingest time.
        """
        return self.conn.execute(
            "SELECT a.id, a.title, a.summary, a.url_canonical AS url, a.published_at, a.source_slug, "
            "src.name AS source_name, src.weight AS source_weight, s.relevance, s.novelty, "
            "s.actionability, s.composite, s.category, s.one_line, v.verdict, v.flags_json "
            "FROM articles a JOIN scores s ON s.article_id = a.id "
            "JOIN sources src ON src.slug = a.source_slug "
            "LEFT JOIN verifications v ON v.article_id = a.id "
            "WHERE COALESCE(a.published_at, a.ingested_at) >= ? AND s.relevance >= ? "
            "ORDER BY s.composite DESC, a.id DESC LIMIT ?",
            (since_iso, min_relevance, limit),
        ).fetchall()

    def recently_briefed_ids(self, hours: float, now: datetime | None = None) -> set[int]:
        since = iso((now or utcnow()) - timedelta(hours=hours))
        ids: set[int] = set()
        for row in self.conn.execute(
            "SELECT article_ids_json FROM briefs WHERE created_at >= ?", (since,)
        ):
            ids.update(int(i) for i in json.loads(row[0]))
        return ids

    def save_brief(self, *, created_at: str, language: str, text: str,
                   article_ids: list[int], delivery: dict[str, str]) -> int:
        cur = self.conn.execute(
            "INSERT INTO briefs(created_at, language, text, article_ids_json, delivery_json) "
            "VALUES (?, ?, ?, ?, ?)",
            (created_at, language, text, json.dumps(article_ids), json.dumps(delivery)),
        )
        self.conn.commit()
        return int(cur.lastrowid)
