from pathlib import Path

import pytest

from ai_news_pipeline.sources.registry import SourcesError, load_sources, run_ingest
from ai_news_pipeline.sources.rss import RSSAdapter, entry_to_article


def test_rss_adapter_parses_fixture(data_dir: Path):
    adapter = RSSAdapter("v", "Vendor", {"path": str(data_dir / "demo_feeds/vendor_blog.xml")})
    items = list(adapter.fetch(10))
    assert len(items) == 3  # the entry without a link is skipped
    first = items[0]
    assert first.title.startswith("Introducing Nimbus-7B")
    assert "<p>" not in first.summary and "pip install nimbus-run" in first.summary
    assert first.published_at == "2026-03-10T09:00:00Z"
    assert first.url_canonical == "https://vendor.example.com/blog/nimbus-7b"
    assert first.source_slug == "v"


def test_rss_adapter_reads_atom_and_respects_limit(data_dir: Path):
    adapter = RSSAdapter("r", "R", {"path": str(data_dir / "demo_feeds/research_notes.xml")})
    items = list(adapter.fetch(2))
    assert len(items) == 2
    assert items[0].published_at == "2026-03-10T08:00:00Z"


def test_missing_feed_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        list(RSSAdapter("x", "X", {"path": str(tmp_path / "nope.xml")}).fetch(5))


def _write(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "sources.yaml"
    p.write_text(body, encoding="utf-8")
    return p


@pytest.mark.parametrize("body,needle", [
    ("nothing: here", "top-level"),
    ("sources:\n  - {slug: Bad_Slug, type: rss, config: {url: http://x}}", "invalid slug"),
    ("sources:\n  - {slug: a, type: rss, config: {url: http://x}}\n  - {slug: a, type: rss, config: {url: http://y}}", "duplicate"),
    ("sources:\n  - {slug: a, type: atomz, config: {url: http://x}}", "unknown type"),
    ("sources:\n  - {slug: a, type: rss, weight: 9, config: {url: http://x}}", "outside"),
    ("sources:\n  - {slug: a, type: rss, weight: heavy, config: {url: http://x}}", "number"),
    ("sources:\n  - {slug: a, type: rss, config: {}}", "exactly one"),
    ("sources:\n  - {slug: a, type: rss, config: {url: http://x, path: f.xml}}", "exactly one"),
])
def test_sources_validation(tmp_path: Path, body: str, needle: str):
    with pytest.raises(SourcesError, match=needle):
        load_sources(_write(tmp_path, body))


def test_shipped_example_sources_file_is_valid():
    root = Path(__file__).resolve().parents[1]
    sources = load_sources(root / "sources.yaml")
    enabled = [s for s in sources if s.enabled]
    assert 3 <= len(enabled) <= 5
    assert all(s.config.get("url", "").startswith("https://") for s in enabled)


def test_ingest_dedupes_across_feeds_and_survives_a_broken_source(data_dir: Path, store):
    sources = load_sources(data_dir / "demo_sources.yaml")
    rep = run_ingest(sources, store, per_source_limit=20)
    assert rep.fetched == 12
    assert rep.dropped_in_batch == 2          # utm/www re-post + re-syndicated copy
    assert rep.stored == 10
    assert list(rep.failed_sources) == ["unreachable-feed"]
    # the higher-weight vendor feed won the dedupe race
    row = store.conn.execute(
        "SELECT source_slug FROM articles WHERE url_canonical = ?",
        ("https://vendor.example.com/blog/nimbus-7b",)).fetchone()
    assert row["source_slug"] == "example-vendor"
    again = run_ingest(sources, store, per_source_limit=20)
    assert again.stored == 0 and again.duplicates_in_store == 10


def test_disabled_sources_are_not_fetched(tmp_path: Path, data_dir: Path, store):
    feed = data_dir / "demo_feeds/vendor_blog.xml"
    p = _write(tmp_path, f"sources:\n  - {{slug: a, type: rss, enabled: false, config: {{path: '{feed}'}}}}")
    rep = run_ingest(load_sources(p), store)
    assert rep.fetched == 0 and store.article_count() == 0


@pytest.mark.parametrize("link, ok", [
    ("https://example.com/post", True),
    ("http://example.com/post?id=1", True),
    ("javascript:alert(1)", False),
    ("data:text/html,hi", False),
    ("ftp://example.com/file", False),
    ("https://user:pass@example.com/post", False),
    ("https:///no-host", False),
    ("https://example.com/a b", False),
    ("https://example.com/a<script>", False),
    ("https://example.com/a)](javascript:alert(1)", False),
    ("https://example.com/a\"onmouseover=x", False),
    ("https://example.com/\u202egnp.exe", False),
    ("https://example.com/a%20b?q=1&r=(2)", True),
])
def test_only_plain_http_links_are_ingested(link, ok):
    entry = {"link": link, "title": "A title", "summary": "Some text."}
    assert (entry_to_article(entry, "Feed", "feed") is not None) is ok
