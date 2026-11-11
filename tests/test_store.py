from datetime import timedelta

from conftest import NOW, add_source, make_article


def test_insert_rejects_same_url_and_same_content(store):
    src = add_source(store)
    assert store.insert_article(make_article("https://example.com/a", "A", "x"), src)
    assert not store.insert_article(make_article("https://www.example.com/a/", "A2", "y"), src)
    assert not store.insert_article(make_article("https://other.example.org/z", "A", "x"), src)
    assert store.article_count() == 1


def test_unscored_and_candidates(store):
    src = add_source(store, weight=1.5)
    store.insert_article(make_article("https://example.com/a", "A"), src)
    store.insert_article(make_article("https://example.com/b", "B"), src)
    ids = store.unscored_ids(10)
    assert len(ids) == 2
    store.save_score(ids[0], relevance=8, novelty=5, actionability=5, composite=9.0,
                     category="models", one_line="line one", model="m")
    store.save_score(ids[1], relevance=3, novelty=5, actionability=5, composite=3.0,
                     category="other", one_line="line two", model="m")
    assert store.unscored_ids(10) == []
    cands = store.verify_candidates(min_relevance=6, limit=10)
    assert [int(r["id"]) for r in cands] == [ids[0]]
    assert store.verify_candidates(min_relevance=6, limit=10, exclude_ids={ids[0]}) == []


def test_recently_briefed_window(store):
    store.save_brief(created_at="2026-03-10T00:00:00Z", language="English", text="t",
                     article_ids=[1, 2], delivery={})
    assert store.recently_briefed_ids(36, now=NOW) == {1, 2}
    assert store.recently_briefed_ids(36, now=NOW + timedelta(hours=40)) == set()


def test_file_store_creates_parent_dirs(tmp_path):
    from ai_news_pipeline.store import Store
    s = Store(tmp_path / "deep" / "dir" / "db.sqlite")
    assert (tmp_path / "deep" / "dir" / "db.sqlite").exists()
    s.close()


def test_future_feed_date_is_clamped_to_ingest_time(store):
    src = add_source(store)
    store.insert_article(make_article("https://example.com/f", "F", "x", published="2099-01-01T00:00:00Z"), src, now=NOW)
    store.insert_article(make_article("https://example.com/p", "P", "y", published="2026-03-10T10:00:00Z"), src, now=NOW)
    rows = store.conn.execute("SELECT url, published_at FROM articles ORDER BY url").fetchall()
    assert [(r[0], r[1]) for r in rows] == [("https://example.com/f", "2026-03-11T08:00:00Z"),
                                            ("https://example.com/p", "2026-03-10T10:00:00Z")]
