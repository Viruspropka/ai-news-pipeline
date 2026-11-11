from ai_news_pipeline.stages.dedupe import dedupe_in_batch

from conftest import make_article


def test_drops_same_canonical_url_and_keeps_first():
    a = make_article("https://example.com/p?utm_source=a", "One", slug="high")
    b = make_article("https://www.example.com/p/", "One (repost)", "other text", slug="low")
    out = list(dedupe_in_batch([a, b]))
    assert out == [a]


def test_drops_same_content_under_another_url():
    a = make_article("https://example.com/a", "Same", "body")
    b = make_article("https://mirror.example.net/b", "Same", "body")
    assert list(dedupe_in_batch([a, b])) == [a]


def test_empty_keys_are_never_merged():
    a = make_article("https://example.com/a", "A")
    b = make_article("https://example.com/b", "B")
    a.content_hash = b.content_hash = ""
    a.url_canonical = b.url_canonical = ""
    assert len(list(dedupe_in_batch([a, b]))) == 2


def test_order_is_preserved():
    items = [make_article(f"https://example.com/{i}", f"T{i}") for i in range(5)]
    assert list(dedupe_in_batch(items)) == items
