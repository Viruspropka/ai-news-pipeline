import pytest

from ai_news_pipeline.sources.base import canonicalise_url, hash_content, strip_html


@pytest.mark.parametrize("raw,expected", [
    ("https://www.Example.com/a/?utm_source=x&b=2&a=1#frag", "https://example.com/a?a=1&b=2"),
    ("HTTP://example.com/", "http://example.com/"),
    ("https://example.com/post?fbclid=123&ref=hn", "https://example.com/post"),
    ("https://example.com/post?UTM_Medium=x&id=7", "https://example.com/post?id=7"),
    ("", ""),
    ("   ", ""),
])
def test_canonicalise(raw, expected):
    assert canonicalise_url(raw) == expected


def test_canonicalise_is_idempotent():
    url = "https://www.example.com/a/b/?utm_campaign=c&z=1&y=2"
    once = canonicalise_url(url)
    assert canonicalise_url(once) == once


def test_hash_ignores_case_whitespace_and_tags():
    a = hash_content("New  Model", "<p>Hello   world</p>")
    b = hash_content("new model", "hello world")
    assert a == b
    assert hash_content("new model", "other") != a


def test_hash_only_reads_first_500_summary_chars():
    base = "x" * 500
    assert hash_content("t", base + "tail one") == hash_content("t", base + "tail two")


def test_strip_html_drops_script_and_style():
    html = "<p>Hi <b>there</b></p><script>alert(1)</script><style>p{}</style> &amp; bye"
    assert strip_html(html) == "Hi there & bye"
