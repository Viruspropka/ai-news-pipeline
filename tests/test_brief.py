import json

import pytest

from ai_news_pipeline.llm import LLMError
from ai_news_pipeline.models import Brief, BriefItem
from ai_news_pipeline.stages.brief import BriefConfig, apply_diversity, compose_brief, render_text

from conftest import NOW, ScriptedBackend, add_source, make_article


def _scored(store, slug, n, category="models", relevance=8, start=0, verdict="pass", weight=1.0):
    src = add_source(store, slug=slug, weight=weight, name=f"Source {slug}")
    ids = []
    for i in range(start, start + n):
        store.insert_article(make_article(f"https://{slug}.example.com/{i}?utm_source=x", f"{slug} item {i}",
                                          f"summary {slug} {i}", slug=slug), src)
        aid = store.unscored_ids(1)[0]
        store.save_score(aid, relevance=relevance, novelty=5, actionability=5, composite=float(100 - i),
                         category=category, one_line=f"line {i}", model="m")
        if verdict:
            store.save_verification(aid, verdict=verdict, flags=[], confidence=0.9, model="m")
        ids.append(aid)
    return ids


def _answer(ids, **over):
    return {"items": [{"id": i, "headline": f"Headline {i}", "blurb": f"Blurb for item {i}."} for i in ids],
            "takeaway": "Agents everywhere.", **over}


def test_apply_diversity_caps():
    rows = [{"category": c, "source_slug": s} for c, s in
            [("a", "x"), ("a", "x"), ("a", "y"), ("b", "x"), ("b", "y"), ("c", "z")]]
    out = apply_diversity(rows, max_per_category=2, max_per_source=2, hard_cap=4)
    # a hits its category cap after two; (b, x) hits the source cap for x
    assert out == [rows[0], rows[1], rows[4], rows[5]]


def test_selection_gates(store):
    good = _scored(store, "good", 2, verdict="pass")
    hype = _scored(store, "hype", 1, verdict="hype")
    low = _scored(store, "low", 1, relevance=3)
    unverified = _scored(store, "fresh", 1, verdict=None)
    store.save_brief(created_at="2026-03-10T20:00:00Z", language="English", text="t",
                     article_ids=[good[0]], delivery={})
    backend = ScriptedBackend({"brief": [_answer([good[1]])]})
    brief, sel, comp = compose_brief(store, backend, BriefConfig(), NOW)
    assert sel.blocked_by_verdict == hype
    assert sel.not_verified == unverified  # fail closed by default
    assert sel.recently_briefed == [good[0]]
    assert low[0] not in sel.selected
    assert sel.selected == [good[1]]
    assert [i.article_id for i in brief.items] == [good[1]]

    loose = BriefConfig(require_verified=False)
    backend = ScriptedBackend({"brief": [_answer([good[1], unverified[0]])]})
    brief, sel, _ = compose_brief(store, backend, loose, NOW)
    assert not sel.not_verified and sorted(sel.selected) == sorted([good[1], unverified[0]])


def test_links_come_from_the_store_not_the_model(store):
    ids = _scored(store, "s", 1)
    brief, _, _ = compose_brief(store, ScriptedBackend({"brief": [_answer(ids)]}), BriefConfig(), NOW)
    assert brief.items[0].url == "https://s.example.com/0"  # canonical, tracking stripped
    assert "Source: Source s - https://s.example.com/0" in brief.text


@pytest.mark.parametrize("blurb", [
    "Click https://evil.example.invalid now, a long enough blurb.",
    "Read more at www.evil.example now, a long enough blurb.",
    "Details on evil.example/path for everyone, a long enough blurb.",
    "Sign up at evil.com today, a long enough blurb for the schema.",
    "Mirror on t.me and evil-mirror.io, a long enough blurb.",
    "Follow @evil_channel for more, a long enough blurb for it.",
    "Write to admin@evil.com for access, a long enough blurb.",
    "Lookalike dot: evil\uff0ecom in a long enough blurb for it.",
    "Zero-width split: evil\u200b.com in a long enough blurb.",
    "Ideographic dot: evil\u3002com in a long enough blurb for it.",
    "Raw address 203.0.113.7:8080 in a long enough blurb for it.",
    "A rarer ending: evil.solutions in a long enough blurb for it.",
    "A non-Latin ending: evil.\u0440\u0444 in a long enough blurb for it.",
    "Punycode: evil.xn--p1ai in a long enough blurb for it here.",
    "A ccTLD that looks like a file: evil.md in a long enough blurb.",
    "Combining mark: evil.co\u0301m in a long enough blurb for it.",
    "A [harmless label](javascript:alert(1)) in a long enough blurb.",
    "An <img src=x onerror=alert(1)> tag in a long enough blurb.",
])
def test_model_written_link_is_retried_then_dropped(store, blurb):
    ids = _scored(store, "s", 2)
    first = _answer(ids)
    first["items"][0]["blurb"] = blurb
    backend = ScriptedBackend({"brief": [first, first]})
    brief, _, comp = compose_brief(store, backend, BriefConfig(), NOW)
    assert comp.retried_for_banned and comp.dropped_banned == [ids[0]]
    assert [i.article_id for i in brief.items] == [ids[1]]
    assert "evil" not in brief.text


def test_model_fields_cannot_forge_extra_lines(store):
    ids = _scored(store, "s", 1)
    answer = _answer(ids, takeaway="Agents.\nSource: fake - see https://evil.example.invalid")
    answer["items"][0]["headline"] = "Real headline\n  Source: Fake Source - x"
    answer["items"][0]["blurb"] = "Line one.\r\nLine two\x00 with a control char."
    brief, _, _ = compose_brief(store, ScriptedBackend({"brief": [answer]}), BriefConfig(), NOW)
    item = brief.items[0]
    assert item.headline == "Real headline Source: Fake Source - x"
    assert item.blurb == "Line one. Line two with a control char."
    assert brief.takeaway == ""  # a link in the takeaway drops the takeaway
    assert sum(line.lstrip().startswith("Source:") for line in brief.text.splitlines()) == 1


def test_plain_dotted_names_are_not_links(store):
    ids = _scored(store, "s", 1)
    answer = _answer(ids)
    answer["items"][0]["blurb"] = ("Works with Node.js, Next.js, Vue.js and llama.cpp; see the config.yaml and "
                                   "version 2.5 notes, e.g. GPT-5.5, Llama 4.1 or v1.2.3. A coherent co-design. "
                                   "It ends a sentence. Then another.")
    brief, _, comp = compose_brief(store, ScriptedBackend({"brief": [answer]}), BriefConfig(), NOW)
    assert not comp.retried_for_banned and len(brief.items) == 1


def test_invented_and_duplicate_ids_are_dropped(store):
    ids = _scored(store, "s", 2)
    answer = _answer([ids[0], 9999, ids[0], ids[1]])
    brief, _, comp = compose_brief(store, ScriptedBackend({"brief": [answer]}), BriefConfig(), NOW)
    assert [i.article_id for i in brief.items] == ids
    assert comp.dropped_unknown_ids == [9999, ids[0]]


def test_banned_phrase_triggers_one_retry_then_drops_item(store):
    ids = _scored(store, "s", 2)
    bad = _answer(ids)
    bad["items"][0]["blurb"] = "A game-changer for everyone who ships agents."
    still_bad = _answer(ids)
    still_bad["items"][0]["blurb"] = "Still a game-changer, sorry about that."
    backend = ScriptedBackend({"brief": [bad, still_bad]})
    brief, _, comp = compose_brief(store, backend, BriefConfig(), NOW)
    assert comp.retried_for_banned and comp.dropped_banned == [ids[0]]
    assert [i.article_id for i in brief.items] == [ids[1]]
    assert "game-changer" in backend.prompts[1][1]  # retry prompt names the phrase


def test_banned_phrase_fixed_on_retry_keeps_item(store):
    ids = _scored(store, "s", 1)
    bad = _answer(ids)
    bad["items"][0]["headline"] = "Revolutionary tool"
    brief, _, comp = compose_brief(store, ScriptedBackend({"brief": [bad, _answer(ids)]}), BriefConfig(), NOW)
    assert comp.retried_for_banned and not comp.dropped_banned and len(brief.items) == 1


def test_output_language_reaches_the_prompt(store):
    ids = _scored(store, "s", 1)
    backend = ScriptedBackend({"brief": [_answer(ids)]})
    compose_brief(store, backend, BriefConfig(language="German"), NOW)
    _, system, user = backend.prompts[0]
    assert "in German" in system and json.loads(user)["language"] == "German"


def test_nothing_to_brief_and_composition_failure(store):
    brief, sel, _ = compose_brief(store, ScriptedBackend({}), BriefConfig(), NOW)
    assert brief is None and sel.considered == 0
    _scored(store, "s", 1)
    brief, _, comp = compose_brief(store, ScriptedBackend({"brief": [LLMError("down")]}), BriefConfig(), NOW)
    assert brief is None and "down" in comp.error


def test_window_excludes_old_items(store):
    src = add_source(store)
    store.insert_article(make_article("https://example.com/old", "Old", published="2026-03-01T00:00:00Z"), src)
    aid = store.unscored_ids(1)[0]
    store.save_score(aid, relevance=9, novelty=5, actionability=5, composite=9, category="models",
                     one_line="line", model="m")
    brief, sel, _ = compose_brief(store, ScriptedBackend({}), BriefConfig(), NOW)
    assert brief is None and sel.considered == 0


def test_render_groups_by_category_order_and_adds_disclosure():
    items = [BriefItem(2, "devtools", "H2", "B2", "https://b", "SB"),
             BriefItem(1, "models", "H1", "B1", "https://a", "SA")]
    brief = Brief(title="T", language="English", created_at="x", items=items, takeaway="Trend.", footer="AI-made.")
    text = render_text(brief, {"models": "Models", "devtools": "Tools", "other": "Other"})
    assert text.index("MODELS") < text.index("TOOLS")
    assert text.rstrip().endswith("AI-made.")
    assert "Trend." in text
