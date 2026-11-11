from datetime import timedelta

from ai_news_pipeline.llm import QuotaExhausted
from ai_news_pipeline.llm.mock import MockBackend
from ai_news_pipeline.pipeline import run
from ai_news_pipeline.quota import QuotaGuard
from ai_news_pipeline.store import Store

from conftest import NOW


class CountingMock(MockBackend):
    def __init__(self, fail_after=None):
        super().__init__()
        self.calls = 0
        self.fail_after = fail_after

    def _complete_json(self, **kw):
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise QuotaExhausted("usage limit reached", retry_after_s=600)
        return super()._complete_json(**kw)


def test_full_offline_run_on_demo_feeds(settings, tmp_path):
    rep = run(settings, now=NOW)
    assert rep.ok and rep.ingest.stored == 10 and rep.score.done == 10
    assert rep.brief is not None
    titles = " ".join(i.headline for i in rep.brief.items)
    assert "Nimbus-7B" in titles and "MCP" in titles
    for blocked in ("revolutionary", "Leaked", "partners", "Top 10", "gardening"):
        assert blocked.lower() not in rep.brief.text.lower()
    assert rep.delivery["file"].startswith("ok: ")
    assert list((tmp_path / "briefs").glob("brief-*.md"))
    assert rep.usage["score"]["calls"] == 10


def test_second_run_reuses_store_and_respects_cooldown(settings):
    first = run(settings, now=NOW)
    second = run(settings, now=NOW + timedelta(hours=1))
    assert second.ingest.stored == 0 and second.score.done == 0
    briefed_first = {i.article_id for i in first.brief.items}
    if second.brief:
        assert not briefed_first & {i.article_id for i in second.brief.items}


def test_quota_trip_stops_run_and_later_runs(settings):
    backend = CountingMock(fail_after=3)
    rep = run(settings, now=NOW, backend=backend)
    assert rep.stopped is not None and rep.score is None and rep.brief is None
    assert backend.calls == 4  # three good calls, one refused, then nothing
    assert rep.stopped.until == NOW + timedelta(seconds=600)

    again = CountingMock()
    skipped = run(settings, now=NOW + timedelta(minutes=5), backend=again)
    assert skipped.stopped is not None and skipped.ingest is None and again.calls == 0

    resumed = run(settings, now=NOW + timedelta(minutes=11), backend=CountingMock())
    assert resumed.ok and resumed.score.done == 7  # the three scored before the stop are kept


def test_quota_guard_file_handling(tmp_path):
    g = QuotaGuard(tmp_path, default_cooldown_min=30)
    assert g.active(NOW) is None
    stop = g.trip("mock", "x" * 1000, now=NOW)
    assert stop.until == NOW + timedelta(minutes=30) and len(stop.reason) == 300
    assert g.active(NOW + timedelta(minutes=29)) is not None
    assert g.active(NOW + timedelta(minutes=31)) is None
    g.path.write_text("not json")
    assert g.active(NOW) is None
    g.clear()
    assert not g.path.exists()


def test_quota_stop_is_capped_at_a_day(tmp_path):
    stop = QuotaGuard(tmp_path).trip("mock", "resets in 9999 hours", retry_after_s=9999 * 3600, now=NOW)
    assert stop.until == NOW + timedelta(hours=24)


def test_stage_subset(settings):
    rep = run(settings, stages=("ingest",), now=NOW)
    assert rep.ingest.stored == 10 and rep.score is None and rep.brief is None
    s = Store(settings.db_path)
    assert s.article_count() == 10 and s.unscored_ids(50)
    s.close()
