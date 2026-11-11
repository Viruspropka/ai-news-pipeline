import io
import json
import urllib.error

import pytest

from ai_news_pipeline.models import Brief, BriefItem
from ai_news_pipeline.sinks import ConsoleSink, FileSink, SinkError, TelegramSink, deliver, make_sinks
from ai_news_pipeline.sinks.base import Sink
from ai_news_pipeline.sinks.telegram import split_message


def _brief(text="Title\n\nbody\n"):
    return Brief(title="Title", language="English", created_at="2026-03-11T08:00:00Z",
                 items=[BriefItem(1, "models", "H", "B", "https://x", "S")], takeaway="", footer="", text=text)


def test_console_sink():
    buf = io.StringIO()
    assert ConsoleSink(buf).deliver(_brief()) == "12 chars"
    assert buf.getvalue() == "Title\n\nbody\n"


def test_file_sink_writes_markdown_and_index_without_overwriting(tmp_path):
    sink = FileSink("briefs", base_dir=tmp_path)
    p1 = sink.deliver(_brief())
    p2 = sink.deliver(_brief())
    assert p1 != p2
    lines = (tmp_path / "briefs" / "briefs.jsonl").read_text().splitlines()
    assert len(lines) == 2 and json.loads(lines[0])["items"][0]["url"] == "https://x"


class FakeResp:
    def __init__(self, payload):
        self.payload = payload

    def read(self):
        return json.dumps(self.payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_telegram_sends_plain_text_chunks():
    sent = []

    def opener(req, timeout):
        sent.append((req.full_url, json.loads(req.data)))
        return FakeResp({"ok": True, "result": {"message_id": len(sent)}})

    sink = TelegramSink("123:abc", "-100", opener=opener)
    text = ("para one\n\n" + "y" * 4000 + "\n\n" + "z" * 300)
    detail = sink.deliver(_brief(text))
    assert detail == "message ids [1, 2]"
    assert all("parse_mode" not in body for _, body in sent)
    assert all(body["link_preview_options"] == {"is_disabled": True} for _, body in sent)
    assert sent[0][0].endswith("/bot123:abc/sendMessage") and sent[0][1]["chat_id"] == "-100"


def test_telegram_errors_never_leak_the_token():
    def opener(req, timeout):
        raise urllib.error.HTTPError(req.full_url, 401, "Unauthorized", {}, None)

    sink = TelegramSink("123:secret-token", "-100", opener=opener)
    with pytest.raises(SinkError) as info:
        sink.deliver(_brief())
    assert "secret-token" not in str(info.value) and "secret-token" not in repr(sink)


def test_telegram_refusal():
    sink = TelegramSink("1:t", "c", opener=lambda req, timeout: FakeResp({"ok": False, "description": "chat not found"}))
    with pytest.raises(SinkError, match="chat not found"):
        sink.deliver(_brief())


def test_telegram_requires_env():
    with pytest.raises(SinkError, match="TELEGRAM_BOT_TOKEN"):
        TelegramSink.from_env({})
    assert TelegramSink.from_env({"TELEGRAM_BOT_TOKEN": "1:t", "TELEGRAM_CHAT_ID": "c"}).chat_id == "c"
    with pytest.raises(SinkError, match="does not look like"):
        TelegramSink.from_env({"TELEGRAM_BOT_TOKEN": "1:t/../x?y", "TELEGRAM_CHAT_ID": "c"})


def test_telegram_unexpected_errors_never_leak_the_token():
    def opener(req, timeout):
        raise ValueError(f"weird failure for {req.full_url}")

    sink = TelegramSink("123:secret-token", "-100", opener=opener)
    with pytest.raises(SinkError) as info:
        sink.deliver(_brief())
    assert "secret-token" not in str(info.value) and info.value.__cause__ is None
    assert info.value.__suppress_context__


@pytest.mark.parametrize("text", ["short", "a\n\nb", "x" * 9000, ("line\n" * 2000), ("p" * 100 + "\n\n") * 100])
def test_split_message_limits_and_content(text):
    chunks = split_message(text, limit=4096)
    assert all(0 < len(c) <= 4096 for c in chunks)
    assert "".join(chunks).replace("\n", "") == text.strip("\n").replace("\n", "")


def test_deliver_isolates_failing_sink(tmp_path):
    class Broken(Sink):
        name = "broken"

        def deliver(self, brief):
            raise RuntimeError("disk full")

    res = deliver(_brief(), [Broken(), FileSink(tmp_path)])
    assert res["broken"].startswith("error: RuntimeError: disk full")
    assert res["file"].startswith("ok: ")


def test_make_sinks():
    sinks = make_sinks([{"type": "console"}, {"type": "file", "dir": "x"}],
                       base_dir="/tmp", env={})
    assert [s.name for s in sinks] == ["console", "file"]
    with pytest.raises(ValueError, match="unknown sink"):
        make_sinks([{"type": "pigeon"}])
