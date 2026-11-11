import shlex
import sys
from pathlib import Path

import pytest

from ai_news_pipeline.llm import LLMError, QuotaExhausted, SchemaError
from ai_news_pipeline.llm.command_backend import CommandBackend, diagnose_returncode

SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}


def _script(tmp_path: Path, body: str) -> str:
    p = tmp_path / "fake_model.py"
    p.write_text("import sys\nprompt = sys.stdin.read()\n" + body, encoding="utf-8")
    return f"{shlex.quote(sys.executable)} {shlex.quote(str(p))}"


def _call(cmd: str, timeout: float = 20):
    return CommandBackend(cmd, timeout_s=timeout).complete_json(
        purpose="score", system="SYS", user="USER", schema=SCHEMA)


def test_reads_json_from_stdout_and_sends_prompt_on_stdin(tmp_path):
    cmd = _script(tmp_path, "assert 'SYS' in prompt and 'USER' in prompt and '\"ok\"' in prompt\n"
                            "print('here you go: {\"ok\": true}')")
    assert _call(cmd) == {"ok": True}


def test_schema_mismatch_is_an_item_error(tmp_path):
    with pytest.raises(SchemaError):
        _call(_script(tmp_path, "print('{\"ok\": \"yes\"}')"))


def test_quota_message_raises_quota_exhausted_with_reset(tmp_path):
    cmd = _script(tmp_path, "print('Error: usage limit reached, resets in 2 hours', file=sys.stderr)\nsys.exit(1)")
    with pytest.raises(QuotaExhausted) as info:
        _call(cmd)
    assert info.value.retry_after_s == 7200


def test_nonzero_exit_is_diagnosed(tmp_path):
    cmd = _script(tmp_path, "print('boom', file=sys.stderr)\nsys.exit(3)")
    with pytest.raises(LLMError, match="rc=3.*boom"):
        _call(cmd)


def test_timeout(tmp_path):
    cmd = _script(tmp_path, "import time\ntime.sleep(5)")
    with pytest.raises(LLMError, match="timed out"):
        _call(cmd, timeout=0.5)


def test_missing_binary_and_empty_command():
    with pytest.raises(LLMError, match="cannot start"):
        _call("/nonexistent/model-cli")
    with pytest.raises(LLMError):
        CommandBackend("  ")


def test_diagnose_returncode():
    assert "SIGTERM" in diagnose_returncode(143)
    assert "memory" in diagnose_returncode(137)
    assert "SIGKILL" in diagnose_returncode(-9)


def test_quota_words_in_a_successful_answer_do_not_stop_the_run(tmp_path):
    # An article about rate limits must not trip the quota guard.
    cmd = _script(tmp_path, "print('note: usage limit reached in the article')\nprint('{\"ok\": true}')")
    assert _call(cmd) == {"ok": True}


def test_command_gets_an_allowlisted_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "dummy-value")
    monkeypatch.setenv("SOME_PROVIDER_API_KEY", "dummy-value")
    monkeypatch.setenv("MY_CLI_TOKEN", "needed")
    cmd = _script(tmp_path, "import os, json\n"
                            "assert 'TELEGRAM_BOT_TOKEN' not in os.environ\n"
                            "assert 'SOME_PROVIDER_API_KEY' not in os.environ\n"
                            "assert os.environ.get('MY_CLI_TOKEN') == 'needed'\n"
                            "assert 'PATH' in os.environ\n"
                            "print(json.dumps({'ok': True}))")
    backend = CommandBackend(cmd, timeout_s=20, env=("MY_CLI_TOKEN",))
    assert backend.complete_json(purpose="score", system="S", user="U", schema=SCHEMA) == {"ok": True}


def test_quota_words_on_stdout_of_a_failed_call_are_a_plain_error(tmp_path):
    cmd = _script(tmp_path, "print('the article says: usage limit reached, resets in 900 hours')\nsys.exit(1)")
    with pytest.raises(LLMError) as info:
        _call(cmd)
    assert not isinstance(info.value, QuotaExhausted)


def test_exit_code_75_means_quota(tmp_path):
    with pytest.raises(QuotaExhausted):
        _call(_script(tmp_path, "sys.exit(75)"))
