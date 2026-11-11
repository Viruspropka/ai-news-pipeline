from pathlib import Path

import pytest

from ai_news_pipeline.config import ConfigError, load_settings


def test_defaults_without_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    s = load_settings(env={})
    assert s.llm_backend == "mock"
    assert s.brief.language == "English"
    assert "other" in s.brief.categories
    assert s.db_path == tmp_path / "data/news.sqlite"
    assert s.sinks == [{"type": "console"}]


def test_shipped_example_config_loads():
    root = Path(__file__).resolve().parents[1]
    s = load_settings(root / "config.example.yaml", env={})
    assert s.sources_file == root / "sources.yaml"
    assert s.llm_backend == "mock"
    assert {x["type"] for x in s.sinks} >= {"console", "file"}


def test_file_values_paths_and_env_overrides(tmp_path):
    cfg = tmp_path / "cfg" / "c.yaml"
    cfg.parent.mkdir()
    cfg.write_text(
        "db_path: db/x.sqlite\noutput_language: Spanish\n"
        "llm: {backend: anthropic, models: {score: claude-haiku-5-5}}\n"
        "brief: {max_items: 5, title: 'Daily {date}'}\n"
        "categories: {models: Modelos, other: Otros}\n"
        "sinks: [{type: file, dir: out}]\n", encoding="utf-8")
    s = load_settings(cfg, env={"ANP_OUTPUT_LANGUAGE": "German", "ANP_SINKS": "console,file"})
    assert s.db_path == cfg.parent / "db/x.sqlite"
    assert s.brief.language == "German"
    assert s.llm_backend == "anthropic" and s.llm_options["models"]["score"] == "claude-haiku-5-5"
    assert s.brief.max_items == 5 and s.brief.categories == {"models": "Modelos", "other": "Otros"}
    assert s.sinks == [{"type": "console"}, {"type": "file", "dir": "out"}]


@pytest.mark.parametrize("body,needle", [
    ("llm: {backend: gpt}", "backend"),
    ("brief: {max_items: 0}", "outside"),
    ("brief: {window_hours: soon}", "number"),
    ("categories: {models: M}", "other"),
    ("sinks: console", "list"),
    ("brief: [1, 2]", "mapping"),
    ("- just a list", "mapping"),
])
def test_invalid_config(tmp_path, body, needle):
    p = tmp_path / "c.yaml"
    p.write_text(body, encoding="utf-8")
    with pytest.raises(ConfigError, match=needle):
        load_settings(p, env={})


def test_missing_config_file(tmp_path):
    with pytest.raises(ConfigError, match="cannot read"):
        load_settings(tmp_path / "none.yaml", env={})


def test_command_env_names_from_file_and_env(tmp_path):
    from ai_news_pipeline.config import ConfigError, load_settings
    cfg = tmp_path / "c.yaml"
    cfg.write_text("llm: {backend: command, command: x, command_env: [MY_TOKEN]}\n")
    assert load_settings(cfg, env={}).llm_options["command_env"] == ["MY_TOKEN"]
    assert load_settings(cfg, env={"ANP_LLM_COMMAND_ENV": "A, B_2"}).llm_options["command_env"] == ["A", "B_2"]
    cfg.write_text("llm: {command_env: ['bad name;rm']}\n")
    with pytest.raises(ConfigError):
        load_settings(cfg, env={})
