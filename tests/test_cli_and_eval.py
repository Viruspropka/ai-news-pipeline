import json

from ai_news_pipeline.cli import main
from ai_news_pipeline.evals import load_cases, run_eval
from ai_news_pipeline.llm.mock import MockBackend
from ai_news_pipeline.stages.common import DEFAULT_CATEGORIES

from conftest import ScriptedBackend


def test_demo_command(tmp_path, capsys):
    assert main(["demo", "--out", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "demo OK" in out and "stopped: model plan exhausted" in out
    assert "AI-generated" in out


def test_run_command_and_stop_exit_code(tmp_path, capsys, monkeypatch, data_dir):
    cfg = tmp_path / "c.yaml"
    cfg.write_text(f"sources_file: {data_dir / 'demo_sources.yaml'}\nsinks: [{{type: file, dir: out}}]\n"
                   "brief: {window_hours: 720}\n", encoding="utf-8")
    assert main(["run", "--config", str(cfg), "--stages", "ingest,score"]) == 0
    assert "score  : scored=10" in capsys.readouterr().out
    (tmp_path / "data/state").mkdir(parents=True)
    (tmp_path / "data/state/quota_stop.json").write_text(json.dumps(
        {"until": "2999-01-01T00:00:00+00:00", "backend": "x", "reason": "test"}))
    assert main(["run", "--config", str(cfg)]) == 75
    assert main(["run", "--config", str(cfg), "--stages", "nope"]) == 2


def test_sources_command_and_config_error(tmp_path, capsys):
    assert main(["sources"]) in (0, 2)  # depends on cwd; must not crash
    bad = tmp_path / "bad.yaml"
    bad.write_text("llm: {backend: nope}")
    assert main(["sources", "--config", str(bad)]) == 2


def test_eval_cases_are_well_formed():
    cases = load_cases()
    assert len(cases) >= 20
    assert {c["verdict"] for c in cases} == {"pass", "hype", "dubious", "thin"}
    assert len({c["id"] for c in cases}) == len(cases)


def test_eval_metrics_with_known_mistakes():
    cases = [
        {"id": "a", "title": "t", "summary": "", "verdict": "pass", "relevance": "high"},
        {"id": "b", "title": "t", "summary": "", "verdict": "hype", "relevance": "high"},
        {"id": "c", "title": "t", "summary": "", "verdict": "thin", "relevance": "low"},
    ]
    s = lambda r: {"relevance": r, "novelty": 5, "actionability": 5, "category": "other", "one_line": "a line"}
    v = lambda x: {"verdict": x, "flags": [], "confidence": 0.5}
    backend = ScriptedBackend({"score": [s(8), s(8), s(8)], "verify": [v("pass"), v("pass"), v("thin")]})
    rep = run_eval(backend, cases, DEFAULT_CATEGORIES)
    assert round(rep.verdict_accuracy(), 3) == 0.667
    assert rep.false_pass_rate() == 0.5      # the hype item slipped through
    assert round(rep.relevance_accuracy(), 3) == 0.667
    assert rep.per_verdict()["pass"]["precision"] == 0.5
    assert "miss: b: expected hype, got pass" in rep.render()


def test_mock_baseline_on_bundled_cases(tmp_path, capsys):
    rep = run_eval(MockBackend(), load_cases(), DEFAULT_CATEGORIES)
    assert rep.false_pass_rate() == 0.0
    assert rep.verdict_accuracy() >= 0.9
    out = tmp_path / "eval.json"
    assert main(["eval", "--backend", "mock", "--json-out", str(out), "--max-false-pass", "0"]) == 0
    assert json.loads(out.read_text())["cases"] == len(load_cases())
