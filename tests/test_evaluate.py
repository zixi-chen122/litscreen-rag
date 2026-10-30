"""run_evaluation with the screener stubbed out: no LLM, embedding or Chroma calls."""
import pandas as pd

import src.evaluate as ev
from src.screening_chain import ScreeningDecision


def test_failed_paper_is_recorded_as_unsure_and_run_continues(tmp_path, monkeypatch):
    papers = pd.DataFrame({
        "doi": ["10.1/a", "10.1/b", "10.1/c"],
        "title": ["A", "B", "C"],
        "abstract": ["x", "y", "z"],
        "gold_label": ["include", "exclude", "include"],
    })
    monkeypatch.setattr(ev, "load_eval_papers", lambda dataset: papers.copy())
    monkeypatch.setattr(ev, "_check_examples_store", lambda *a: None)
    monkeypatch.setattr(ev, "load_criteria", lambda dataset: {"review_topic": "t", "inclusion_criteria": ["i"], "exclusion_criteria": []})

    def fake_screen(title, abstract, k_examples=3, criteria=None, example_selection="similar"):
        if title == "B":
            raise RuntimeError("step limit reached")
        return ScreeningDecision(decision="include", rationale="ok", confidence=0.9)

    monkeypatch.setattr(ev, "screen_paper", fake_screen)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()

    ev.run_evaluation(run_tag="t")

    out = pd.read_csv(tmp_path / "data" / "evaluation_results_t.csv", keep_default_na=False)
    assert list(out["predicted_label"]) == ["include", "unsure", "include"]
    assert out.loc[1, "error"].startswith("RuntimeError: step limit reached")
    assert out.loc[0, "error"] == "" and out.loc[2, "error"] == ""
    # Latency and token usage are recorded for every paper, failed ones too.
    assert (out["latency_s"] >= 0).all()
    assert list(out["input_tokens"]) == [0, 0, 0] and list(out["output_tokens"]) == [0, 0, 0]


def test_example_selection_is_passed_through_and_recorded(tmp_path, monkeypatch):
    papers = pd.DataFrame({"doi": ["10.1/a"], "title": ["A"], "abstract": ["x"], "gold_label": ["include"]})
    monkeypatch.setattr(ev, "load_eval_papers", lambda dataset: papers.copy())
    monkeypatch.setattr(ev, "_check_examples_store", lambda *a: None)
    monkeypatch.setattr(ev, "all_examples", lambda: [None] * 60)
    monkeypatch.setattr(ev, "load_criteria", lambda dataset: {"review_topic": "t", "inclusion_criteria": ["i"], "exclusion_criteria": []})
    seen = {}

    def fake_screen(title, abstract, k_examples=3, criteria=None, example_selection="similar"):
        seen["selection"] = example_selection
        return ScreeningDecision(decision="include", rationale="ok", confidence=0.9)

    monkeypatch.setattr(ev, "screen_paper", fake_screen)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()

    ev.run_evaluation(run_tag="t", example_selection="all")

    out = pd.read_csv(tmp_path / "data" / "evaluation_results_all_t.csv")
    assert seen["selection"] == "all"
    assert out.loc[0, "run_example_selection"] == "all"
    assert out.loc[0, "run_k_examples"] == 60


def test_example_selection_rejected_with_agent_or_zero_k():
    import pytest

    with pytest.raises(SystemExit):
        ev.run_evaluation(use_agent=True, example_selection="random")
    with pytest.raises(SystemExit):
        ev.run_evaluation(k_examples=0, example_selection="all")
