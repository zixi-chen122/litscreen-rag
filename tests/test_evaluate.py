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

    def fake_screen(title, abstract, k_examples=3, criteria=None):
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
