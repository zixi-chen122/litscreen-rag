import pandas as pd

from src.evaluate import compute_metrics, criteria_fingerprint
from src.summarize_runs import CONFIG_COLUMNS, summarize


def _run(predicted, k=3, criteria="aaaa1111"):
    return pd.DataFrame({
        "gold_label": ["include", "include", "exclude", "exclude"],
        "predicted_label": predicted,
        "run_label": "final",
        "run_mode": "chain",
        "run_k_examples": k,
        "run_criteria": criteria,
    })


def test_compute_metrics_counts_unsure_as_not_include():
    m = compute_metrics(_run(["include", "unsure", "exclude", "include"]))
    assert m["precision"] == 0.5
    assert m["recall"] == 0.5
    assert m["unsure"] == 1


def test_criteria_fingerprint_changes_when_criteria_change():
    base = {"review_topic": "t", "inclusion_criteria": ["a"], "exclusion_criteria": []}
    assert criteria_fingerprint(base) == criteria_fingerprint(dict(base))
    assert criteria_fingerprint(base) != criteria_fingerprint(dict(base, inclusion_criteria=["b"]))


def test_summarize_pools_repeat_runs_but_not_different_setups():
    rows = []
    for df in [
        _run(["include", "include", "exclude", "exclude"]),           # RAG run 1
        _run(["include", "include", "include", "exclude"]),           # RAG run 2
        _run(["include", "exclude", "exclude", "exclude"], k=0),      # baseline
        _run(["include", "include", "exclude", "exclude"], criteria="bbbb2222"),
    ]:
        rows.append({**df.iloc[0][CONFIG_COLUMNS].to_dict(), **compute_metrics(df)})

    summary = summarize(pd.DataFrame(rows), current_criteria="aaaa1111")

    assert len(summary) == 3
    rag = summary[(summary["setup"] == "RAG (k=3)") & (summary["criteria"] == "aaaa1111 (current)")].iloc[0]
    assert rag["runs"] == 2
    assert rag["precision"] == "0.83 (0.67–1.00)"


def test_compute_metrics_corrects_for_downsampled_excludes():
    # 2 includes (1 found), 2 excludes (1 wrongly included, 1 unsure), each exclude standing for 4.
    m = compute_metrics(_run(["include", "exclude", "include", "unsure"]), exclude_weight=4.0)
    assert m["precision"] == 0.5
    assert m["precision_adj"] == 1 / (1 + 4)        # the false include counts 4 times
    assert m["unsure_rate_adj"] == 4 / (2 + 2 * 4)  # 4 weighted unsure out of 10 weighted papers
    assert compute_metrics(_run(["include", "exclude", "include", "unsure"]))["precision_adj"] == 0.5


def test_compute_metrics_unsure_kept_and_final_lost():
    df = _run(["unsure", "exclude", "include", "exclude"]).assign(
        label_final=["include", "include", "exclude", "exclude"]
    )
    m = compute_metrics(df)
    assert m["recall"] == 0.0                 # strict: unsure is not an include
    assert m["recall_unsure_kept"] == 0.5     # the unsure include goes to a person
    assert m["final_includes"] == 2
    assert m["final_lost"] == 1               # the second final include was excluded
