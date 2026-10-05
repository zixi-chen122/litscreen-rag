"""
Summarises every saved evaluation run for a dataset: one row per
configuration (mode, retrieved examples, criteria), with the mean and range
of each metric across its repeat runs.

    python -m src.summarize_runs --dataset Donners_2021

Runs are grouped by the run_* columns evaluate.py writes into each results
file. Files without those columns (made before they existed) are listed as
skipped rather than guessed at.
"""
import argparse
from pathlib import Path

import pandas as pd

from src.datasets import load_criteria, load_exclude_weight, load_final_labels
from src.evaluate import compute_metrics, criteria_fingerprint

CONFIG_COLUMNS = ["run_label", "run_mode", "run_k_examples", "run_criteria"]


def collect_runs(dataset: str, results_dir: Path = Path("data")) -> tuple[pd.DataFrame, list[str]]:
    rows, skipped = [], []
    for path in sorted(results_dir.glob("evaluation_results*.csv")):
        df = pd.read_csv(path, keep_default_na=False)
        if "run_dataset" not in df.columns:
            skipped.append(path.name)
            continue
        if df["run_dataset"].iloc[0] != dataset:
            continue
        if "run_label" not in df.columns:  # runs saved before labels were recorded
            df["run_label"] = "final"
        if "label_final" not in df.columns and df["run_label"].iloc[0] != "final":
            df["label_final"] = df["paper_id"].map(load_final_labels(dataset))
        # Older runs don't record the weight; fall back to the dataset's split.json.
        weight = float(df["run_exclude_weight"].iloc[0]) if "run_exclude_weight" in df.columns else load_exclude_weight(dataset)
        config = df.iloc[0][CONFIG_COLUMNS].to_dict()
        extra = {
            "errors": int((df["error"] != "").sum()) if "error" in df.columns else 0,
            "searches": float(pd.to_numeric(df["n_searches"], errors="coerce").mean()) if "n_searches" in df.columns else float("nan"),
        }
        rows.append({"file": path.name, **config, **compute_metrics(df, weight), "exclude_weight": weight, **extra})
    return pd.DataFrame(rows), skipped


def _fmt(values: pd.Series, digits: int) -> str:
    mean = f"{values.mean():.{digits}f}"
    if len(values) == 1 or values.min() == values.max():
        return mean
    return f"{mean} ({values.min():.{digits}f}–{values.max():.{digits}f})"


def summarize(runs: pd.DataFrame, current_criteria: str) -> pd.DataFrame:
    summary = []
    for config, group in runs.groupby(CONFIG_COLUMNS):
        label, mode, k, criteria = config
        summary.append({
            "setup": "agent" if mode == "agent" else ("baseline (k=0)" if k == 0 else f"RAG (k={k})"),
            "labels": label,
            "criteria": criteria + (" (current)" if criteria == current_criteria else ""),
            "runs": len(group),
            "precision": _fmt(group["precision"], 2),
            "recall": _fmt(group["recall"], 2),
            "f1": _fmt(group["f1"], 2),
            "unsure": _fmt(group["unsure"], 1),
            "precision (adj)": _fmt(group["precision_adj"], 2),
            "unsure % (adj)": _fmt(group["unsure_rate_adj"] * 100, 1),
            "recall (unsure kept)": _fmt(group["recall_unsure_kept"], 2),
            "final lost": f"{_fmt(group['final_lost'], 1)} of {group['final_includes'].iloc[0]}",
            "searches/paper": _fmt(group["searches"], 2) if "searches" in group and group["searches"].notna().all() else "",
        })
    return pd.DataFrame(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--per-run", action="store_true", help="also list each run's file and metrics")
    args = parser.parse_args()

    runs, skipped = collect_runs(args.dataset)
    if runs.empty:
        raise SystemExit(f"No tagged runs found for {args.dataset}.")
    current = criteria_fingerprint(load_criteria(args.dataset))
    print(summarize(runs, current).to_string(index=False))
    print("\nValues are mean (min–max) across runs; unsure is a count of test papers.")
    print("recall (unsure kept) = share of gold includes not excluded, since unsure papers go to a person.")
    print("final lost = papers in the final review (after full text) that the model excluded.")
    weights = runs["exclude_weight"].unique()
    if (weights != 1.0).any():
        print(f"(adj) = corrected to the dataset's real inclusion rate; each test exclude "
              f"stands for {', '.join(f'{w:g}' for w in weights)} real excludes.")
    else:
        print("(adj) columns equal the raw ones here: test excludes were not downsampled.")
    if args.per_run:
        print()
        print(runs.round(2).to_string(index=False))
    failed = runs[runs["errors"] > 0] if "errors" in runs else runs.iloc[0:0]
    for _, r in failed.iterrows():
        print(f"WARNING: {r['file']} has {r['errors']} papers that failed and count as unsure.")
    if skipped:
        print(f"\nSkipped (no run metadata): {', '.join(skipped)}")
