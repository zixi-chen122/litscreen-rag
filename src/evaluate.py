"""
Runs the screening agent across the full sample corpus and reports
precision / recall / F1 against the gold_label column, plus a per-paper
breakdown. This is the difference between "a demo that looks plausible"
and a screening tool with a measured accuracy you can actually cite.

--dataset picks the papers and criteria (see src/datasets.py). Run
`python -m src.ingest --dataset <same name>` first so the retrieved examples
come from the same review.
"""
import argparse
import hashlib
import json

import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

from src.datasets import SAMPLE, load_criteria, load_eval_papers, load_exclude_weight, load_label_source
from src.retriever import get_examples_store
from src.screening_chain import screen_paper
from src.screening_agent_tools import build_screening_agent, screen_paper_agent


def _check_examples_store(dataset: str, df: pd.DataFrame) -> None:
    """Refuse to score against examples from another dataset, or examples
    that include the very papers being scored."""
    store = get_examples_store()
    ingested = (store._collection.metadata or {}).get("dataset")
    if ingested != dataset:
        raise SystemExit(
            f"Examples collection holds dataset {ingested!r}, not {dataset!r}. "
            f"Run: python -m src.ingest --dataset {dataset}"
        )
    example_dois = {m["doi"] for m in store.get(include=["metadatas"])["metadatas"]}
    leaked = example_dois & set(df["doi"])
    if leaked:
        raise SystemExit(f"Leakage: {len(leaked)} evaluation papers are also retrieval examples: {sorted(leaked)[:5]}")


def criteria_fingerprint(criteria: dict) -> str:
    """Short stable hash of the criteria, so runs with edited criteria are
    never pooled with runs on the original ones."""
    return hashlib.sha1(json.dumps(criteria, sort_keys=True).encode()).hexdigest()[:8]


def compute_metrics(df: pd.DataFrame, exclude_weight: float = 1.0) -> dict:
    """Metrics on the test set as screened, plus precision and unsure rate
    corrected to the dataset's real inclusion rate: when test excludes were
    downsampled, each one stands in for `exclude_weight` real excludes.
    Recall only involves includes, so it needs no correction."""
    # Treat 'unsure' as a miss against either gold class for a conservative metric.
    y_true = [1 if lbl == "include" else 0 for lbl in df["gold_label"]]
    y_pred = [1 if lbl == "include" else 0 for lbl in df["predicted_label"]]

    gold_inc = df["gold_label"] == "include"
    true_pos = (gold_inc & (df["predicted_label"] == "include")).sum()
    false_pos = (~gold_inc & (df["predicted_label"] == "include")).sum() * exclude_weight
    unsure = df["predicted_label"] == "unsure"
    weighted_unsure = (unsure & gold_inc).sum() + (unsure & ~gold_inc).sum() * exclude_weight
    weighted_total = gold_inc.sum() + (~gold_inc).sum() * exclude_weight
    # In practice 'unsure' papers go to a human, so they are not lost.
    kept = df["predicted_label"] != "exclude"
    # The final review's includes (after full text). Equal to the gold labels
    # unless the gold standard is the abstract-stage decision.
    final_inc = (df["label_final"] if "label_final" in df.columns else df["gold_label"]) == "include"
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "unsure": int(unsure.sum()),
        "precision_adj": true_pos / (true_pos + false_pos) if true_pos + false_pos else 0.0,
        "unsure_rate_adj": weighted_unsure / weighted_total,
        "recall_unsure_kept": (gold_inc & kept).sum() / gold_inc.sum() if gold_inc.sum() else 0.0,
        "final_includes": int(final_inc.sum()),
        "final_lost": int((final_inc & ~kept).sum()),
    }


def run_evaluation(use_agent: bool = False, dataset: str = SAMPLE, k_examples: int = 3, run_tag: str | None = None):
    if use_agent and k_examples != 3:
        raise SystemExit("--k-examples applies to the chain only; the agent decides its own searches.")
    df = load_eval_papers(dataset)
    criteria = load_criteria(dataset)
    if k_examples > 0:  # the criteria-only baseline never touches Chroma
        _check_examples_store(dataset, df)
    results = []

    if use_agent:
        agent = build_screening_agent(criteria=criteria)
        def screen(row, stats):
            return screen_paper_agent(row["title"], row["abstract"], doi=row["doi"], agent=agent, stats=stats)
    else:
        def screen(row, stats):
            return screen_paper(row["title"], row["abstract"], k_examples=k_examples, criteria=criteria)

    for _, row in df.iterrows():
        stats = {}
        try:
            result = screen(row, stats).model_dump()
            error = ""
        except Exception as e:  # e.g. the agent hit its step limit, or an API error
            # Record the paper as unsure (it would go to a person) and keep the
            # run going; the error is kept so these papers can be found later.
            error = f"{type(e).__name__}: {e}"[:300]
            result = {"decision": "unsure", "rationale": f"ERROR {error}", "confidence": 0.0}
        results.append({**result, **stats, "error": error})
        print(
            f"[{result['decision'].upper():7s}] (conf {result['confidence']:.2f}) {row['title'][:70]}\n"
            f"          gold={row['gold_label']}  rationale={result['rationale']}\n"
        )

    # Keep the rationale and confidence (and the agent's review flag) alongside
    # each prediction, so errors and 'unsure' calls can be analysed afterwards.
    df = pd.concat([df, pd.DataFrame(results).rename(columns={"decision": "predicted_label"})], axis=1)

    # Record the run's configuration in the file itself; summarize_runs.py
    # groups runs by these columns rather than by file name.
    df = df.assign(
        run_dataset=dataset,
        run_mode="agent" if use_agent else "chain",
        run_k_examples=k_examples,
        run_criteria=criteria_fingerprint(criteria),
        run_label=load_label_source(dataset),
        run_exclude_weight=load_exclude_weight(dataset),
        run_tag=run_tag or "",
    )

    weight = load_exclude_weight(dataset)
    m = compute_metrics(df, weight)
    n_errors = int((df["error"] != "").sum())
    print("=" * 60)
    if n_errors:
        print(f"WARNING: {n_errors} papers failed and were recorded as unsure; see the 'error' column.")
    if "n_searches" in df.columns:
        print(f"Agent searches per paper: mean {df['n_searches'].mean():.2f}, max {df['n_searches'].max():.0f}; "
              f"flagged for review: {int(df['n_flags'].sum())}")
    print(f"Precision: {m['precision']:.2f}  Recall: {m['recall']:.2f}  F1: {m['f1']:.2f}")
    print(f"Unsure count: {m['unsure']} / {len(df)}")
    if weight != 1.0:
        print(f"At the dataset's real inclusion rate (each test exclude = {weight:g} real ones): "
              f"precision {m['precision_adj']:.2f}, unsure {m['unsure_rate_adj']:.1%}")

    suffix = (
        ("" if dataset == SAMPLE else f"_{dataset}")
        + ("_agent" if use_agent else "")
        + ("" if k_examples == 3 else f"_k{k_examples}")
        + (f"_{run_tag}" if run_tag else "")
    )
    out_path = f"data/evaluation_results{suffix}.csv"
    df.to_csv(out_path, index=False)
    print(f"Full results written to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", action="store_true", help="Use the tool-calling agent instead of the fixed RAG chain")
    parser.add_argument("--dataset", default=SAMPLE, help="'sample' or a prepared SYNERGY dataset name, e.g. Donners_2021")
    parser.add_argument("--k-examples", type=int, default=3, help="retrieved examples per paper (chain only); 0 = criteria-only baseline, no retrieval")
    parser.add_argument("--run-tag", default=None, help="appended to the output file name (e.g. run1) so repeat runs don't overwrite each other")
    args = parser.parse_args()
    run_evaluation(use_agent=args.agent, dataset=args.dataset, k_examples=args.k_examples, run_tag=args.run_tag)
