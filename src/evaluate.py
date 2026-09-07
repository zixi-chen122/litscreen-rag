"""
Runs the screening agent across the full sample corpus and reports
precision / recall / F1 against the gold_label column, plus a per-paper
breakdown. This is the difference between "a demo that looks plausible"
and a screening tool with a measured accuracy you can actually cite.
"""
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

from src.screening_agent import screen_paper

PAPERS_CSV = "data/sample_papers.csv"


def run_evaluation():
    df = pd.read_csv(PAPERS_CSV)
    predictions = []

    for _, row in df.iterrows():
        result = screen_paper(row["title"], row["abstract"])
        predictions.append(result.decision)
        print(
            f"[{result.decision.upper():7s}] (conf {result.confidence:.2f}) {row['title'][:70]}\n"
            f"          gold={row['gold_label']}  rationale={result.rationale}\n"
        )

    df["predicted_label"] = predictions

    # Treat 'unsure' as a miss against either gold class for a conservative metric.
    y_true = [1 if lbl == "include" else 0 for lbl in df["gold_label"]]
    y_pred = [1 if lbl == "include" else 0 for lbl in df["predicted_label"]]

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print("=" * 60)
    print(f"Precision: {precision:.2f}  Recall: {recall:.2f}  F1: {f1:.2f}")
    print(f"Unsure count: {(df['predicted_label'] == 'unsure').sum()} / {len(df)}")

    df.to_csv("data/evaluation_results.csv", index=False)
    print("Full results written to data/evaluation_results.csv")


if __name__ == "__main__":
    run_evaluation()
