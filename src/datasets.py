"""
Resolves a --dataset name to the three inputs the pipeline needs, in the
column shapes ingest.py and evaluate.py expect:

  - examples : doi, title, abstract, decision, rationale  (goes into Chroma)
  - eval     : doi, title, abstract, gold_label           (screened and scored)
  - criteria : {review_topic, inclusion_criteria: [..], exclusion_criteria: [..]}

"sample" is the bundled toy data in data/. Any other name is a SYNERGY
dataset prepared by `python -m src.load_synergy --dataset <name>`, read from
data/synergy/prepared/<name>/.
"""
import json
from pathlib import Path

import pandas as pd
import yaml

SAMPLE = "sample"
SYNERGY_PREPARED_DIR = Path("data/synergy/prepared")


def _synergy_dir(dataset: str) -> Path:
    path = SYNERGY_PREPARED_DIR / dataset
    if not path.is_dir():
        raise FileNotFoundError(
            f"No prepared SYNERGY data at {path}. "
            f"Run: python -m src.load_synergy --dataset {dataset}"
        )
    return path


def _from_synergy(csv_path: Path, label_column: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # paper_id is the DOI when there is one, else the OpenAlex ID, so it is
    # always present -- unlike doi, which Chroma metadata can't hold as NaN.
    return df.assign(doi=df["paper_id"]).rename(columns={"label": label_column})


def load_examples(dataset: str) -> pd.DataFrame:
    if dataset == SAMPLE:
        return pd.read_csv("data/labeled_examples.csv")
    df = _from_synergy(_synergy_dir(dataset) / "seed.csv", "decision")
    return df.assign(rationale="")  # SYNERGY records labels only, no rationales


def load_eval_papers(dataset: str) -> pd.DataFrame:
    if dataset == SAMPLE:
        return pd.read_csv("data/sample_papers.csv")
    return _from_synergy(_synergy_dir(dataset) / "test.csv", "gold_label")


def load_criteria(dataset: str) -> dict:
    if dataset == SAMPLE:
        with open("data/criteria.yaml") as f:
            return yaml.safe_load(f)
    raw = json.loads((_synergy_dir(dataset) / "criteria.json").read_text())
    # load_synergy writes each criteria section as one block of text.
    as_list = lambda text: [text.strip()] if text and text.strip() else []
    return {
        "review_topic": raw["review_topic"],
        "inclusion_criteria": as_list(raw["inclusion_criteria"]),
        "exclusion_criteria": as_list(raw["exclusion_criteria"]),
    }


def load_final_labels(dataset: str) -> pd.Series | None:
    """Final (after full text) labels of the eval papers, indexed by paper_id,
    for results files saved before they carried a label_final column."""
    if dataset == SAMPLE:
        return None
    test = pd.read_csv(_synergy_dir(dataset) / "test.csv")
    column = "label_final" if "label_final" in test.columns else "label"
    return test.set_index("paper_id")[column]


def load_label_source(dataset: str) -> str:
    """Which gold labels the eval set uses: 'final' (after full text) or
    'abstract' (after title/abstract screening). Datasets prepared before
    load_synergy wrote split.json used final labels."""
    if dataset == SAMPLE:
        return "final"
    split_path = _synergy_dir(dataset) / "split.json"
    if not split_path.exists():
        return "final"
    return json.loads(split_path.read_text()).get("label", "final")


def load_exclude_weight(dataset: str) -> float:
    """How many real excludes each test exclude stands for (1.0 unless
    load_synergy downsampled the test excludes with --n-test-exclude)."""
    if dataset == SAMPLE:
        return 1.0
    split_path = _synergy_dir(dataset) / "split.json"
    if not split_path.exists():
        return 1.0
    return float(json.loads(split_path.read_text()).get("test_exclude_weight", 1.0))
