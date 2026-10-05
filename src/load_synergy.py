"""
Turn one SYNERGY dataset into the three inputs LitScreen RAG needs:

  1. seed set  -> previously screened examples to ingest into Chroma
  2. test set  -> papers to screen and score (must NEVER be in Chroma)
  3. criteria  -> review_topic / inclusion_criteria / exclusion_criteria

Usage (after `synergy get -d Donners_2021 -o data/synergy`):

    python -m src.load_synergy --dataset Donners_2021

Outputs (in data/synergy/prepared/<dataset>/ by default):
    seed.csv, test.csv, criteria.json, split.json (the settings used)

--label picks the gold standard: "final" (label_included, the decision after
full-text review; the default) or "abstract" (label_abstract_included, the
decision after title/abstract screening -- only some SYNERGY+ datasets have it).

Every output CSV has the same columns:
    paper_id, doi, openalex_id, title, abstract, label, label_final
    (label = the chosen gold standard; label_final = the decision after full text)
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

LABEL_MAP = {1: "include", 0: "exclude"}
LABEL_COLUMNS = {"final": "label_included", "abstract": "label_abstract_included"}


# ---------------------------------------------------------------------------
# 1. Records
# ---------------------------------------------------------------------------

def normalize_doi(doi) -> str | None:
    """'https://doi.org/10.1/ABC' -> '10.1/abc'. Returns None if missing."""
    if not isinstance(doi, str) or not doi.strip():
        return None
    doi = doi.strip().lower()
    doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", doi)
    return doi or None


def load_records(csv_path: Path, label: str = "final") -> pd.DataFrame:
    """Load a SYNERGY dataset CSV and clean it into a standard shape."""
    df = pd.read_csv(csv_path)
    label_column = LABEL_COLUMNS[label]

    required = {"openalex_id", "doi", "title", "abstract", label_column}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path.name} is missing columns: {sorted(missing)}")

    n_raw = len(df)

    # Drop records we can't screen: no title or no abstract.
    df = df.dropna(subset=["title", "abstract"])
    df = df[df["abstract"].str.strip().str.len() > 0].copy()

    # Stable ID: DOI when available, otherwise OpenAlex ID (always present).
    # This matches the idea behind your DOI-based ingestion IDs, but never
    # leaves a record without an ID.
    df["doi"] = df["doi"].map(normalize_doi)
    df["paper_id"] = df["doi"].fillna(df["openalex_id"])

    # Two records with the same ID would collide in Chroma; keep the first.
    df = df.drop_duplicates(subset="paper_id", keep="first")

    if df[label_column].isna().any():
        raise ValueError(f"{label_column} is missing for {df[label_column].isna().sum()} records; "
                         f"this dataset may not have {label} labels")
    df["label"] = df[label_column].astype(int).map(LABEL_MAP)
    # Always carry the final (after full text) decision too, so results scored
    # against abstract-stage labels can also be checked against the final review.
    df["label_final"] = df["label_included"].astype(int).map(LABEL_MAP)
    if df["label"].isna().any():
        raise ValueError(f"{label_column} contains values other than 0/1")

    print(f"Loaded {csv_path.name}: {n_raw} rows -> {len(df)} usable "
          f"({(df['label'] == 'include').sum()} include, "
          f"{(df['label'] == 'exclude').sum()} exclude)")

    return df[["paper_id", "doi", "openalex_id", "title", "abstract", "label", "label_final"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Seed / test split
# ---------------------------------------------------------------------------

def split_seed_test(
    df: pd.DataFrame,
    n_seed_include: int,
    n_seed_exclude: int,
    n_test_exclude: int | None,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Seed = a fixed number of includes and excludes (stratified on purpose,
    so retrieval can actually show the model some includes).
    Test = everything else, optionally downsampled to all remaining includes
    plus n_test_exclude random excludes to keep API costs down.
    """
    includes = df[df["label"] == "include"]
    excludes = df[df["label"] == "exclude"]

    if n_seed_include >= len(includes):
        raise ValueError(
            f"Only {len(includes)} includes available; n_seed_include={n_seed_include} "
            "would leave none to test. Lower --n-seed-include."
        )

    seed = pd.concat([
        includes.sample(n=n_seed_include, random_state=random_state),
        excludes.sample(n=min(n_seed_exclude, len(excludes)), random_state=random_state),
    ])

    test = df[~df["paper_id"].isin(seed["paper_id"])]

    if n_test_exclude is not None:
        test_inc = test[test["label"] == "include"]
        test_exc = test[test["label"] == "exclude"]
        test = pd.concat([
            test_inc,
            test_exc.sample(n=min(n_test_exclude, len(test_exc)), random_state=random_state),
        ])

    # Shuffle so includes aren't all at the top (matters if a run stops midway).
    seed = seed.sample(frac=1, random_state=random_state).reset_index(drop=True)
    test = test.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # The one rule that must never break: no test paper in the seed set.
    overlap = set(seed["paper_id"]) & set(test["paper_id"])
    assert not overlap, f"Leakage: {len(overlap)} papers in both seed and test"

    return seed, test


# ---------------------------------------------------------------------------
# 3. Criteria
# ---------------------------------------------------------------------------

# Matches headings like "Exclusion criteria:", "Exclusion:", "Excluded:"
_EXCLUSION_HEADING = re.compile(r"\bexclu(sion|ded|de)\b[^:\n]*[:\n]", re.IGNORECASE)
_INCLUSION_HEADING = re.compile(r"^\s*inclu(sion|ded|de)\b[^:\n]*[:\n]\s*", re.IGNORECASE)


def split_criteria(text: str) -> tuple[str, str]:
    """
    Best-effort split of one eligibility text into (inclusion, exclusion).
    If no exclusion heading is found, everything goes into inclusion and
    exclusion is left empty -- check criteria.json and edit by hand if needed.
    """
    match = _EXCLUSION_HEADING.search(text)
    if not match:
        return text.strip(), ""
    inclusion = _INCLUSION_HEADING.sub("", text[: match.start()]).strip()
    exclusion = text[match.end():].strip()
    return inclusion, exclusion


def load_criteria(metadata_path: Path, dataset: str) -> dict:
    meta = pd.read_csv(metadata_path)
    if "key" not in meta.columns:
        raise ValueError(f"{metadata_path.name} has no 'key' column; columns are {list(meta.columns)}")

    rows = meta[meta["key"] == dataset]
    if rows.empty:
        raise ValueError(f"{dataset} not found in {metadata_path.name}. "
                         f"Available: {sorted(meta['key'].tolist())}")
    row = rows.iloc[0]

    eligibility = row.get("eligibility_criteria")
    if not isinstance(eligibility, str) or not eligibility.strip():
        raise ValueError(f"No eligibility_criteria text for {dataset}; "
                         "you'll need to write the criteria by hand.")

    inclusion, exclusion = split_criteria(eligibility)
    if not exclusion:
        print("WARNING: couldn't find an exclusion section; all criteria text "
              "is in inclusion_criteria. Review criteria.json by hand.")

    # The review paper's own title is the best available topic statement.
    title = row.get("title")
    review_topic = title if isinstance(title, str) and title.strip() else dataset

    return {
        "dataset": dataset,
        "review_topic": review_topic,
        "inclusion_criteria": inclusion,
        "exclusion_criteria": exclusion,
        "eligibility_criteria_raw": eligibility,  # kept so you can compare with the split
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Prepare a SYNERGY dataset for LitScreen RAG.")
    p.add_argument("--dataset", required=True, help="e.g. Donners_2021")
    p.add_argument("--data-dir", default="data/synergy", type=Path)
    p.add_argument("--out-dir", default=None, type=Path,
                   help="default: <data-dir>/prepared/<dataset>")
    p.add_argument("--n-seed-include", type=int, default=5)
    p.add_argument("--n-seed-exclude", type=int, default=30)
    p.add_argument("--n-test-exclude", type=int, default=None,
                   help="downsample test excludes to this many (default: keep all)")
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--label", choices=sorted(LABEL_COLUMNS), default="final",
                   help="gold standard: 'final' (after full text) or 'abstract' (after title/abstract screening)")
    args = p.parse_args()

    out_dir = args.out_dir or args.data_dir / "prepared" / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(args.data_dir / f"{args.dataset}.csv", args.label)
    seed, test = split_seed_test(
        records, args.n_seed_include, args.n_seed_exclude,
        args.n_test_exclude, args.random_state,
    )
    criteria = load_criteria(args.data_dir / "metadata" / "review_metadata.csv", args.dataset)

    seed.to_csv(out_dir / "seed.csv", index=False)
    test.to_csv(out_dir / "test.csv", index=False)
    (out_dir / "criteria.json").write_text(json.dumps(criteria, indent=2, ensure_ascii=False))
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    # Each test exclude stands in for this many real ones when excludes were
    # downsampled; summarize_runs.py uses it to correct precision and unsure rate.
    n_exclude_available = (records["label"] == "exclude").sum() - (seed["label"] == "exclude").sum()
    settings["test_exclude_weight"] = round(float(n_exclude_available / (test["label"] == "exclude").sum()), 4)
    (out_dir / "split.json").write_text(json.dumps(settings, indent=2))

    n_inc = (test["label"] == "include").sum()
    print(f"\nSeed: {len(seed)} papers ({(seed['label'] == 'include').sum()} include)")
    print(f"Test: {len(test)} papers ({n_inc} include, {n_inc / len(test):.1%} prevalence)")
    if args.n_test_exclude is not None:
        print("NOTE: test excludes were downsampled, so precision will look better "
              "than at the dataset's real inclusion rate. Report this.")
    print(f"Wrote seed.csv, test.csv, criteria.json, split.json to {out_dir} ({args.label} labels)")


if __name__ == "__main__":
    main()
