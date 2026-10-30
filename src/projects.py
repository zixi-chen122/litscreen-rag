"""
Review projects: a researcher's own screening work, kept in SQLite.

A project holds one review's criteria and its papers. Papers come in from a
CSV or RIS export (Zotero, EndNote, Rayyan, a database search) and are
deduplicated by DOI, or by normalised title when there is no DOI.

Every paper with a label -- given in the import file, or set by a reviewer --
is a screened example: it is embedded into the project's own Chroma
collection, so retrieval for one project never sees another project's papers.
Papers without a label are what gets screened. Each AI decision is stored
with the criteria fingerprint and mode it was made under, so a batch run
resumes where it stopped, and editing the criteria starts a fresh pass rather
than mixing decisions made under different criteria.
"""
import io
import json
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

import pandas as pd
from langchain_core.documents import Document

from src import config
from src.config import get_embeddings
from src.retriever import get_examples_store
from src.screening_chain import criteria_fingerprint

LABELS = ("include", "exclude")

SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    criteria TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS papers (
    id INTEGER PRIMARY KEY,
    project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    dedupe_key TEXT NOT NULL,
    title TEXT NOT NULL,
    abstract TEXT NOT NULL DEFAULT '',
    doi TEXT NOT NULL DEFAULT '',
    year TEXT NOT NULL DEFAULT '',
    label TEXT CHECK (label IN ('include', 'exclude')),
    label_source TEXT CHECK (label_source IN ('import', 'reviewer')),
    label_note TEXT NOT NULL DEFAULT '',
    -- labeled_at changes on every label change (including removal);
    -- indexed_at records which change the examples collection reflects.
    labeled_at TEXT,
    indexed_at TEXT,
    UNIQUE (project_id, dedupe_key)
);
CREATE TABLE IF NOT EXISTS screenings (
    id INTEGER PRIMARY KEY,
    paper_id INTEGER NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    criteria_fp TEXT NOT NULL,
    mode TEXT NOT NULL,
    model TEXT NOT NULL DEFAULT '',
    decision TEXT NOT NULL,
    rationale TEXT NOT NULL DEFAULT '',
    confidence REAL NOT NULL DEFAULT 0,
    flagged INTEGER NOT NULL DEFAULT 0,
    error TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS screenings_by_paper ON screenings (paper_id, criteria_fp, mode);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def examples_collection(project_id: int) -> str:
    """Name of the project's Chroma collection of screened examples."""
    return f"project_{project_id}_examples"


class ReviewStore:
    """All project, paper and decision storage. Opens a short-lived connection
    per call, so one store can be shared by Streamlit sessions and threads."""

    def __init__(self, path: str | None = None):
        self.path = path or config.DB_PATH
        with self._conn() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        try:
            with conn:  # commits, or rolls back on error
                yield conn
        finally:
            conn.close()

    # --- projects -------------------------------------------------------

    def create_project(self, name: str, criteria: dict) -> int:
        _validate_criteria(criteria)
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO projects (name, criteria, created_at) VALUES (?, ?, ?)",
                (name.strip(), json.dumps(criteria), _now()),
            )
            return cur.lastrowid

    def list_projects(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute("SELECT id, name FROM projects ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]

    def get_project(self, project_id: int) -> dict:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
        if row is None:
            raise KeyError(f"No project with id {project_id}")
        return {**dict(row), "criteria": json.loads(row["criteria"])}

    def get_project_by_name(self, name: str) -> dict:
        with self._conn() as conn:
            row = conn.execute("SELECT id FROM projects WHERE name = ?", (name,)).fetchone()
        if row is None:
            raise KeyError(f"No project named {name!r}")
        return self.get_project(row["id"])

    def update_criteria(self, project_id: int, criteria: dict) -> None:
        _validate_criteria(criteria)
        with self._conn() as conn:
            conn.execute("UPDATE projects SET criteria = ? WHERE id = ?", (json.dumps(criteria), project_id))

    def delete_project(self, project_id: int) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))

    # --- papers ---------------------------------------------------------

    def add_papers(self, project_id: int, records: list[dict]) -> tuple[int, int]:
        """Add papers (dicts with title, and optionally abstract, doi, year,
        label). Returns (added, duplicates skipped)."""
        added = 0
        now = _now()
        with self._conn() as conn:
            for r in records:
                label = r.get("label") or None
                cur = conn.execute(
                    "INSERT OR IGNORE INTO papers (project_id, dedupe_key, title, abstract, doi, year,"
                    " label, label_source, labeled_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        project_id, dedupe_key(r), r["title"], r.get("abstract") or "", r.get("doi") or "",
                        str(r.get("year") or ""), label, "import" if label else None, now if label else None,
                    ),
                )
                added += cur.rowcount
        return added, len(records) - added

    def set_label(self, paper_id: int, label: str | None, note: str = "") -> None:
        """Record a reviewer's decision, or clear it with label=None."""
        if label is not None and label not in LABELS:
            raise ValueError(f"label must be one of {LABELS} or None, not {label!r}")
        with self._conn() as conn:
            conn.execute(
                "UPDATE papers SET label = ?, label_source = ?, label_note = ?, labeled_at = ? WHERE id = ?",
                (label, "reviewer" if label else None, note if label else "", _now(), paper_id),
            )

    def papers(self, project_id: int) -> pd.DataFrame:
        """Every paper with its latest successful AI decision under the
        project's current criteria (empty if not screened yet), and the
        final decision: the reviewer's or imported label if there is one,
        else the AI's."""
        fp = criteria_fingerprint(self.get_project(project_id)["criteria"])
        with self._conn() as conn:
            df = pd.read_sql_query(
                """
                SELECT p.id, p.title, p.abstract, p.doi, p.year, p.label, p.label_source, p.label_note,
                       s.decision AS ai_decision, s.rationale AS ai_rationale, s.confidence AS ai_confidence,
                       s.flagged AS ai_flagged, s.mode AS ai_mode, s.model AS ai_model,
                       (SELECT e.error FROM screenings e WHERE e.paper_id = p.id AND e.criteria_fp = :fp
                        ORDER BY e.id DESC LIMIT 1) AS last_error
                FROM papers p
                LEFT JOIN screenings s ON s.id = (
                    SELECT id FROM screenings WHERE paper_id = p.id AND criteria_fp = :fp AND error = ''
                    ORDER BY id DESC LIMIT 1)
                WHERE p.project_id = :project_id
                ORDER BY p.id
                """,
                conn,
                params={"fp": fp, "project_id": project_id},
            )
        return df.assign(final_decision=df["label"].fillna(df["ai_decision"]).fillna(""))

    def papers_to_screen(self, project_id: int, criteria_fp: str, mode: str) -> list[dict]:
        """Unlabeled papers with no successful decision for these criteria and
        mode. Failed attempts are retried."""
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT id, title, abstract, doi FROM papers p
                WHERE project_id = ? AND label IS NULL AND NOT EXISTS (
                    SELECT 1 FROM screenings s WHERE s.paper_id = p.id AND s.criteria_fp = ?
                    AND s.mode = ? AND s.error = '')
                ORDER BY id
                """,
                (project_id, criteria_fp, mode),
            ).fetchall()
        return [dict(r) for r in rows]

    def record_screening(self, paper_id: int, criteria_fp: str, mode: str, model: str,
                         result: dict | None, error: str = "") -> None:
        result = result or {"decision": "unsure", "rationale": "", "confidence": 0.0}
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO screenings (paper_id, criteria_fp, mode, model, decision, rationale, confidence,"
                " flagged, error, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    paper_id, criteria_fp, mode, model, result["decision"], result.get("rationale", ""),
                    float(result.get("confidence", 0.0)), int(bool(result.get("flagged_for_review"))),
                    error, _now(),
                ),
            )

    # --- examples collection ---------------------------------------------

    def unindexed_labels(self, project_id: int) -> list[dict]:
        """Papers whose label changed since the examples collection was last synced."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id, title, abstract, doi, label, label_note, labeled_at FROM papers"
                " WHERE project_id = ? AND labeled_at IS NOT indexed_at",
                (project_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def mark_indexed(self, changes: list[dict]) -> None:
        with self._conn() as conn:
            conn.executemany(
                "UPDATE papers SET indexed_at = ? WHERE id = ?",
                [(c["labeled_at"], c["id"]) for c in changes],
            )

    def count_labeled(self, project_id: int) -> int:
        with self._conn() as conn:
            return conn.execute(
                "SELECT COUNT(*) FROM papers WHERE project_id = ? AND label IS NOT NULL", (project_id,)
            ).fetchone()[0]


def _validate_criteria(criteria: dict) -> None:
    if not str(criteria.get("review_topic", "")).strip():
        raise ValueError("The review topic is required.")
    if not criteria.get("inclusion_criteria") and not criteria.get("exclusion_criteria"):
        raise ValueError("Give at least one inclusion or exclusion criterion.")


def sync_examples(store: ReviewStore, project_id: int) -> int:
    """Bring the project's examples collection in line with its labels:
    embed newly labeled papers and drop ones whose label was removed.
    Returns the number of papers changed."""
    changes = store.unindexed_labels(project_id)
    if not changes:
        return 0
    chroma = get_examples_store(examples_collection(project_id))
    labeled = [c for c in changes if c["label"]]
    removed = [str(c["id"]) for c in changes if not c["label"]]
    if labeled:
        chroma.add_documents(
            [
                Document(
                    page_content=f"Title: {c['title']}\n\nAbstract: {c['abstract']}",
                    metadata={"doi": c["doi"], "title": c["title"], "decision": c["label"], "rationale": c["label_note"]},
                )
                for c in labeled
            ],
            ids=[str(c["id"]) for c in labeled],  # upsert, so relabeling replaces the old example
        )
    if removed:
        chroma.delete(ids=removed)
    store.mark_indexed(changes)
    return len(changes)


def delete_project(store: ReviewStore, project_id: int) -> None:
    """Delete a project, its papers and decisions, and its examples collection."""
    from langchain_chroma import Chroma

    from src.config import CHROMA_PERSIST_DIR

    Chroma(
        collection_name=examples_collection(project_id),
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_PERSIST_DIR,
    ).delete_collection()
    store.delete_project(project_id)


# --- import --------------------------------------------------------------

_DOI_PREFIX = re.compile(r"^(https?://(dx\.)?doi\.org/|doi:\s*)", re.IGNORECASE)

COLUMN_ALIASES = {
    "title": ("title", "ti", "article title", "primary title"),
    "abstract": ("abstract", "ab", "abstract note", "abstractnote"),
    "doi": ("doi", "do"),
    "year": ("year", "publication year", "py", "date"),
    "label": ("label", "decision", "gold_label", "included", "include", "screening decision"),
}

_LABEL_VALUES = {
    "include": "include", "included": "include", "yes": "include", "y": "include", "1": "include", "true": "include",
    "exclude": "exclude", "excluded": "exclude", "no": "exclude", "n": "exclude", "0": "exclude", "false": "exclude",
}


def normalise_doi(doi: str) -> str:
    return _DOI_PREFIX.sub("", (doi or "").strip()).lower()


def dedupe_key(record: dict) -> str:
    doi = normalise_doi(record.get("doi") or "")
    if doi:
        return f"doi:{doi}"
    return "title:" + re.sub(r"[^a-z0-9]", "", record["title"].lower())


def _parse_label(value) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return _LABEL_VALUES.get(str(value).strip().lower())


def parse_csv(data: bytes) -> list[dict]:
    """Records from a CSV. Column names are matched case-insensitively
    against common export names; only a title column is required. A label
    column (include/exclude, yes/no, 1/0) marks already-screened papers."""
    df = pd.read_csv(io.BytesIO(data), dtype=str, keep_default_na=False)
    by_lower = {c.strip().lower(): c for c in df.columns}
    columns = {
        field: next((by_lower[a] for a in aliases if a in by_lower), None)
        for field, aliases in COLUMN_ALIASES.items()
    }
    if columns["title"] is None:
        raise ValueError(f"No title column found. Columns were: {list(df.columns)}")
    records = []
    for _, row in df.iterrows():
        get = lambda field: row[columns[field]].strip() if columns[field] else ""
        if not get("title"):
            continue
        records.append({
            "title": get("title"), "abstract": get("abstract"), "doi": normalise_doi(get("doi")),
            "year": get("year")[:4], "label": _parse_label(get("label")),
        })
    return records


def parse_ris(data: bytes) -> list[dict]:
    """Records from a RIS file, as exported by Zotero, EndNote, Mendeley and
    most bibliographic databases."""
    text = data.decode("utf-8-sig", errors="replace")
    records, current, last_tag = [], {}, None
    for line in text.splitlines():
        match = re.match(r"^([A-Z][A-Z0-9])  -\s?(.*)$", line)
        if not match:
            if last_tag and line.strip():  # continuation of a wrapped field
                current[last_tag][-1] += " " + line.strip()
            continue
        tag, value = match.group(1), match.group(2).strip()
        if tag == "ER":
            records.append(current)
            current, last_tag = {}, None
            continue
        current.setdefault(tag, []).append(value)
        last_tag = tag

    def first(rec, *tags):
        return next((rec[t][0] for t in tags if rec.get(t)), "")

    return [
        {
            "title": first(r, "TI", "T1", "CT", "BT"),
            "abstract": first(r, "AB", "N2"),
            "doi": normalise_doi(first(r, "DO")),
            "year": first(r, "PY", "Y1", "DA")[:4],
            "label": None,
        }
        for r in records
        if first(r, "TI", "T1", "CT", "BT")
    ]


def parse_upload(filename: str, data: bytes) -> list[dict]:
    if filename.lower().endswith((".ris", ".txt")):
        return parse_ris(data)
    if filename.lower().endswith(".csv"):
        return parse_csv(data)
    raise ValueError("Upload a .csv or .ris file.")
