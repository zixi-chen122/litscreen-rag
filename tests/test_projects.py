"""
Unit tests for project storage, file import and batch screening. Uses a
temporary SQLite file; Chroma and the LLM are replaced with fakes.
"""
from unittest.mock import MagicMock

import pytest

import src.batch as batch
import src.projects as projects
from src.projects import ReviewStore, dedupe_key, parse_csv, parse_ris
from src.screening_chain import ScreeningDecision, criteria_fingerprint

CRITERIA = {"review_topic": "RPM in heart failure", "inclusion_criteria": ["Adults"], "exclusion_criteria": ["Children"]}


@pytest.fixture
def store(tmp_path):
    return ReviewStore(str(tmp_path / "test.db"))


@pytest.fixture
def project(store):
    return store.create_project("HF review", CRITERIA)


def test_create_project_rejects_empty_criteria(store):
    with pytest.raises(ValueError):
        store.create_project("Empty", {"review_topic": "x", "inclusion_criteria": [], "exclusion_criteria": []})


def test_add_papers_skips_duplicates_by_doi_and_title(store, project):
    added, dupes = store.add_papers(project, [
        {"title": "Paper A", "doi": "10.1/a"},
        {"title": "Paper A (other database)", "doi": "https://doi.org/10.1/A"},
        {"title": "No DOI: Paper B"},
        {"title": "no doi paper b"},
    ])
    assert (added, dupes) == (2, 2)


def test_dedupe_key_prefers_doi():
    assert dedupe_key({"title": "T", "doi": "DOI: 10.5/X"}) == "doi:10.5/x"
    assert dedupe_key({"title": "A Title!", "doi": ""}) == "title:atitle"


def test_parse_csv_maps_common_column_names_and_labels():
    data = b"Title,Abstract Note,DOI,Publication Year,Included\nT1,A1,https://doi.org/10.1/x,2019,yes\nT2,,,,\n,skipped,,,\n"
    records = parse_csv(data)
    assert records == [
        {"title": "T1", "abstract": "A1", "doi": "10.1/x", "year": "2019", "label": "include"},
        {"title": "T2", "abstract": "", "doi": "", "year": "", "label": None},
    ]


def test_parse_csv_requires_title_column():
    with pytest.raises(ValueError, match="title"):
        parse_csv(b"name,abstract\nx,y\n")


def test_parse_ris_reads_records():
    data = (
        b"TY  - JOUR\nTI  - Telemonitoring in heart failure\nAB  - A randomized trial.\nDO  - 10.1/ris\n"
        b"PY  - 2020///\nER  - \n\nTY  - JOUR\nT1  - Second paper\nN2  - Abstract two\n"
        b"continued here\nER  - \n"
    )
    records = parse_ris(data)
    assert [r["title"] for r in records] == ["Telemonitoring in heart failure", "Second paper"]
    assert records[0]["doi"] == "10.1/ris" and records[0]["year"] == "2020"
    assert records[1]["abstract"] == "Abstract two continued here"


def test_papers_to_screen_excludes_labeled_and_already_screened(store, project):
    store.add_papers(project, [{"title": "Labeled", "label": "include"}, {"title": "U1"}, {"title": "U2"}])
    fp = criteria_fingerprint(CRITERIA)
    todo = store.papers_to_screen(project, fp, "chain")
    assert [p["title"] for p in todo] == ["U1", "U2"]

    store.record_screening(todo[0]["id"], fp, "chain", "m", {"decision": "exclude", "rationale": "r", "confidence": 0.9})
    store.record_screening(todo[1]["id"], fp, "chain", "m", None, error="Timeout")  # failures are retried
    assert [p["title"] for p in store.papers_to_screen(project, fp, "chain")] == ["U2"]
    assert len(store.papers_to_screen(project, "other-criteria", "chain")) == 2


def test_papers_shows_latest_decision_under_current_criteria_and_final_label(store, project):
    store.add_papers(project, [{"title": "P1"}, {"title": "P2"}])
    ids = store.papers(project)["id"].tolist()
    fp = criteria_fingerprint(CRITERIA)
    store.record_screening(ids[0], "old-criteria", "chain", "m", {"decision": "include", "confidence": 1})
    store.record_screening(ids[0], fp, "chain", "m", {"decision": "unsure", "confidence": 0.3, "flagged_for_review": True})
    store.record_screening(ids[1], fp, "chain", "m", {"decision": "include", "confidence": 0.8})
    store.set_label(ids[1], "exclude", "Pediatric")

    df = store.papers(project).set_index("id")
    assert df.loc[ids[0], "ai_decision"] == "unsure" and df.loc[ids[0], "ai_flagged"] == 1
    assert df.loc[ids[0], "final_decision"] == "unsure"
    assert df.loc[ids[1], "final_decision"] == "exclude"  # the reviewer overrides the AI
    assert df.loc[ids[1], "label_source"] == "reviewer"


def test_set_label_rejects_unknown_labels(store, project):
    store.add_papers(project, [{"title": "P"}])
    with pytest.raises(ValueError):
        store.set_label(store.papers(project)["id"][0], "maybe")


def test_sync_examples_upserts_labels_and_removes_cleared_ones(store, project, monkeypatch):
    chroma = MagicMock()
    monkeypatch.setattr(projects, "get_examples_store", lambda name: chroma)
    store.add_papers(project, [{"title": "Imported", "label": "include"}, {"title": "Unlabeled"}])
    unlabeled_id = int(store.papers(project).query("title == 'Unlabeled'")["id"].iloc[0])

    assert projects.sync_examples(store, project) == 1
    docs = chroma.add_documents.call_args.args[0]
    assert docs[0].metadata["decision"] == "include"
    assert projects.sync_examples(store, project) == 0  # nothing changed since

    store.set_label(unlabeled_id, "exclude", "Children only")
    projects.sync_examples(store, project)
    assert chroma.add_documents.call_args.kwargs["ids"] == [str(unlabeled_id)]
    assert chroma.add_documents.call_args.args[0][0].metadata["rationale"] == "Children only"

    store.set_label(unlabeled_id, None)
    projects.sync_examples(store, project)
    chroma.delete.assert_called_once_with(ids=[str(unlabeled_id)])


def test_screen_project_records_results_and_failures_then_resumes(store, project, monkeypatch):
    monkeypatch.setattr(batch, "sync_examples", lambda store, project_id: 0)
    store.add_papers(project, [{"title": "Good"}, {"title": "Bad"}, {"title": "Later"}])

    def fake_screen(title, abstract, criteria, example_docs, chat_model):
        if title == "Bad":
            raise TimeoutError("model timed out")
        return ScreeningDecision(decision="include", rationale="ok", confidence=0.9)

    monkeypatch.setattr(batch, "screen_paper", fake_screen)
    seen = []
    summary = batch.screen_project(store, project, limit=2, concurrency=1,
                                   on_result=lambda done, total, paper, result, error: seen.append((done, total, error)))

    assert summary == {"screened": 1, "failed": 1, "remaining": 2}
    assert [s[:2] for s in seen] == [(1, 2), (2, 2)]
    df = store.papers(project).set_index("title")
    assert df.loc["Good", "ai_decision"] == "include"
    assert "TimeoutError" in df.loc["Bad", "last_error"]

    monkeypatch.setattr(batch, "screen_paper", lambda *a, **k: ScreeningDecision(decision="exclude", rationale="r", confidence=0.5))
    assert batch.screen_project(store, project, concurrency=2) == {"screened": 2, "failed": 0, "remaining": 0}
    assert store.papers(project).set_index("title").loc["Bad", "ai_decision"] == "exclude"
