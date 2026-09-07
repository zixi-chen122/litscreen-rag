import pandas as pd

from src.ingest import _row_to_document


def test_row_to_document_builds_expected_content_and_metadata():
    row = pd.Series({"doi": "10.0/x", "title": "A Title", "abstract": "An abstract."})
    doc = _row_to_document(row, extra_metadata={"decision": "include"})

    assert "Title: A Title" in doc.page_content
    assert "Abstract: An abstract." in doc.page_content
    assert doc.metadata["doi"] == "10.0/x"
    assert doc.metadata["title"] == "A Title"
    assert doc.metadata["decision"] == "include"


def test_sample_papers_csv_has_expected_columns():
    df = pd.read_csv("data/sample_papers.csv")
    assert set(["doi", "title", "abstract", "gold_label"]).issubset(df.columns)
    assert len(df) > 0
    assert set(df["gold_label"].unique()).issubset({"include", "exclude", "unsure"})
