"""
Loads the paper corpus and the labeled-examples pool, embeds them, and
persists two Chroma collections:

  - litscreen_papers            : the corpus to be screened / queried
  - litscreen_labeled_examples  : already-screened papers used for
                                   few-shot grounding in screening_agent.py
"""
import pandas as pd
from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.config import get_embeddings, CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME

PAPERS_CSV = "data/sample_papers.csv"
EXAMPLES_CSV = "data/labeled_examples.csv"
EXAMPLES_COLLECTION_NAME = "litscreen_labeled_examples"


def _row_to_document(row: pd.Series, extra_metadata: dict) -> Document:
    text = f"Title: {row['title']}\n\nAbstract: {row['abstract']}"
    metadata = {"doi": row["doi"], "title": row["title"], **extra_metadata}
    return Document(page_content=text, metadata=metadata)


def build_papers_store() -> Chroma:
    df = pd.read_csv(PAPERS_CSV)
    docs = [_row_to_document(row, {}) for _, row in df.iterrows()]
    embeddings = get_embeddings()
    store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=CHROMA_COLLECTION_NAME,
        persist_directory=CHROMA_PERSIST_DIR,
    )
    print(f"Ingested {len(docs)} papers into collection '{CHROMA_COLLECTION_NAME}'.")
    return store


def build_examples_store() -> Chroma:
    df = pd.read_csv(EXAMPLES_CSV)
    docs = [
        _row_to_document(row, {"decision": row["decision"], "rationale": row["rationale"]})
        for _, row in df.iterrows()
    ]
    embeddings = get_embeddings()
    store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=EXAMPLES_COLLECTION_NAME,
        persist_directory=CHROMA_PERSIST_DIR,
    )
    print(f"Ingested {len(docs)} labeled examples into collection '{EXAMPLES_COLLECTION_NAME}'.")
    return store


if __name__ == "__main__":
    build_papers_store()
    build_examples_store()
