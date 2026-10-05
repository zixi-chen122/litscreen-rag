"""
Loads the paper corpus and the labeled-examples pool, embeds them, and
persists two Chroma collections:

  - litscreen_papers            : the corpus to be screened / queried
  - litscreen_labeled_examples  : already-screened papers used for
                                   few-shot grounding in screening_chain.py

--dataset picks the source data (see src/datasets.py). For a SYNERGY dataset
only the examples collection is built, from the seed split: the test split is
what gets scored, so it is never put in Chroma.
"""
import argparse

import pandas as pd
from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.config import get_embeddings, CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME
from src.datasets import SAMPLE, load_eval_papers, load_examples

EXAMPLES_COLLECTION_NAME = "litscreen_labeled_examples"


def _row_to_document(row: pd.Series, extra_metadata: dict) -> Document:
    text = f"Title: {row['title']}\n\nAbstract: {row['abstract']}"
    metadata = {"doi": row["doi"], "title": row["title"], **extra_metadata}
    return Document(page_content=text, metadata=metadata)


def _rebuild_collection(docs: list[Document], collection_name: str, dataset: str) -> Chroma:
    """Drop any existing collection and recreate it from `docs`.

    Chroma.from_documents appends to an existing collection, so without this
    stale or duplicate documents from earlier runs would linger and leak into
    retrieval. The dataset name is stored on the collection so evaluate.py can
    check it is scoring against the examples it expects.
    """
    embeddings = get_embeddings()
    Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
    ).delete_collection()
    return Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=CHROMA_PERSIST_DIR,
        collection_metadata={"dataset": dataset},
    )


def build_papers_store(dataset: str = SAMPLE) -> Chroma:
    df = load_eval_papers(dataset)
    docs = [_row_to_document(row, {}) for _, row in df.iterrows()]
    store = _rebuild_collection(docs, CHROMA_COLLECTION_NAME, dataset)
    print(f"Ingested {len(docs)} papers into collection '{CHROMA_COLLECTION_NAME}'.")
    return store


def build_examples_store(dataset: str = SAMPLE) -> Chroma:
    df = load_examples(dataset)
    docs = [
        _row_to_document(row, {"decision": row["decision"], "rationale": row["rationale"]})
        for _, row in df.iterrows()
    ]
    store = _rebuild_collection(docs, EXAMPLES_COLLECTION_NAME, dataset)
    print(f"Ingested {len(docs)} labeled examples into collection '{EXAMPLES_COLLECTION_NAME}'.")
    return store


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=SAMPLE, help="'sample' or a prepared SYNERGY dataset name, e.g. Donners_2021")
    dataset = parser.parse_args().dataset
    if dataset == SAMPLE:
        build_papers_store(dataset)
    else:
        print(f"Skipping '{CHROMA_COLLECTION_NAME}': the {dataset} test split must stay out of Chroma.")
    build_examples_store(dataset)
