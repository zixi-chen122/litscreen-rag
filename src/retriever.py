"""
Thin wrappers around the two Chroma collections built by ingest.py.
"""
from langchain_chroma import Chroma

from src.config import get_embeddings, CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME
from src.ingest import EXAMPLES_COLLECTION_NAME


def get_papers_store() -> Chroma:
    return Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_PERSIST_DIR,
    )


def get_examples_store() -> Chroma:
    return Chroma(
        collection_name=EXAMPLES_COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_PERSIST_DIR,
    )


def retrieve_similar_examples(paper_text: str, k: int = 3):
    """Retrieve the k most similar already-screened examples for few-shot grounding."""
    store = get_examples_store()
    return store.similarity_search(paper_text, k=k)


def retrieve_relevant_papers(query: str, k: int = 5):
    """Retrieve the k most relevant papers from the corpus for a free-text query."""
    store = get_papers_store()
    return store.similarity_search(query, k=k)
