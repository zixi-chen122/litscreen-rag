"""
Thin wrappers around the two Chroma collections built by ingest.py.
"""
import hashlib
import random

from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.config import get_embeddings, CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME
from src.ingest import EXAMPLES_COLLECTION_NAME


def get_papers_store() -> Chroma:
    return Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_PERSIST_DIR,
    )


def get_examples_store(collection_name: str = EXAMPLES_COLLECTION_NAME) -> Chroma:
    return Chroma(
        collection_name=collection_name,
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_PERSIST_DIR,
    )


def retrieve_similar_examples(paper_text: str, k: int = 3, collection_name: str = EXAMPLES_COLLECTION_NAME):
    """Retrieve the k most similar already-screened examples for few-shot grounding."""
    store = get_examples_store(collection_name)
    return store.similarity_search(paper_text, k=k)


def all_examples(collection_name: str = EXAMPLES_COLLECTION_NAME) -> list[Document]:
    """Every already-screened example, in a fixed order (by DOI) so prompts
    are identical across runs. No embedding call is made."""
    got = get_examples_store(collection_name).get(include=["documents", "metadatas"])
    docs = [Document(page_content=text, metadata=meta) for text, meta in zip(got["documents"], got["metadatas"])]
    return sorted(docs, key=lambda d: d.metadata.get("doi", ""))


def random_examples(paper_text: str, k: int = 3, collection_name: str = EXAMPLES_COLLECTION_NAME) -> list[Document]:
    """k examples drawn at random, ignoring similarity: the control for
    retrieve_similar_examples. Seeded by the paper's text, so each paper gets
    its own draw but repeat runs see the same examples."""
    pool = all_examples(collection_name)
    seed = int(hashlib.sha1(paper_text.encode()).hexdigest()[:8], 16)
    return random.Random(seed).sample(pool, min(k, len(pool)))


def retrieve_relevant_papers(query: str, k: int = 5):
    """Retrieve the k most relevant papers from the corpus for a free-text query."""
    store = get_papers_store()
    return store.similarity_search(query, k=k)
