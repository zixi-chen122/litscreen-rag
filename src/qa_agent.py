"""
Free-text RAG question-answering over the paper corpus.

Example: "Which papers use a randomized controlled trial design?" retrieves
the most relevant abstracts from the corpus and asks the LLM to answer
grounded only in those abstracts, citing paper titles/DOIs.
"""
from langchain_core.prompts import ChatPromptTemplate

from src.config import get_chat_model
from src.retriever import retrieve_relevant_papers

QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You answer questions about a corpus of research paper abstracts. "
            "Only use the provided context — if the context does not contain the "
            "answer, say so explicitly rather than guessing. Always cite the paper "
            "title and DOI for any claim you make.",
        ),
        (
            "human",
            "CONTEXT (retrieved papers):\n{context}\n\nQUESTION: {question}",
        ),
    ]
)


def _format_context(docs) -> str:
    blocks = []
    for doc in docs:
        blocks.append(f"[{doc.metadata.get('doi')}] {doc.metadata.get('title')}\n{doc.page_content}")
    return "\n\n".join(blocks)


def answer_question(question: str, k: int = 5, chat_model=None) -> str:
    docs = retrieve_relevant_papers(question, k=k)
    model = chat_model or get_chat_model()
    chain = QA_PROMPT | model
    response = chain.invoke({"context": _format_context(docs), "question": question})
    return response.content
