"""
Core RAG screening logic.

For each candidate paper, the decision is grounded in two retrieved sources:
  1. The review's written inclusion/exclusion criteria (data/criteria.yaml,
     or a SYNERGY dataset's criteria.json -- see src/datasets.py)
  2. The most similar already-screened examples (retrieved from the
     'litscreen_labeled_examples' Chroma collection), so decisions on
     borderline papers stay consistent with prior human/agent judgments
     instead of being made in isolation.

This is what makes it RAG rather than a bare prompt-classifier: the model
never decides from the abstract alone, it decides from the abstract +
retrieved grounding context.
"""
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

from src.config import get_chat_model
from src.datasets import SAMPLE, load_criteria
from src.retriever import retrieve_similar_examples


class ScreeningDecision(BaseModel):
    decision: str = Field(description="One of: 'include', 'exclude', 'unsure'")
    rationale: str = Field(description="One or two sentence justification citing which criterion applied")
    confidence: float = Field(description="Confidence in the decision, from 0.0 to 1.0")


SCREENING_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a systematic-review screening assistant. You decide whether a "
            "candidate paper should be included or excluded from a review, strictly "
            "based on the given criteria and grounded by similar previously-screened "
            "examples. Be conservative: if the abstract does not give enough "
            "information to confidently apply a criterion, respond 'unsure' rather "
            "than guessing.",
        ),
        (
            "human",
            "REVIEW TOPIC:\n{review_topic}\n\n"
            "INCLUSION CRITERIA:\n{inclusion_criteria}\n\n"
            "EXCLUSION CRITERIA:\n{exclusion_criteria}\n\n"
            "SIMILAR PREVIOUSLY-SCREENED EXAMPLES (for consistency, not as strict rules):\n"
            "{examples}\n\n"
            "CANDIDATE PAPER TO SCREEN:\n"
            "Title: {title}\n"
            "Abstract: {abstract}\n\n"
            "Decide include / exclude / unsure for the candidate paper.",
        ),
    ]
)


def _load_criteria(dataset: str = SAMPLE) -> dict:
    return load_criteria(dataset)


def _format_criteria(items: list[str]) -> str:
    return "\n".join(f"- {c}" for c in items) if items else "(none specified)"


def _format_examples(example_docs) -> str:
    if not example_docs:
        return "(no similar examples found)"
    lines = []
    for doc in example_docs:
        lines.append(
            f"- [{doc.metadata.get('decision', '?').upper()}] {doc.metadata.get('title', '')}\n"
            f"  Rationale: {doc.metadata.get('rationale', '')}"
        )
    return "\n".join(lines)


def screen_paper(
    title: str, abstract: str, k_examples: int = 3, chat_model=None, criteria: dict | None = None
) -> ScreeningDecision:
    """Screen a single paper and return a structured ScreeningDecision.

    `criteria` defaults to the sample dataset's criteria.yaml. k_examples=0
    skips retrieval entirely (the criteria-only baseline); the prompt is
    otherwise unchanged so the two runs are directly comparable.
    """
    criteria = criteria or _load_criteria()
    if k_examples > 0:
        examples = _format_examples(
            retrieve_similar_examples(f"Title: {title}\n\nAbstract: {abstract}", k=k_examples)
        )
    else:
        examples = "(none provided)"

    model = chat_model or get_chat_model()
    structured_model = model.with_structured_output(ScreeningDecision)

    chain = SCREENING_PROMPT | structured_model
    return chain.invoke(
        {
            "review_topic": criteria["review_topic"],
            "inclusion_criteria": _format_criteria(criteria["inclusion_criteria"]),
            "exclusion_criteria": _format_criteria(criteria["exclusion_criteria"]),
            "examples": examples,
            "title": title,
            "abstract": abstract,
        }
    )
