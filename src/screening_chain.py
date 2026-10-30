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
import hashlib
import json
from typing import Literal

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

from src.config import get_chat_model
from src.datasets import SAMPLE, load_criteria
from src.retriever import all_examples, random_examples, retrieve_similar_examples

# How examples are chosen. 'similar' is the RAG setup; 'random' and 'all' are
# its controls: random k examples, and the whole seed set in every prompt.
ExampleSelection = Literal["similar", "random", "all"]
EXAMPLE_SELECTIONS = ("similar", "random", "all")


class ScreeningDecision(BaseModel):
    decision: Literal["include", "exclude", "unsure"] = Field(description="One of: 'include', 'exclude', 'unsure'")
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
            "{examples_heading} (for consistency, not as strict rules):\n"
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


def criteria_fingerprint(criteria: dict) -> str:
    """Short stable hash of the criteria, so runs with edited criteria are
    never pooled with runs on the original ones."""
    return hashlib.sha1(json.dumps(criteria, sort_keys=True).encode()).hexdigest()[:8]


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
    title: str,
    abstract: str,
    k_examples: int = 3,
    chat_model=None,
    criteria: dict | None = None,
    example_docs=None,
    example_selection: ExampleSelection = "similar",
) -> ScreeningDecision:
    """Screen a single paper and return a structured ScreeningDecision.

    `criteria` defaults to the sample dataset's criteria.yaml. k_examples=0
    skips retrieval entirely (the criteria-only baseline); the prompt is
    otherwise unchanged so the two runs are directly comparable.
    `example_selection` picks the k examples by similarity (the default),
    at random, or ('all') ignores k and includes every seed example.
    `example_docs` uses already-retrieved examples instead of retrieving
    (the demo UI retrieves once so it can show them).
    """
    criteria = criteria or _load_criteria()
    paper_text = f"Title: {title}\n\nAbstract: {abstract}"
    if example_docs is not None:
        examples = _format_examples(example_docs)
    elif example_selection == "all":
        examples = _format_examples(all_examples())
    elif k_examples > 0 and example_selection == "random":
        examples = _format_examples(random_examples(paper_text, k=k_examples))
    elif k_examples > 0:
        examples = _format_examples(retrieve_similar_examples(paper_text, k=k_examples))
    else:
        examples = "(none provided)"
    # Only the similarity setup may call its examples similar; its wording is
    # unchanged so its runs stay comparable with earlier ones.
    heading = "SIMILAR PREVIOUSLY-SCREENED EXAMPLES" if example_selection == "similar" else "PREVIOUSLY-SCREENED EXAMPLES"

    model = chat_model or get_chat_model()
    structured_model = model.with_structured_output(ScreeningDecision)

    chain = SCREENING_PROMPT | structured_model
    return chain.invoke(
        {
            "review_topic": criteria["review_topic"],
            "inclusion_criteria": _format_criteria(criteria["inclusion_criteria"]),
            "exclusion_criteria": _format_criteria(criteria["exclusion_criteria"]),
            "examples_heading": heading,
            "examples": examples,
            "title": title,
            "abstract": abstract,
        }
    )
