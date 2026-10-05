"""
Agentic version of the screener, built with LangChain's `create_agent`.

Unlike the fixed chain in screening_chain.py (which always retrieves k
examples up front and makes one LLM call), here the model decides what to do:

  - search_similar_examples : query the labeled-examples collection, as many
                              times and with whatever query the model needs
                              (e.g. once for the population, once for design)
  - flag_for_human_review   : route an ambiguous paper to a human reviewer,
                              appending it to a JSONL review queue

The final answer is returned as the same ScreeningDecision schema, so
evaluate.py can compare both screeners on identical metrics.
"""
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from langchain.agents import create_agent
from langchain.tools import ToolRuntime, tool
from pydantic import BaseModel, Field

from src.config import get_chat_model
from src.retriever import retrieve_similar_examples
from src.screening_chain import _format_criteria, _format_examples, _load_criteria

REVIEW_QUEUE_PATH = Path("data/review_queue.jsonl")
MAX_EXAMPLES_PER_SEARCH = 5


@dataclass
class PaperContext:
    """Per-invocation context injected into tools (not visible to the model)."""

    doi: str
    title: str


class AgentScreeningDecision(BaseModel):
    decision: str = Field(description="One of: 'include', 'exclude', 'unsure'")
    rationale: str = Field(description="One or two sentence justification citing which criterion applied")
    confidence: float = Field(description="Confidence in the decision, from 0.0 to 1.0")
    flagged_for_review: bool = Field(
        default=False, description="True if flag_for_human_review was called for this paper"
    )


@tool
def search_similar_examples(query: str, k: int = 3) -> str:
    """Search previously-screened papers similar to `query` and return their
    human decisions and rationales. Use the candidate's title/abstract, or a
    focused query about one criterion (e.g. its study population or study
    design) when that criterion is borderline."""
    k = max(1, min(k, MAX_EXAMPLES_PER_SEARCH))
    return _format_examples(retrieve_similar_examples(query, k=k))


@tool
def flag_for_human_review(reason: str, runtime: ToolRuntime[PaperContext]) -> str:
    """Send the current paper to a human reviewer. Call this when the abstract
    lacks the information needed to apply a criterion, or when similar
    examples conflict. `reason` should name the unresolved criterion."""
    ctx = runtime.context
    entry = {
        "doi": ctx.doi,
        "title": ctx.title,
        "reason": reason,
        "flagged_at": datetime.now(timezone.utc).isoformat(),
    }
    REVIEW_QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REVIEW_QUEUE_PATH.open("a") as f:
        f.write(json.dumps(entry) + "\n")
    return f"Flagged {ctx.doi} for human review."


def _build_system_prompt(criteria: dict | None = None) -> str:
    criteria = criteria or _load_criteria()
    inclusion = _format_criteria(criteria["inclusion_criteria"])
    exclusion = _format_criteria(criteria["exclusion_criteria"])
    return (
        "You are a systematic-review screening assistant. Decide whether a "
        "candidate paper should be included in or excluded from the review, "
        "strictly based on the criteria below.\n\n"
        f"REVIEW TOPIC:\n{criteria['review_topic']}\n\n"
        f"INCLUSION CRITERIA:\n{inclusion}\n\n"
        f"EXCLUSION CRITERIA:\n{exclusion}\n\n"
        "PROCESS:\n"
        "1. Always call search_similar_examples at least once so your decision "
        "is consistent with prior screening decisions. Search again with a "
        "focused query if one criterion is borderline.\n"
        "2. If the abstract does not give enough information to confidently "
        "apply a criterion, call flag_for_human_review and answer 'unsure' "
        "with flagged_for_review=true. Do not guess.\n"
        "3. Otherwise answer 'include' or 'exclude', citing the deciding criterion."
    )


def build_screening_agent(chat_model=None, criteria: dict | None = None):
    """Build the agent once; reuse it across papers. `criteria` defaults to the sample dataset's."""
    return create_agent(
        model=chat_model or get_chat_model(),
        tools=[search_similar_examples, flag_for_human_review],
        system_prompt=_build_system_prompt(criteria),
        response_format=AgentScreeningDecision,
        context_schema=PaperContext,
    )


def screen_paper_agent(
    title: str, abstract: str, doi: str = "", agent=None, max_steps: int = 12, stats: dict | None = None
) -> AgentScreeningDecision:
    """Screen a single paper with the tool-using agent.

    If `stats` is given, it is filled with how many times the agent searched
    and flagged, so evaluations can report the agent's extra cost per paper.
    """
    agent = agent or build_screening_agent()
    result = agent.invoke(
        {"messages": [{"role": "user", "content": f"Title: {title}\nAbstract: {abstract}"}]},
        context=PaperContext(doi=doi, title=title),
        config={"recursion_limit": max_steps},
    )
    decision: AgentScreeningDecision = result["structured_response"]

    tool_calls = [
        call["name"]
        for msg in result["messages"]
        for call in (getattr(msg, "tool_calls", None) or [])
    ]
    if stats is not None:
        stats["n_searches"] = tool_calls.count("search_similar_examples")
        stats["n_flags"] = tool_calls.count("flag_for_human_review")

    # Trust the tool trace over the model's self-report for the flag.
    flagged = "flag_for_human_review" in tool_calls
    decision.flagged_for_review = flagged
    if flagged:
        decision.decision = "unsure"
    return decision
