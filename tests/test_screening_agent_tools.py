"""
Unit tests for the tool-calling screening agent. No LLM or embedding calls:
tools are exercised directly and the compiled agent is replaced with a fake.
"""
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, ToolMessage

import src.screening_agent_tools as sat
from src.screening_agent_tools import (
    AgentScreeningDecision,
    PaperContext,
    flag_for_human_review,
    search_similar_examples,
    screen_paper_agent,
)


def test_search_similar_examples_formats_and_clamps_k(monkeypatch):
    calls = {}

    def fake_retrieve(text, k=3):
        calls["k"] = k
        return [Document(page_content="x", metadata={"title": "Ex", "decision": "exclude", "rationale": "Pediatric."})]

    monkeypatch.setattr(sat, "retrieve_similar_examples", fake_retrieve)
    out = search_similar_examples.invoke({"query": "children heart disease", "k": 50})

    assert calls["k"] == sat.MAX_EXAMPLES_PER_SEARCH
    assert "[EXCLUDE] Ex" in out


def test_flag_for_human_review_appends_to_queue(tmp_path, monkeypatch):
    queue = tmp_path / "queue.jsonl"
    monkeypatch.setattr(sat, "REVIEW_QUEUE_PATH", queue)
    runtime = SimpleNamespace(context=PaperContext(doi="10.1/x", title="Paper X"))

    msg = flag_for_human_review.func(reason="Outcome unclear", runtime=runtime)

    entry = json.loads(queue.read_text().strip())
    assert entry["doi"] == "10.1/x"
    assert entry["reason"] == "Outcome unclear"
    assert "10.1/x" in msg


def _fake_agent(messages, structured):
    agent = MagicMock()
    agent.invoke.return_value = {"messages": messages, "structured_response": structured}
    return agent


def test_screen_paper_agent_returns_structured_decision():
    structured = AgentScreeningDecision(decision="include", rationale="Meets all criteria.", confidence=0.9)
    messages = [
        AIMessage(content="", tool_calls=[{"name": "search_similar_examples", "args": {"query": "q"}, "id": "1"}]),
        ToolMessage(content="...", tool_call_id="1"),
    ]
    agent = _fake_agent(messages, structured)

    result = screen_paper_agent("T", "A", doi="10.1/y", agent=agent)

    assert result.decision == "include"
    assert result.flagged_for_review is False
    assert agent.invoke.call_args.kwargs["context"] == PaperContext(doi="10.1/y", title="T")


def test_screen_paper_agent_flag_forces_unsure():
    # Model called the flag tool but still answered 'exclude' -> trust the tool trace.
    structured = AgentScreeningDecision(decision="exclude", rationale="Unclear outcome.", confidence=0.4)
    messages = [
        AIMessage(content="", tool_calls=[{"name": "flag_for_human_review", "args": {"reason": "r"}, "id": "2"}]),
        ToolMessage(content="Flagged", tool_call_id="2"),
    ]

    result = screen_paper_agent("T", "A", agent=_fake_agent(messages, structured))

    assert result.decision == "unsure"
    assert result.flagged_for_review is True


def test_screen_paper_agent_reports_search_and_flag_counts():
    structured = AgentScreeningDecision(decision="exclude", rationale="r", confidence=0.5)
    messages = [
        AIMessage(content="", tool_calls=[
            {"name": "search_similar_examples", "args": {"query": "q1"}, "id": "1"},
            {"name": "search_similar_examples", "args": {"query": "q2"}, "id": "2"},
        ]),
        AIMessage(content="", tool_calls=[{"name": "flag_for_human_review", "args": {"reason": "r"}, "id": "3"}]),
    ]
    stats = {}

    screen_paper_agent("T", "A", agent=_fake_agent(messages, structured), stats=stats)

    assert stats == {"n_searches": 2, "n_flags": 1}
