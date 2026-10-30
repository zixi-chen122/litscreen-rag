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
    get_paper_metadata,
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


def test_openalex_key_handles_dois_openalex_ids_and_unknowns():
    assert sat._openalex_key("10.1186/abc") == "doi:10.1186/abc"
    assert sat._openalex_key("https://openalex.org/w2273754290") == "W2273754290"
    assert sat._openalex_key("") is None


def test_get_paper_metadata_formats_openalex_record(monkeypatch):
    work = {
        "type": "review",
        "publication_year": 2019,
        "language": "en",
        "primary_location": {"source": {"display_name": "Some Conf", "type": "conference"}},
        "primary_topic": {"display_name": "Heart failure"},
        "keywords": [{"display_name": "telemonitoring"}, {"display_name": "readmission"}],
    }
    requested = []
    monkeypatch.setattr(sat, "_fetch_openalex_work", lambda key: requested.append(key) or work)
    runtime = SimpleNamespace(context=PaperContext(doi="10.1/x", title="Paper X"))

    out = get_paper_metadata.func(runtime=runtime)

    assert requested == ["doi:10.1/x"]
    assert "Publication type: review" in out
    assert "Venue: Some Conf (conference)" in out
    assert "Keywords: telemonitoring, readmission" in out


def test_get_paper_metadata_without_record_says_so(monkeypatch):
    monkeypatch.setattr(sat, "_fetch_openalex_work", lambda key: None)
    runtime = SimpleNamespace(context=PaperContext(doi="10.0001/synth", title="T"))

    assert "No bibliographic record" in get_paper_metadata.func(runtime=runtime)


def test_fetch_openalex_work_does_not_cache_failures(monkeypatch):
    import httpx

    monkeypatch.setattr(sat, "_openalex_cache", {})
    request = httpx.Request("GET", "https://api.openalex.org/works/doi:10.1/x")
    responses = [httpx.ConnectError("down"), httpx.Response(200, json={"type": "article"}, request=request)]

    def fake_get(url, params, timeout):
        item = responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(sat.httpx, "get", fake_get)

    assert sat._fetch_openalex_work("doi:10.1/x") is None
    assert sat._fetch_openalex_work("doi:10.1/x") == {"type": "article"}
    assert sat._fetch_openalex_work("doi:10.1/x") == {"type": "article"}  # cached, no third request


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

    assert stats == {"n_searches": 2, "n_metadata_lookups": 0, "n_flags": 1}


def test_screen_paper_agent_traces_tool_calls_in_order():
    structured = AgentScreeningDecision(decision="exclude", rationale="r", confidence=0.5)
    messages = [
        AIMessage(content="", tool_calls=[{"name": "search_similar_examples", "args": {"query": "q1"}, "id": "1"}]),
        AIMessage(content="", tool_calls=[{"name": "flag_for_human_review", "args": {"reason": "no outcome"}, "id": "2"}]),
    ]
    trace = []

    screen_paper_agent("T", "A", agent=_fake_agent(messages, structured), trace=trace)

    assert trace == [
        {"name": "search_similar_examples", "args": {"query": "q1"}},
        {"name": "flag_for_human_review", "args": {"reason": "no outcome"}},
    ]
