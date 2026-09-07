"""
Unit tests for the screening agent. All LLM/embedding calls are mocked or
avoided so these run without any API key or cost, including in CI.
"""
from unittest.mock import MagicMock
from langchain_core.documents import Document

from src.screening_agent import _format_examples, _load_criteria, screen_paper, ScreeningDecision


def test_load_criteria_has_expected_keys():
    criteria = _load_criteria()
    assert "review_topic" in criteria
    assert "inclusion_criteria" in criteria
    assert "exclusion_criteria" in criteria
    assert len(criteria["inclusion_criteria"]) > 0
    assert len(criteria["exclusion_criteria"]) > 0


def test_format_examples_empty():
    assert _format_examples([]) == "(no similar examples found)"


def test_format_examples_formats_decision_and_rationale():
    docs = [
        Document(
            page_content="irrelevant",
            metadata={"title": "Example Paper", "decision": "include", "rationale": "Meets all criteria."},
        )
    ]
    formatted = _format_examples(docs)
    assert "INCLUDE" in formatted
    assert "Example Paper" in formatted
    assert "Meets all criteria." in formatted


def test_screen_paper_uses_structured_output_and_retrieval(monkeypatch):
    # Avoid hitting a real vector store for retrieval.
    monkeypatch.setattr(
        "src.screening_agent.retrieve_similar_examples",
        lambda text, k=3: [],
    )

    expected = ScreeningDecision(decision="include", rationale="Matches all inclusion criteria.", confidence=0.9)

    # Build a fake chat model whose with_structured_output(...).invoke(...) returns `expected`.
    fake_structured_model = MagicMock()
    fake_structured_model.invoke.return_value = expected

    fake_chat_model = MagicMock()
    fake_chat_model.with_structured_output.return_value = fake_structured_model

    # LangChain's `prompt | model` pipe returns a Runnable; monkeypatch the
    # ChatPromptTemplate's __or__ to just return the structured model itself,
    # so `chain.invoke(...)` resolves to our fake result.
    import src.screening_agent as sa

    # `prompt | model` resolves via the *class's* __or__, not the instance's,
    # so patch it on the class for the duration of this test.
    monkeypatch.setattr(type(sa.SCREENING_PROMPT), "__or__", lambda self, other: other, raising=False)

    result = screen_paper("Some Title", "Some abstract text.", chat_model=fake_chat_model)

    assert result.decision == "include"
    assert result.confidence == 0.9
    fake_chat_model.with_structured_output.assert_called_once_with(ScreeningDecision)
