"""
Unit tests for the screening agent. All LLM/embedding calls are mocked or
avoided so these run without any API key or cost, including in CI.
"""
from unittest.mock import MagicMock
from langchain_core.documents import Document

from src.screening_chain import _format_examples, _load_criteria, screen_paper, ScreeningDecision


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
        "src.screening_chain.retrieve_similar_examples",
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
    import src.screening_chain as sa

    # `prompt | model` resolves via the *class's* __or__, not the instance's,
    # so patch it on the class for the duration of this test.
    monkeypatch.setattr(type(sa.SCREENING_PROMPT), "__or__", lambda self, other: other, raising=False)

    result = screen_paper("Some Title", "Some abstract text.", chat_model=fake_chat_model)

    assert result.decision == "include"
    assert result.confidence == 0.9
    fake_chat_model.with_structured_output.assert_called_once_with(ScreeningDecision)


def test_screen_paper_with_zero_examples_skips_retrieval(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("retrieval should be skipped when k_examples=0")

    monkeypatch.setattr("src.screening_chain.retrieve_similar_examples", fail_if_called)

    fake_structured_model = MagicMock()
    fake_structured_model.invoke.return_value = ScreeningDecision(
        decision="exclude", rationale="Fails inclusion criterion 1.", confidence=0.8
    )
    fake_chat_model = MagicMock()
    fake_chat_model.with_structured_output.return_value = fake_structured_model

    import src.screening_chain as sa

    monkeypatch.setattr(type(sa.SCREENING_PROMPT), "__or__", lambda self, other: other, raising=False)

    result = screen_paper("Some Title", "Some abstract text.", k_examples=0, chat_model=fake_chat_model)

    assert result.decision == "exclude"
    assert fake_structured_model.invoke.call_args.args[0]["examples"] == "(none provided)"


def test_screen_paper_uses_given_example_docs_without_retrieving(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("retrieval should be skipped when example_docs is given")

    monkeypatch.setattr("src.screening_chain.retrieve_similar_examples", fail_if_called)

    fake_structured_model = MagicMock()
    fake_structured_model.invoke.return_value = ScreeningDecision(
        decision="include", rationale="Matches all inclusion criteria.", confidence=0.9
    )
    fake_chat_model = MagicMock()
    fake_chat_model.with_structured_output.return_value = fake_structured_model

    import src.screening_chain as sa

    monkeypatch.setattr(type(sa.SCREENING_PROMPT), "__or__", lambda self, other: other, raising=False)

    docs = [Document(page_content="x", metadata={"title": "Given Example", "decision": "include", "rationale": "r"})]
    screen_paper("Some Title", "Some abstract text.", chat_model=fake_chat_model, example_docs=docs)

    assert "Given Example" in fake_structured_model.invoke.call_args.args[0]["examples"]


def _fake_model_returning_include():
    fake_structured_model = MagicMock()
    fake_structured_model.invoke.return_value = ScreeningDecision(
        decision="include", rationale="Matches all inclusion criteria.", confidence=0.9
    )
    fake_chat_model = MagicMock()
    fake_chat_model.with_structured_output.return_value = fake_structured_model
    return fake_chat_model, fake_structured_model


def _pool(n):
    return [
        Document(page_content=f"p{i}", metadata={"doi": f"10.1/{i:02d}", "title": f"Pool {i}", "decision": "exclude", "rationale": "r"})
        for i in range(n)
    ]


def test_screen_paper_all_selection_includes_every_example_without_similarity(monkeypatch):
    import src.screening_chain as sa

    monkeypatch.setattr(sa, "retrieve_similar_examples", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("no similarity search")))
    monkeypatch.setattr(sa, "all_examples", lambda: _pool(5))
    monkeypatch.setattr(type(sa.SCREENING_PROMPT), "__or__", lambda self, other: other, raising=False)
    chat_model, structured = _fake_model_returning_include()

    screen_paper("T", "A", chat_model=chat_model, example_selection="all")

    inputs = structured.invoke.call_args.args[0]
    assert all(f"Pool {i}" in inputs["examples"] for i in range(5))
    assert inputs["examples_heading"] == "PREVIOUSLY-SCREENED EXAMPLES"


def test_screen_paper_similar_selection_keeps_original_heading(monkeypatch):
    import src.screening_chain as sa

    monkeypatch.setattr(sa, "retrieve_similar_examples", lambda text, k=3: [])
    monkeypatch.setattr(type(sa.SCREENING_PROMPT), "__or__", lambda self, other: other, raising=False)
    chat_model, structured = _fake_model_returning_include()

    screen_paper("T", "A", chat_model=chat_model)

    assert structured.invoke.call_args.args[0]["examples_heading"] == "SIMILAR PREVIOUSLY-SCREENED EXAMPLES"


def test_random_examples_is_reproducible_per_paper_and_varies_across_papers(monkeypatch):
    import src.retriever as rt

    monkeypatch.setattr(rt, "all_examples", lambda collection_name=None: _pool(60))
    first = [d.metadata["doi"] for d in rt.random_examples("paper one", k=3)]
    again = [d.metadata["doi"] for d in rt.random_examples("paper one", k=3)]
    others = {tuple(d.metadata["doi"] for d in rt.random_examples(f"paper {i}", k=3)) for i in range(10)}
    assert first == again
    assert len(first) == 3 and len(set(first)) == 3
    assert len(others) > 1
    assert len(rt.random_examples("x", k=100)) == 60
