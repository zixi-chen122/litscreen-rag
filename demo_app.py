"""
Streamlit demo UI for LitScreen RAG.

Run with: streamlit run demo_app.py

This is the benchmark demo: the bundled sample and prepared SYNERGY datasets,
with gold labels. For screening your own review, use app.py.

The dataset dropdown picks the dataset; it starts on whichever dataset was last
ingested, and switching re-ingests, so the criteria, paper list and
retrieved examples always come from the same review.
"""
import pandas as pd
import streamlit as st

from src.datasets import SAMPLE, list_datasets, load_criteria, load_eval_papers, load_label_source
from src.ingest import build_examples_store, build_papers_store
from src.screening_chain import screen_paper
from src.screening_agent_tools import build_screening_agent, screen_paper_agent
from src.qa_agent import answer_question
from src.retriever import get_examples_store, retrieve_similar_examples


def ingested_dataset() -> str:
    """Dataset name stored on the examples collection by ingest.py."""
    return (get_examples_store()._collection.metadata or {}).get("dataset") or SAMPLE


st.set_page_config(page_title="LitScreen RAG", layout="wide")

st.title("📚 LitScreen RAG — Literature Screening Assistant")
st.caption(
    "A RAG-grounded assistant for systematic-review paper screening. "
    "Decisions are grounded in written criteria + retrieved similar "
    "already-screened examples, not a bare prompt call."
)

# Switching datasets re-ingests its screened examples (same as
# `python -m src.ingest --dataset <name>`), so retrieval always matches.
available = list_datasets()
current = ingested_dataset()
if "dataset" not in st.session_state:
    st.session_state.dataset = current if current in available else SAMPLE
dataset = st.selectbox(
    "Dataset",
    available,
    key="dataset",
    help="Switching re-embeds the dataset's screened examples; it takes a few seconds.",
)
if dataset != current:
    with st.spinner(f"Loading {dataset} screened examples into Chroma..."):
        if dataset == SAMPLE:
            build_papers_store(dataset)
        build_examples_store(dataset)

criteria = load_criteria(dataset)
papers = load_eval_papers(dataset).reset_index(drop=True)
label_source = load_label_source(dataset)

st.markdown(f"**Review:** {criteria['review_topic']}")

# Criteria are editable for this session only; the dataset files are never
# changed. Unedited criteria are passed through exactly as loaded, so the
# prompt matches what evaluate.py used.
original_inc = "\n".join(criteria["inclusion_criteria"])
original_exc = "\n".join(criteria["exclusion_criteria"])
inc_key, exc_key = f"inclusion_{dataset}", f"exclusion_{dataset}"
with st.expander("Inclusion and exclusion criteria (editable)"):
    st.caption("One criterion per line. Edits apply to screening in this session only; the dataset files are not changed.")
    inclusion_text = st.text_area("Inclusion criteria", original_inc, height=200, key=inc_key)
    exclusion_text = st.text_area("Exclusion criteria", original_exc, height=200, key=exc_key)
    criteria_edited = inclusion_text != original_inc or exclusion_text != original_exc
    if criteria_edited and st.button("Reset to the original criteria"):
        del st.session_state[inc_key], st.session_state[exc_key]
        st.rerun()

if criteria_edited:
    lines = lambda text: [line.strip() for line in text.splitlines() if line.strip()]
    criteria = {**criteria, "inclusion_criteria": lines(inclusion_text), "exclusion_criteria": lines(exclusion_text)}
    st.warning("Screening with **edited** criteria. Results are not comparable to RESULTS.md.")

tab1, tab2 = st.tabs(["Screen a paper", "Ask a question about the corpus"])

with tab1:
    st.subheader("Screen a single paper")
    paper_labels = [f"{i + 1}. [{row['gold_label']}] {row['title']}" for i, row in papers.iterrows()]
    picked = st.selectbox(
        f"Load a paper from the {dataset} test set (or paste your own below)",
        ["(none)"] + paper_labels,
        key=f"paper_{dataset}",
    )
    paper = papers.iloc[paper_labels.index(picked)] if picked in paper_labels else None
    title = st.text_input("Paper title", paper["title"] if paper is not None else "")
    abstract = st.text_area("Abstract", paper["abstract"] if paper is not None else "", height=200)
    mode = st.radio(
        "Screening mode",
        ["Fixed chain", "Agent"],
        horizontal=True,
        help="Fixed chain: always retrieves the 3 most similar examples, then makes one model call. "
        "Agent: decides when and what to search, and can flag the paper for human review (2-3 model calls).",
    )
    compare = st.checkbox("Also screen without retrieval (criteria-only baseline)")

    def show_decision(result):
        color = {"include": "green", "exclude": "red", "unsure": "orange"}.get(result.decision, "gray")
        st.markdown(f"### Decision: :{color}[{result.decision.upper()}]")
        st.write(f"**Confidence:** {result.confidence:.2f}")
        st.write(f"**Rationale:** {result.rationale}")
        if getattr(result, "flagged_for_review", False):
            st.write(":orange[**Flagged for human review**]")

    if st.button("Screen this paper", type="primary"):
        if not title or not abstract:
            st.warning("Please provide both a title and an abstract.")
        else:
            examples, trace = [], []
            with st.spinner("Screening..."):
                if mode == "Agent":
                    agent = build_screening_agent(criteria=criteria)
                    doi = str(paper["doi"]) if paper is not None and pd.notna(paper["doi"]) else ""
                    result = screen_paper_agent(title, abstract, doi=doi, agent=agent, trace=trace)
                else:
                    examples = retrieve_similar_examples(f"Title: {title}\n\nAbstract: {abstract}", k=3)
                    result = screen_paper(title, abstract, criteria=criteria, example_docs=examples)
                baseline = screen_paper(title, abstract, k_examples=0, criteria=criteria) if compare else None

            if baseline is not None:
                col_main, col_base = st.columns(2)
                with col_main:
                    st.caption("Agent" if mode == "Agent" else "With retrieval (RAG)")
                    show_decision(result)
                with col_base:
                    st.caption("Without retrieval (baseline)")
                    show_decision(baseline)
            else:
                show_decision(result)

            if paper is not None:
                gold = "title/abstract screening" if label_source == "abstract" else "final, after full text"
                line = f"**Reviewers' decision ({gold}):** {paper['gold_label']}"
                if label_source == "abstract" and pd.notna(paper.get("label_final")):
                    line += f"  \n**Final decision (after full text):** {paper['label_final']}"
                st.write(line)

            if mode == "Agent":
                st.markdown("#### What the agent did")
                for i, call in enumerate(trace, 1):
                    if call["name"] == "search_similar_examples":
                        st.write(f"{i}. **Searched** screened examples for: _{call['args'].get('query', '')}_ (k={call['args'].get('k', 3)})")
                    elif call["name"] == "flag_for_human_review":
                        st.write(f"{i}. **Flagged for human review:** {call['args'].get('reason', '')}")
                    else:
                        st.write(f"{i}. {call['name']}: {call['args']}")
                if not trace:
                    st.write("No tool calls: the agent answered directly.")
            else:
                st.markdown("#### Retrieved examples given to the model")
                if not examples:
                    st.info("No screened examples found. Run `python -m src.ingest` first.")
                for i, doc in enumerate(examples, 1):
                    label = doc.metadata.get("decision", "?")
                    color = {"include": "green", "exclude": "red"}.get(label, "gray")
                    with st.expander(f"{i}. :{color}[{label.upper()}] {doc.metadata.get('title', '')}"):
                        rationale = doc.metadata.get("rationale") or "(none recorded: SYNERGY has labels only)"
                        st.write(f"**Reviewer rationale:** {rationale}")
                        st.write(doc.page_content)

    st.divider()
    st.subheader(f"Or batch-screen the {dataset} test set")
    n_batch = st.number_input("Number of papers", min_value=1, max_value=len(papers), value=min(10, len(papers)))
    if st.button("Run batch screening"):
        df = papers.head(int(n_batch))
        progress = st.progress(0)
        results = []
        for i, row in df.iterrows():
            r = screen_paper(row["title"], row["abstract"], criteria=criteria)
            results.append({"title": row["title"], "gold_label": row["gold_label"], "decision": r.decision, "confidence": r.confidence, "rationale": r.rationale})
            progress.progress((i + 1) / len(df))
        st.dataframe(pd.DataFrame(results))

with tab2:
    st.subheader("Ask a question about the paper corpus")
    st.caption("Searches the bundled sample corpus: SYNERGY test papers are kept out of Chroma.")
    question = st.text_input("Your question", "Which papers use a randomized controlled trial design?")
    if st.button("Ask", type="primary"):
        with st.spinner("Retrieving relevant papers and answering..."):
            answer = answer_question(question)
        st.markdown(answer)
