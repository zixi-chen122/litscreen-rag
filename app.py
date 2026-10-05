"""
Streamlit demo UI for LitScreen RAG.

Run with: streamlit run app.py
"""
import pandas as pd
import streamlit as st

from src.screening_chain import screen_paper
from src.qa_agent import answer_question

st.set_page_config(page_title="LitScreen RAG", layout="wide")
st.title("📚 LitScreen RAG — Literature Screening Assistant")
st.caption(
    "A RAG-grounded assistant for systematic-review paper screening. "
    "Decisions are grounded in written criteria + retrieved similar "
    "already-screened examples, not a bare prompt call."
)

tab1, tab2 = st.tabs(["Screen a paper", "Ask a question about the corpus"])

with tab1:
    st.subheader("Screen a single paper")
    title = st.text_input("Paper title", "")
    abstract = st.text_area("Abstract", "", height=200)

    if st.button("Screen this paper", type="primary"):
        if not title or not abstract:
            st.warning("Please provide both a title and an abstract.")
        else:
            with st.spinner("Retrieving similar examples and screening..."):
                result = screen_paper(title, abstract)
            color = {"include": "green", "exclude": "red", "unsure": "orange"}.get(result.decision, "gray")
            st.markdown(f"### Decision: :{color}[{result.decision.upper()}]")
            st.write(f"**Confidence:** {result.confidence:.2f}")
            st.write(f"**Rationale:** {result.rationale}")

    st.divider()
    st.subheader("Or batch-screen the sample corpus")
    if st.button("Run batch screening on data/sample_papers.csv"):
        df = pd.read_csv("data/sample_papers.csv")
        progress = st.progress(0)
        results = []
        for i, row in df.iterrows():
            r = screen_paper(row["title"], row["abstract"])
            results.append({"title": row["title"], "decision": r.decision, "confidence": r.confidence, "rationale": r.rationale})
            progress.progress((i + 1) / len(df))
        st.dataframe(pd.DataFrame(results))

with tab2:
    st.subheader("Ask a question about the paper corpus")
    question = st.text_input("Your question", "Which papers use a randomized controlled trial design?")
    if st.button("Ask", type="primary"):
        with st.spinner("Retrieving relevant papers and answering..."):
            answer = answer_question(question)
        st.markdown(answer)
