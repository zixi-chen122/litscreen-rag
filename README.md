# LitScreen RAG

An AI-powered systematic literature review screening assistant, built as a full
Retrieval-Augmented Generation (RAG) pipeline rather than a single prompt call.

This project takes a common but tedious research task — deciding which papers
in a search result set should be **included** or **excluded** from a
systematic review, based on a set of inclusion/exclusion criteria — and
automates it with an LLM that is grounded in:

1. The review's written criteria (PICO-style), and
2. Retrieved examples of *already screened* papers with similar content,
   so the model's decisions stay consistent across a large batch instead of
   drifting paper-to-paper.

It also exposes a general **RAG question-answering** mode: ask a natural
language question ("which papers use a randomized controlled design?") and
get an answer grounded in retrieved abstracts, with citations back to the
source papers.

## Why this project exists

This was built as a portfolio project to demonstrate hands-on experience with
the modern GenAI application stack: RAG architecture, vector databases,
prompt engineering, agent/tool orchestration, and basic MLOps (evaluation
harness, CI, containerization) — going beyond the simpler prompt-classification
approach used in an earlier academic version of this idea.

## Architecture

```
                      ┌─────────────────────┐
                      │   criteria.yaml      │  (inclusion/exclusion rules)
                      └──────────┬───────────┘
                                 │
   papers.csv ──► ingest.py ──► Chroma vector store ◄── retriever.py
   (title,                          │                        │
    abstract)                       │                        │
                                     ▼                        ▼
                          screening_agent.py          qa_agent.py
                          (per-paper decision,         (free-text Q&A
                           grounded in similar          over the corpus,
                           screened examples)           with citations)
                                     │                        │
                                     ▼                        ▼
                             evaluate.py               app.py (Streamlit UI)
                        (precision/recall vs
                          gold labels)
```

- **Vector store**: [Chroma](https://www.trychroma.com/) — local, file-backed,
  zero infrastructure to run or demo.
- **LLM**: Azure OpenAI (`gpt-5-mini` deployment) by default, with a one-line
  switch to plain OpenAI for local testing without an Azure account.
- **Orchestration**: [LangChain](https://python.langchain.com/) for the
  retrieval chains and structured-output parsing.
- **Evaluation**: precision/recall/F1 against a small gold-labeled sample set,
  because "it looks plausible" isn't good enough for a screening tool.

## Project structure

```
litscreen-rag/
├── src/
│   ├── config.py            # env/config, LLM + embeddings client setup
│   ├── ingest.py             # load papers, chunk, embed, persist to Chroma
│   ├── retriever.py          # similarity search + few-shot example retrieval
│   ├── screening_agent.py    # RAG-grounded include/exclude decision chain
│   ├── qa_agent.py           # RetrievalQA over the corpus with citations
│   └── evaluate.py           # precision/recall/F1 against gold labels
├── data/
│   ├── criteria.yaml          # example inclusion/exclusion criteria
│   ├── sample_papers.csv      # example paper corpus to screen
│   └── labeled_examples.csv   # small set of already-screened examples (few-shot pool)
├── tests/                      # unit tests (LLM calls are mocked — no API key needed)
├── app.py                      # Streamlit demo UI
├── Dockerfile
├── requirements.txt
└── .github/workflows/ci.yml    # runs tests on every push
```

## Setup

1. **Clone and install**
   ```bash
   git clone <your-repo-url>
   cd litscreen-rag
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Configure credentials** — copy `.env.example` to `.env` and fill in:

   For **Azure AI Foundry** (recommended, matches production cloud setups).
   Newly-created resources expose the **v1 API** (no `api-version` needed) —
   check your resource's Playground "Call model" tab: if the endpoint shown
   ends in `/openai/v1`, use this:
   ```
   LLM_PROVIDER=azure_v1
   AZURE_OPENAI_ENDPOINT=https://<your-resource>.services.ai.azure.com/openai/v1
   AZURE_OPENAI_API_KEY=...
   AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-5-mini
   AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-small
   ```

   If instead your resource shows the older endpoint style
   (`https://<resource>.openai.azure.com/`), use the classic path:
   ```
   LLM_PROVIDER=azure
   AZURE_OPENAI_API_KEY=...
   AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com/
   AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-5-mini
   AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-small
   AZURE_OPENAI_API_VERSION=2024-10-21
   ```

   Or plain **OpenAI** (fastest way to try it without a cloud account):
   ```
   LLM_PROVIDER=openai
   OPENAI_API_KEY=...
   ```

3. **Ingest the sample corpus**
   ```bash
   python -m src.ingest
   ```

4. **Run screening on the sample papers**
   ```bash
   python -m src.evaluate
   ```
   This prints per-paper decisions plus precision/recall/F1 against the gold
   labels in `data/sample_papers.csv`.

5. **Launch the interactive demo**
   ```bash
   streamlit run app.py
   ```

## Running the tests

```bash
pytest -v
```

Tests mock the LLM calls, so they run in CI without any API key or cost.

## Using your own data

Replace `data/sample_papers.csv` with your own paper set (columns: `title`,
`abstract`, `doi`, optionally `gold_label`), edit `data/criteria.yaml` with
your review's actual inclusion/exclusion criteria, and re-run `ingest.py`.

## Roadmap / possible extensions

- Swap Chroma for a managed vector DB (Pinecone, Azure AI Search) for
  larger corpora.
- Add MLflow tracking for prompt versions and evaluation runs over time.
- Add a LangChain agent with tools (`search_corpus`, `screen_paper`,
  `flag_for_human_review`) instead of a fixed chain, for more complex
  multi-step review workflows.
