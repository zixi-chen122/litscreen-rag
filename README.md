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

It is usable on your own review: create a project with your criteria,
import your search results (RIS or CSV), label a few papers, let the AI
screen the rest, and work through a review queue of everything it was unsure
about plus spot checks of its decisions. See
[Screening your own review](#screening-your-own-review).

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
                          screening_chain.py          qa_agent.py
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
│   ├── screening_chain.py    # RAG-grounded include/exclude decision chain
│   ├── screening_agent_tools.py  # tool-using agent: searches examples, looks up OpenAlex metadata, can flag for review
│   ├── qa_agent.py           # RetrievalQA over the corpus with citations
│   ├── datasets.py           # resolves --dataset to examples, eval papers and criteria
│   ├── load_synergy.py       # turns a SYNERGY dataset into seed/test splits + criteria
│   ├── projects.py           # review projects in SQLite: import (CSV/RIS), dedupe, labels, AI decisions
│   ├── batch.py              # resumable, concurrent batch screening of a project (also a CLI)
│   ├── evaluate.py           # precision/recall/F1 against gold labels
│   └── summarize_runs.py     # mean and range of each setup across repeat runs
├── data/
│   ├── criteria.yaml          # example inclusion/exclusion criteria
│   ├── sample_papers.csv      # example paper corpus to screen
│   └── labeled_examples.csv   # small set of already-screened examples (few-shot pool)
├── tests/                      # unit tests (LLM calls are mocked — no API key needed)
├── app.py                      # Streamlit app: screen your own review
├── demo_app.py                 # Streamlit benchmark demo on the sample / SYNERGY datasets
├── Dockerfile
├── requirements.txt            # runtime dependencies (requirements-dev.txt adds pytest, ruff)
├── .env.example
└── .github/workflows/ci.yml    # lint, tests and Docker build on every push
```

## Setup

1. **Clone and install**
   ```bash
   git clone <your-repo-url>
   cd litscreen-rag
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements-dev.txt   # or requirements.txt to run only
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

5. **Launch the benchmark demo**
   ```bash
   streamlit run demo_app.py
   ```

## Screening your own review

```bash
streamlit run app.py
```

1. **Create a project** in the sidebar with the review question and its
   inclusion and exclusion criteria, one per line.
2. **Import papers** as RIS (Zotero, EndNote, Mendeley, most databases) or
   CSV. Duplicates are dropped by DOI, or by title when there is no DOI. A
   CSV column named `label`, `decision` or `included` marks papers you have
   already screened.
3. **Label 10 or more papers yourself**, including some includes (Review tab,
   "All unscreened"). Labeled papers are the examples the AI retrieves to stay
   consistent with your judgement; each project has its own.
4. **Run AI screening.** Decisions are saved as they arrive, so a run can be
   stopped and resumed. For large sets use the CLI, which does not depend on a
   browser tab staying open:
   ```bash
   python -m src.batch --project "My review" [--agent] [--limit 500]
   ```
5. **Review.** Work through the papers the AI was unsure about or flagged,
   and spot-check its includes and excludes. Your decisions override the AI's,
   become new examples for later runs, and feed the agreement figures on the
   Results tab, including how many AI excludes you overturned.
6. **Export** every decision, with the AI's rationale, as CSV.

Editing the criteria starts a fresh AI pass; earlier decisions are kept in the
database but not shown. The AI is a second screener, not a replacement: treat
its excludes as unverified until the agreement figures on your own spot
checks justify relying on them.

## Deployment

The Docker image runs `app.py` as a non-root user and keeps all state (the
SQLite database and Chroma collections) on the `/data` volume:

```bash
docker build -t litscreen .
docker run -d -p 8501:8501 --env-file .env -v litscreen-data:/data litscreen
```

Set `APP_PASSWORD` whenever the app is reachable by anyone else, and put it
behind HTTPS (a reverse proxy, or a platform such as Azure Container Apps).
Back up the `/data` volume: it holds your team's screening decisions. See
`.env.example` for timeouts, retries and screening concurrency. The app is
built for a small team on one instance; SQLite and the local Chroma store do
not support several replicas.

## Running the tests

```bash
pytest -v
ruff check .
```

Tests mock the LLM calls, so they run in CI without any API key or cost.

## Using your own data

Replace `data/sample_papers.csv` with your own paper set (columns: `title`,
`abstract`, `doi`, optionally `gold_label`), edit `data/criteria.yaml` with
your review's actual inclusion/exclusion criteria, and re-run `ingest.py`.

## Evaluating on a SYNERGY dataset

[SYNERGY](https://github.com/asreview/synergy-dataset) provides real
systematic-review screening decisions. Prepare one into a seed split (ingested
as retrieval examples) and a held-out test split (screened and scored):

```bash
synergy get -d Donners_2021 -o data/synergy
python -m src.load_synergy --dataset Donners_2021
python -m src.ingest --dataset Donners_2021
python -m src.evaluate --dataset Donners_2021 --run-tag run1   # add --agent for the agent
python -m src.evaluate --dataset Donners_2021 --k-examples 0 --run-tag run1   # no-retrieval baseline
python -m src.evaluate --dataset Donners_2021 --example-selection random --run-tag run1   # 3 random examples
python -m src.evaluate --dataset Donners_2021 --example-selection all --run-tag run1      # every seed example
python -m src.summarize_runs --dataset Donners_2021
```

`--run-tag` keeps repeat runs in separate files, and `summarize_runs` reports
each setup's mean and range across them. Preliminary results on two SYNERGY
reviews are in [RESULTS.md](RESULTS.md).

Check `data/synergy/prepared/<dataset>/criteria.json` first: the criteria are
split automatically and may need hand-editing. `evaluate` refuses to run if the
examples in Chroma came from a different dataset or contain any test paper.
`--dataset sample` (the default) uses the bundled data in `data/`.

## Roadmap / possible extensions

- Swap Chroma and SQLite for managed services (Azure AI Search, Postgres) to
  run several replicas.
- Per-user accounts (SSO) and an audit log of who made each decision.
- Add MLflow tracking for prompt versions and evaluation runs over time.
- Confirm the SYNERGY results on a review from SYNERGY's "test" group (see
  [RESULTS.md](RESULTS.md)).
