"""
Batch-screens a project's unlabeled papers.

Papers are screened SCREENING_CONCURRENCY at a time, and each decision is
written to the database as soon as it arrives. Stopping a run (closing the
browser tab, Ctrl-C) loses at most the papers in flight; the next run picks
up the papers that still have no decision under the current criteria and
mode, including any that failed.

Long runs are better started from the command line than from the UI:

    python -m src.batch --project "My review" [--agent] [--limit 200]
"""
import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

from src.config import SCREENING_CONCURRENCY, chat_model_name
from src.projects import ReviewStore, examples_collection, sync_examples
from src.retriever import get_examples_store
from src.screening_agent_tools import build_screening_agent, screen_paper_agent
from src.screening_chain import criteria_fingerprint, screen_paper

log = logging.getLogger(__name__)

MODES = ("chain", "agent")


def screen_project(
    store: ReviewStore,
    project_id: int,
    mode: str = "chain",
    limit: int | None = None,
    on_result: Callable[[int, int, dict, dict | None, str], None] | None = None,
    concurrency: int = SCREENING_CONCURRENCY,
    chat_model=None,
) -> dict:
    """Screen up to `limit` of the project's remaining papers.

    `on_result(done, total, paper, result, error)` is called after each paper,
    in the calling thread. Returns counts of screened, failed and remaining papers.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, not {mode!r}")
    project = store.get_project(project_id)
    criteria = project["criteria"]
    fp = criteria_fingerprint(criteria)
    sync_examples(store, project_id)  # reviewer decisions since the last run become examples
    has_examples = store.count_labeled(project_id) > 0
    collection = examples_collection(project_id)

    todo = store.papers_to_screen(project_id, fp, mode)
    remaining_before = len(todo)
    todo = todo[:limit] if limit else todo

    if mode == "agent":
        agent = build_screening_agent(chat_model=chat_model, criteria=criteria, examples_collection=collection)

        def screen(paper):
            return screen_paper_agent(paper["title"], paper["abstract"], doi=paper["doi"], agent=agent)
    else:
        examples_store = get_examples_store(collection) if has_examples else None

        def screen(paper):
            text = f"Title: {paper['title']}\n\nAbstract: {paper['abstract']}"
            examples = examples_store.similarity_search(text, k=3) if examples_store else []
            return screen_paper(paper["title"], paper["abstract"], criteria=criteria,
                                example_docs=examples, chat_model=chat_model)

    model = chat_model_name()
    done = failed = 0
    executor = ThreadPoolExecutor(max_workers=max(1, concurrency))
    try:
        futures = {executor.submit(screen, paper): paper for paper in todo}
        for future in as_completed(futures):
            paper = futures[future]
            try:
                result, error = future.result().model_dump(), ""
            except Exception as e:  # API error after retries, agent step limit, bad output
                result, error = None, f"{type(e).__name__}: {e}"[:300]
                failed += 1
                log.warning("Screening paper %s failed: %s", paper["id"], error)
            store.record_screening(paper["id"], fp, mode, model, result, error)
            done += 1
            if on_result:
                on_result(done, len(todo), paper, result, error)
    finally:
        # On interruption, drop queued papers instead of waiting for them.
        executor.shutdown(wait=False, cancel_futures=True)

    return {"screened": done - failed, "failed": failed, "remaining": remaining_before - (done - failed)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", required=True, help="project name, as shown in the app")
    parser.add_argument("--agent", action="store_true", help="use the tool-calling agent instead of the fixed chain")
    parser.add_argument("--limit", type=int, default=None, help="screen at most this many papers")
    parser.add_argument("--concurrency", type=int, default=SCREENING_CONCURRENCY)
    args = parser.parse_args()

    store = ReviewStore()
    project = store.get_project_by_name(args.project)

    def report(done, total, paper, result, error):
        outcome = f"ERROR {error}" if error else result["decision"].upper()
        print(f"[{done}/{total}] {outcome:8s} {paper['title'][:80]}")

    summary = screen_project(store, project["id"], mode="agent" if args.agent else "chain",
                             limit=args.limit, on_result=report, concurrency=args.concurrency)
    print(f"Screened {summary['screened']}, failed {summary['failed']}, "
          f"{summary['remaining']} still to screen.")


if __name__ == "__main__":
    main()
