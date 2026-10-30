"""
LitScreen: screen your own systematic review.

Run with: streamlit run app.py

Workflow: create a project with your review's criteria, import the search
results (CSV or RIS), label a handful of papers yourself, let the AI screen
the rest, then work through the review queue -- every paper it is unsure
about or flagged, plus spot checks of its includes and excludes. Your
decisions become retrieval examples for later screening and always override
the AI's in the export.

Set APP_PASSWORD to require a password. For the benchmark demo on the sample
and SYNERGY datasets, run demo_app.py instead.
"""
import hmac
import os

import pandas as pd
import streamlit as st

from src.batch import screen_project
from src.config import SCREENING_CONCURRENCY
from src.projects import ReviewStore, delete_project, parse_upload
from src.screening_chain import criteria_fingerprint

st.set_page_config(page_title="LitScreen", page_icon="📚", layout="wide")

# Batches larger than this should run from the CLI, which survives a closed tab.
UI_BATCH_LIMIT = 500
EXPORT_COLUMNS = [
    "title", "abstract", "doi", "year", "final_decision", "label", "label_source", "label_note",
    "ai_decision", "ai_confidence", "ai_rationale", "ai_flagged", "ai_mode", "ai_model", "last_error",
]


def require_password() -> None:
    password = os.getenv("APP_PASSWORD")
    if not password or st.session_state.get("authenticated"):
        return
    st.title("📚 LitScreen")
    entered = st.text_input("Password", type="password")
    if entered and hmac.compare_digest(entered.encode(), password.encode()):
        st.session_state.authenticated = True
        st.rerun()
    elif entered:
        st.error("Wrong password.")
    st.stop()


@st.cache_resource
def get_store() -> ReviewStore:
    return ReviewStore()


def lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def criteria_form(prefix: str, criteria: dict | None = None) -> dict:
    criteria = criteria or {"review_topic": "", "inclusion_criteria": [], "exclusion_criteria": []}
    topic = st.text_area("Review topic / question", criteria["review_topic"], key=f"{prefix}_topic", height=80)
    inclusion = st.text_area("Inclusion criteria (one per line)", "\n".join(criteria["inclusion_criteria"]),
                             key=f"{prefix}_inc", height=160)
    exclusion = st.text_area("Exclusion criteria (one per line)", "\n".join(criteria["exclusion_criteria"]),
                             key=f"{prefix}_exc", height=160)
    return {"review_topic": topic.strip(), "inclusion_criteria": lines(inclusion), "exclusion_criteria": lines(exclusion)}


def decision_badge(decision: str | None) -> str:
    color = {"include": "green", "exclude": "red", "unsure": "orange"}.get(decision or "", "gray")
    return f":{color}[**{(decision or 'not screened').upper()}**]"


require_password()
store = get_store()

# --- sidebar: pick or create a project -------------------------------------

with st.sidebar:
    st.header("📚 LitScreen")
    project_list = store.list_projects()
    names = {p["name"]: p["id"] for p in project_list}
    # A widget's value can only be set before it is drawn, so creating or
    # deleting a project leaves the new selection here for the next run.
    if "pending_project" in st.session_state:
        pending = st.session_state.pop("pending_project")
        if pending is None:
            st.session_state.pop("project_name", None)
        else:
            st.session_state.project_name = pending
    selected = st.selectbox("Project", list(names), index=None if not names else 0,
                            placeholder="No projects yet", key="project_name")
    with st.expander("New project", expanded=not names):
        with st.form("new_project", clear_on_submit=False):
            new_name = st.text_input("Project name")
            new_criteria = criteria_form("new")
            if st.form_submit_button("Create project", type="primary"):
                try:
                    if not new_name.strip():
                        raise ValueError("Give the project a name.")
                    if new_name.strip() in names:
                        raise ValueError("A project with that name already exists.")
                    store.create_project(new_name, new_criteria)
                    st.session_state.pending_project = new_name.strip()
                    st.rerun()
                except ValueError as e:
                    st.error(str(e))

if selected is None:
    st.title("📚 LitScreen")
    st.info("Create a project in the sidebar to start: give it your review's topic and criteria.")
    st.stop()

project_id = names[selected]
project = store.get_project(project_id)
criteria = project["criteria"]
fp = criteria_fingerprint(criteria)
papers = store.papers(project_id)

st.title(selected)
st.caption(criteria["review_topic"])

unlabeled = papers[papers["label"].isna()]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Papers", len(papers))
c2.metric("Labeled by you or on import", int(papers["label"].notna().sum()))
c3.metric("Screened by AI", int(unlabeled["ai_decision"].notna().sum()))
c4.metric("Awaiting review", int((unlabeled["ai_decision"] == "unsure").sum()))

tab_import, tab_screen, tab_review, tab_results, tab_settings = st.tabs(
    ["1 · Import papers", "2 · AI screening", "3 · Review", "4 · Results & export", "Settings"]
)

# --- 1. import ----------------------------------------------------------------

with tab_import:
    st.markdown(
        "Upload search results as **RIS** (Zotero, EndNote, Mendeley, most databases) or **CSV**. "
        "CSV needs a title column; abstract, DOI and year are picked up when present. "
        "A CSV column named *label*, *decision* or *included* (include/exclude, yes/no, 1/0) marks papers "
        "you have already screened: they become examples the AI checks its decisions against. "
        "Duplicates (same DOI, or same title) are skipped."
    )
    uploads = st.file_uploader("Search results", type=["ris", "txt", "csv"], accept_multiple_files=True)
    if uploads and st.button("Import", type="primary"):
        for upload in uploads:
            try:
                records = parse_upload(upload.name, upload.getvalue())
            except Exception as e:  # malformed file
                st.error(f"{upload.name}: could not read it ({e}).")
                continue
            added, dupes = store.add_papers(project_id, records)
            no_abstract = sum(1 for r in records if not r["abstract"])
            st.success(f"{upload.name}: added {added} papers, skipped {dupes} duplicates."
                       + (f" {no_abstract} have no abstract and will likely be marked unsure." if no_abstract else ""))
        st.button("Refresh")

    n_labeled = int(papers["label"].notna().sum())
    if len(papers) and n_labeled < 10:
        st.info(
            f"Only {n_labeled} papers are labeled. Screening works best with at least 10 labeled examples, "
            "including some includes: label a few in the Review tab (\"All unscreened\") before running the AI."
        )
    if len(papers):
        st.dataframe(papers[["title", "year", "doi", "label"]], hide_index=True, use_container_width=True)

# --- 2. AI screening ------------------------------------------------------------

with tab_screen:
    mode_label = st.radio(
        "Screening mode",
        ["Fixed chain", "Agent"],
        horizontal=True,
        help="Fixed chain: retrieves the 3 most similar labeled papers, then makes one model call per paper. "
        "Agent: decides what to search, can look up the paper's publication type and venue on OpenAlex, "
        "and flags unclear papers for you (about 2-4 model calls per paper).",
    )
    mode = "agent" if mode_label == "Agent" else "chain"
    remaining = len(store.papers_to_screen(project_id, fp, mode))
    st.write(f"**{remaining}** unlabeled papers have no {mode_label.lower()} decision under the current criteria.")
    if remaining:
        n = st.number_input("Papers to screen in this run", 1, min(remaining, UI_BATCH_LIMIT),
                            min(remaining, 50 if mode == "agent" else 100))
        calls = int(n) * (3 if mode == "agent" else 1)
        st.caption(
            f"About {calls} model calls, {SCREENING_CONCURRENCY} papers at a time. Each decision is saved as it "
            "arrives: you can stop at any time and resume later. "
            + (f"For more than {UI_BATCH_LIMIT} papers, use the command line: "
               f"`python -m src.batch --project \"{selected}\"" + (" --agent" if mode == "agent" else "") + "`"
               if remaining > UI_BATCH_LIMIT else "")
        )
        if st.button("Start screening", type="primary"):
            progress = st.progress(0.0, text="Preparing examples...")
            log = st.empty()
            recent: list[str] = []

            def report(done, total, paper, result, error):
                progress.progress(done / total, text=f"{done} / {total} screened")
                outcome = f":gray[error: {error[:80]}]" if error else decision_badge(result["decision"])
                recent.insert(0, f"{outcome} {paper['title'][:120]}")
                log.markdown("  \n".join(recent[:10]))

            summary = screen_project(store, project_id, mode=mode, limit=int(n), on_result=report)
            if summary["failed"]:
                st.warning(f"{summary['failed']} papers failed (see the Results tab, column last_error). "
                           "They will be retried on the next run.")
            st.success(f"Screened {summary['screened']} papers. {summary['remaining']} left.")
            st.button("Refresh counts")

# --- 3. review ------------------------------------------------------------------

QUEUES = {
    "Unsure or flagged by the AI": lambda df: df["ai_decision"] == "unsure",
    "AI includes": lambda df: df["ai_decision"] == "include",
    "AI excludes (random spot check)": lambda df: df["ai_decision"] == "exclude",
    "All unscreened": lambda df: df["ai_decision"].isna(),
}

with tab_review:
    queue_name = st.selectbox("Queue", list(QUEUES))
    skipped = st.session_state.setdefault("skipped", set())
    queue = unlabeled[QUEUES[queue_name](unlabeled) & ~unlabeled["id"].isin(skipped)]
    if queue_name.startswith("AI excludes"):
        queue = queue.sample(frac=1, random_state=project_id)  # stable random order
    elif queue_name.startswith("AI includes") or queue_name.startswith("Unsure"):
        queue = queue.sort_values("ai_confidence")  # least confident first
    st.caption(f"{len(queue)} papers in this queue.")

    if (last := st.session_state.get("last_labeled")) and st.button("↶ Undo last decision"):
        store.set_label(last, None)
        del st.session_state["last_labeled"]
        st.rerun()

    if queue.empty:
        st.success("Nothing left in this queue.")
        if skipped and st.button("Show skipped papers again"):
            skipped.clear()
            st.rerun()
    else:
        paper = queue.iloc[0]
        st.subheader(paper["title"])
        meta = " · ".join(x for x in [paper["year"], f"doi:{paper['doi']}" if paper["doi"] else ""] if x)
        if meta:
            st.caption(meta)
        st.write(paper["abstract"] or "_No abstract._")
        if pd.notna(paper["ai_decision"]):
            st.markdown(
                f"AI decision: {decision_badge(paper['ai_decision'])} "
                f"(confidence {paper['ai_confidence']:.2f})"
                + (" · :orange[flagged for review]" if paper["ai_flagged"] else "")
            )
            st.write(f"_{paper['ai_rationale']}_")
        note = st.text_input("Note (optional, e.g. the deciding criterion; used as the example's rationale)",
                             key=f"note_{paper['id']}")
        b1, b2, b3, _ = st.columns([1, 1, 1, 4])
        for column, label in ((b1, "include"), (b2, "exclude")):
            if column.button(label.capitalize(), key=f"{label}_{paper['id']}", type="primary", use_container_width=True):
                store.set_label(int(paper["id"]), label, note)
                st.session_state.last_labeled = int(paper["id"])
                st.rerun()
        if b3.button("Skip", key=f"skip_{paper['id']}", use_container_width=True):
            skipped.add(int(paper["id"]))
            st.rerun()

# --- 4. results -------------------------------------------------------------------

with tab_results:
    st.markdown("**Final decisions** (yours where you gave one, else the AI's)")
    counts = papers["final_decision"].replace("", "not screened").value_counts()
    st.dataframe(counts.rename("papers"), use_container_width=False)

    checked = papers[(papers["label_source"] == "reviewer") & papers["ai_decision"].notna()]
    st.markdown("**How the AI compares with your decisions**")
    if checked.empty:
        st.caption("Review some AI-screened papers to see how often you agree with it.")
    else:
        decided = checked[checked["ai_decision"] != "unsure"]
        agree = (decided["ai_decision"] == decided["label"]).mean() if len(decided) else float("nan")
        missed = checked[(checked["ai_decision"] == "exclude") & (checked["label"] == "include")]
        m1, m2, m3 = st.columns(3)
        m1.metric("Papers you checked", len(checked))
        m2.metric("Agreement (AI include/exclude)", f"{agree:.0%}" if len(decided) else "–")
        m3.metric("AI excludes you included", len(missed))
        if len(missed):
            st.warning(
                "The AI excluded papers you included. Spot-check more of its excludes before relying on them, "
                "and consider sharpening the criteria they cite."
            )
            st.dataframe(missed[["title", "ai_rationale", "label_note"]], hide_index=True, use_container_width=True)

    st.download_button(
        "Download all decisions (CSV)",
        papers[EXPORT_COLUMNS].to_csv(index=False).encode(),
        file_name=f"{selected.replace(' ', '_')}_screening.csv",
        mime="text/csv",
        type="primary",
    )
    st.dataframe(papers[EXPORT_COLUMNS], hide_index=True, use_container_width=True)

# --- settings ----------------------------------------------------------------------

with tab_settings:
    st.markdown("**Criteria**")
    st.caption("Changing the criteria starts a fresh AI pass: earlier AI decisions were made under the old "
               "criteria and are kept, but no longer shown. Your own labels are unaffected.")
    with st.form("edit_criteria"):
        edited = criteria_form(f"edit_{project_id}", criteria)
        if st.form_submit_button("Save criteria"):
            try:
                store.update_criteria(project_id, edited)
                st.rerun()
            except ValueError as e:
                st.error(str(e))

    st.divider()
    st.markdown("**Delete project**")
    confirm = st.text_input(f"Type the project name ({selected}) to delete it with all its papers and decisions")
    if st.button("Delete project", disabled=confirm != selected):
        delete_project(store, project_id)
        st.session_state.pending_project = None
        st.rerun()
