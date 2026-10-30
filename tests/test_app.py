"""
Smoke tests for the Streamlit app: it renders with no projects and with a
project of papers, and the password gate holds. No model calls are made.
"""
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from src.projects import ReviewStore

CRITERIA = {"review_topic": "RPM in heart failure", "inclusion_criteria": ["Adults"], "exclusion_criteria": ["Children"]}


@pytest.fixture
def db(tmp_path, monkeypatch):
    path = str(tmp_path / "app.db")
    monkeypatch.setattr("src.config.DB_PATH", path)
    monkeypatch.delenv("APP_PASSWORD", raising=False)
    st.cache_resource.clear()  # the app caches its store across runs
    return ReviewStore(path)


def test_app_renders_without_projects(db):
    at = AppTest.from_file("app.py", default_timeout=30).run()
    assert not at.exception
    assert any("Create a project" in i.value for i in at.info)


def test_app_renders_project_with_review_queue(db):
    project = db.create_project("HF review", CRITERIA)
    db.add_papers(project, [{"title": "Labeled", "label": "include"}, {"title": "Unscreened paper"}])

    at = AppTest.from_file("app.py", default_timeout=30).run()

    assert not at.exception
    assert at.title[0].value == "HF review"
    assert [m.value for m in at.metric][:2] == ["2", "1"]
    queue = next(s for s in at.selectbox if s.label == "Queue")
    queue.set_value("All unscreened").run()
    assert any(h.value == "Unscreened paper" for h in at.subheader)


def test_app_requires_password_when_set(db, monkeypatch):
    monkeypatch.setenv("APP_PASSWORD", "secret")
    at = AppTest.from_file("app.py", default_timeout=30).run()
    assert not at.sidebar.selectbox
    at.text_input[0].input("wrong").run()
    assert at.error and not at.sidebar.selectbox
    at.text_input[0].input("secret").run()
    assert at.sidebar.selectbox and not at.exception
