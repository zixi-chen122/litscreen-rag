from src import datasets


def test_list_datasets_includes_sample_and_only_complete_prepared_dirs(tmp_path, monkeypatch):
    complete = tmp_path / "Xu_2022"
    complete.mkdir()
    for name in ("seed.csv", "test.csv", "criteria.json"):
        (complete / name).write_text("")
    (tmp_path / "Half_Prepared").mkdir()
    (tmp_path / "Half_Prepared" / "seed.csv").write_text("")
    monkeypatch.setattr(datasets, "SYNERGY_PREPARED_DIR", tmp_path)

    assert datasets.list_datasets() == ["sample", "Xu_2022"]
