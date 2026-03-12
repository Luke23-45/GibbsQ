from experiments.evaluation import policy_comparison as pc


def test_iter_with_progress_fallback_without_tqdm(monkeypatch):
    monkeypatch.setattr(pc, "tqdm", None)
    items = [1, 2, 3]
    wrapped = pc._iter_with_progress(items, desc="x", total=3)
    assert list(wrapped) == items


def test_iter_with_progress_wraps_when_tqdm_available(monkeypatch):
    calls = {}

    def fake_tqdm(items, **kwargs):
        calls["kwargs"] = kwargs
        return items

    monkeypatch.setattr(pc, "tqdm", fake_tqdm)
    items = ["a", "b"]
    wrapped = pc._iter_with_progress(items, desc="Policies", total=2)
    assert list(wrapped) == items
    assert calls["kwargs"]["desc"] == "Policies"
    assert calls["kwargs"]["total"] == 2
