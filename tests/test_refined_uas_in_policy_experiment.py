from experiments.evaluation.baselines_comparison import CORRECTED_POLICIES


def test_policy_comparison_includes_refined_uas_tier3_baseline():
    refined_entries = [entry for entry in CORRECTED_POLICIES if entry["name"] == "refined_uas"]

    assert len(refined_entries) == 1
    refined_entry = refined_entries[0]
    assert refined_entry["tier"] == 3
    assert refined_entry["requires_mu"] is True
    assert refined_entry["alpha"] == 20.0
    assert "Refined UAS" in refined_entry["label"]


def test_policy_comparison_uses_multiple_tier3_candidates():
    tier3_names = [entry["name"] for entry in CORRECTED_POLICIES if entry["tier"] == 3]

    assert "uas" in tier3_names
    assert "refined_uas" in tier3_names
