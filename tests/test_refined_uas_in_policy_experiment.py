from gibbsq.experiments.evaluation.baselines_comparison import CORRECTED_POLICIES


def test_policy_comparison_includes_reflected_uas_tier3_baseline():
    reflected_entries = [entry for entry in CORRECTED_POLICIES if entry["name"] == "reflected_uas"]

    assert len(reflected_entries) == 1
    reflected_entry = reflected_entries[0]
    assert reflected_entry["tier"] == 3
    assert reflected_entry["requires_mu"] is True
    assert reflected_entry["alpha"] == 20.0
    assert reflected_entry["label"] == "Reflected UAS"


def test_policy_comparison_uses_clean_publication_tier3_candidates():
    tier3_names = [entry["name"] for entry in CORRECTED_POLICIES if entry["tier"] == 3]

    assert tier3_names == ["uas", "reflected_uas"]
