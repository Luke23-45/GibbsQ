from experiments.evaluation.baselines_comparison import CORRECTED_POLICIES


def test_policy_comparison_includes_calibrated_uas_tier3_baseline():
    calibrated_entries = [entry for entry in CORRECTED_POLICIES if entry["name"] == "calibrated_uas"]

    assert len(calibrated_entries) == 1
    calibrated_entry = calibrated_entries[0]
    assert calibrated_entry["tier"] == 3
    assert calibrated_entry["requires_mu"] is True
    assert calibrated_entry["alpha"] == 20.0
    assert calibrated_entry["label"] == "Calibrated UAS"


def test_policy_comparison_uses_clean_publication_tier3_candidates():
    tier3_names = [entry["name"] for entry in CORRECTED_POLICIES if entry["tier"] == 3]

    assert tier3_names == ["uas", "calibrated_uas"]
