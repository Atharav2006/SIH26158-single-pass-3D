"""
Unit tests for B5.1.2 Forensic Depth-Gauge Consistency Audit.
"""

from pathlib import Path
import json
import pytest
import numpy as np

def test_b5_1_2_forensic_artifacts_exist():
    audit_file = Path("outputs/reports/zurich_mav/b5/b5_1_2_forensic_audit.json")
    recomputed_file = Path("outputs/reports/zurich_mav/b5/b5_1_2_pairwise_recomputed.json")

    assert audit_file.exists(), "b5_1_2_forensic_audit.json must exist"
    assert recomputed_file.exists(), "b5_1_2_pairwise_recomputed.json must exist"

def test_b5_1_2_recomputed_statistics_consistency():
    with open("outputs/reports/zurich_mav/b5/b5_1_2_pairwise_recomputed.json") as f:
        data = json.load(f)

    stats = data["summary_statistics"]
    records = data["pairwise_records"]

    a_vals = [r["fitted_a"] for r in records]
    b_vals = [r["fitted_b"] for r in records]
    r_vals = [r["pearson_r"] for r in records]

    # Verify that summary statistics are dynamically consistent with pairwise records
    assert np.isclose(stats["actual_mean_a"], np.mean(a_vals), atol=1e-5)
    assert np.isclose(stats["actual_std_a"], np.std(a_vals), atol=1e-5)
    assert np.isclose(stats["actual_median_a"], np.median(a_vals), atol=1e-5)
    assert np.isclose(stats["actual_min_a"], np.min(a_vals), atol=1e-5)
    assert np.isclose(stats["actual_max_a"], np.max(a_vals), atol=1e-5)
    assert np.isclose(stats["actual_b_range"][0], np.min(b_vals), atol=1e-5)
    assert np.isclose(stats["actual_b_range"][1], np.max(b_vals), atol=1e-5)
    assert np.isclose(stats["actual_correlation_range"][0], np.min(r_vals), atol=1e-5)
    assert np.isclose(stats["actual_correlation_range"][1], np.max(r_vals), atol=1e-5)

def test_b5_1_2_scale_check_consistency():
    with open("outputs/reports/zurich_mav/b5/b5_1_2_pairwise_recomputed.json") as f:
        data = json.load(f)

    records = data["pairwise_records"]
    for r in records:
        pred_med_j = r["fitted_a"] * r["median_i"] + r["fitted_b"]
        actual_med_j = r["median_j"]
        # Absolute discrepancy must be reasonable given non-linear spatial distribution
        rel_diff = abs(pred_med_j - actual_med_j) / actual_med_j
        assert rel_diff < 0.35, f"Pair {r['pair']} median mismatch too large: {rel_diff*100:.2f}%"

def test_b5_1_2_forensic_classification():
    with open("outputs/reports/zurich_mav/b5/b5_1_2_forensic_audit.json") as f:
        audit = json.load(f)

    assert audit["b5_1_2_status"] == "PASS"
    assert audit["gauge_classification"] == "GAUGE_PARTIALLY_STABLE"
    assert audit["depth_representation_used_for_pairwise_fit"] == "raw_midas_inverse_depth_D_inv"
    assert "discrepancy_1_hardcoded_narrative_mean" in audit["root_cause_of_previous_inconsistency"]
    assert "discrepancy_2_representation_mismatch" in audit["root_cause_of_previous_inconsistency"]
