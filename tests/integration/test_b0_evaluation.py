import pytest
import csv
import json
import math
from pathlib import Path

EVAL_DIR = Path("outputs/reports/zurich_mav/b0")
EVAL_CSV = EVAL_DIR / "b0_gt_evaluation_pairs.csv"
EVAL_JSON = EVAL_DIR / "b0_evaluation.json"

def test_evaluation_pairs_count_and_imgids():
    """Test that exactly 12 keyframe pairs are extracted with step 30."""
    assert EVAL_CSV.is_file(), f"Missing evaluation pairs CSV: {EVAL_CSV}"
    with open(EVAL_CSV, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 12, f"Expected 12 ground truth pairs, found {len(rows)}"
    expected_imgids = [1, 31, 61, 91, 121, 151, 181, 211, 241, 271, 301, 331]
    actual_imgids = [int(r["imgid"]) for r in rows]
    assert actual_imgids == expected_imgids, f"imgid mismatch: {actual_imgids}"

def test_timestamps_monotonic():
    """Test that ground truth timestamps are strictly monotonically increasing."""
    with open(EVAL_CSV, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    ts_list = [float(r["gt_timestamp"]) for r in rows]
    for i in range(len(ts_list) - 1):
        assert ts_list[i + 1] > ts_list[i], f"Non-monotonic timestamps: {ts_list[i]} -> {ts_list[i+1]}"

def test_no_unmatched_frames_in_evaluation_csv():
    """Test that every row in b0_gt_evaluation_pairs.csv has non-empty coordinates."""
    with open(EVAL_CSV, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for r in rows:
        assert r["gt_x"] and r["gt_y"] and r["gt_z"]
        assert r["colmap_x"] and r["colmap_y"] and r["colmap_z"]

def test_evaluation_json_metrics_validity():
    """Test that b0_evaluation.json contains valid ATE, RPE, and scale metrics."""
    assert EVAL_JSON.is_file(), f"Missing evaluation JSON: {EVAL_JSON}"
    with open(EVAL_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["evaluation_subset"]["evaluated_ground_truth_keyframes"] == 12
    ate = data["ate_metrics_m"]
    assert 0.0 < ate["rmse_m"] < 0.1, f"ATE RMSE out of expected range: {ate['rmse_m']}"
    assert ate["mean_m"] > 0.0

    rpe = data["rpe_metrics"]
    assert 0.0 < rpe["translational_rpe"]["rmse_m"] < 0.1
    assert 0.0 < rpe["rotational_rpe"]["rmse_deg"] < 10.0

    scale = data["sim3_alignment"]["scale_factor_s"]
    assert 0.05 < scale < 0.5, f"Unexpected scale factor: {scale}"

def test_visualizations_exist_and_non_empty():
    """Test that all three evaluation visualization PNGs exist on disk."""
    topdown_png = EVAL_DIR / "b0_gt_vs_colmap_topdown.png"
    pos_err_png = EVAL_DIR / "b0_position_error.png"
    traj_3d_png = EVAL_DIR / "b0_trajectory_comparison_3d.png"

    for p in [topdown_png, pos_err_png, traj_3d_png]:
        assert p.is_file(), f"Missing visualization artifact: {p}"
        assert p.stat().st_size > 5000, f"Visualization file too small: {p} ({p.stat().st_size} bytes)"
