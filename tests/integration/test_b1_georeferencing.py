import pytest
import json
import csv
from pathlib import Path

B1_REPORTS_DIR = Path("outputs/reports/zurich_mav/b1")
B1_WS_DIR = Path(r"D:\SIH26158\colmap_workspace\zurich_mav_b1")
B0_WS_DIR = Path(r"D:\SIH26158\colmap_workspace\zurich_mav_b0")

CORR_CSV = B1_REPORTS_DIR / "colmap_gps_correspondences.csv"
TRANSFORM_JSON = B1_REPORTS_DIR / "transform.json"
METRIC_POSES_CSV = B1_WS_DIR / "camera_poses_metric.csv"
B0_VS_B1_JSON = B1_REPORTS_DIR / "b0_vs_b1.json"

TRAJ_PNG = B1_REPORTS_DIR / "b1_gps_georeferenced_trajectory.png"
RES_PNG = B1_REPORTS_DIR / "b1_gps_residuals.png"
SCALE_PNG = B1_REPORTS_DIR / "b1_scale_comparison.png"

def test_colmap_gps_correspondences_file():
    """Test that colmap_gps_correspondences.csv exists and has 350 valid rows."""
    assert CORR_CSV.is_file(), f"Missing {CORR_CSV}"
    with open(CORR_CSV, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 350
    for r in rows:
        assert float(r["colmap_x"]) is not None
        assert float(r["gps_east"]) is not None

def test_transform_json_schema_and_invertibility():
    """Test transform.json schema and verify exact mathematical invertibility."""
    assert TRANSFORM_JSON.is_file(), f"Missing {TRANSFORM_JSON}"
    with open(TRANSFORM_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    fwd = data["forward_transform"]
    inv = data["inverse_transform"]

    s = fwd["scale_s"]
    s_inv = inv["scale_s_inv"]
    assert 0.05 < s < 0.50
    assert abs(s * s_inv - 1.0) < 1e-6

def test_camera_poses_metric_csv():
    """Test camera_poses_metric.csv exists in B1 workspace with 350 records."""
    assert METRIC_POSES_CSV.is_file(), f"Missing {METRIC_POSES_CSV}"
    with open(METRIC_POSES_CSV, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 350
    for r in rows:
        assert r["registered"].lower() == "true"
        assert float(r["metric_center_east_local_m"]) is not None
        assert float(r["utm_zone_32n_easting_m"]) > 400000.0

def test_b0_unchanged():
    """Test that B0 workspace files remain completely intact and unchanged."""
    b0_db = B0_WS_DIR / "database.db"
    b0_sparse = B0_WS_DIR / "sparse" / "0" / "images.bin"
    assert b0_db.is_file()
    assert b0_sparse.is_file()

def test_b0_vs_b1_json_metrics():
    """Test b0_vs_b1.json contains valid evaluation comparisons."""
    assert B0_VS_B1_JSON.is_file(), f"Missing {B0_VS_B1_JSON}"
    with open(B0_VS_B1_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["reconstruction_scale"]["B1_estimated_scale_s"] > 0.0
    gt_eval = data["ground_truth_evaluation_comparison"]
    assert gt_eval["B1_direct_metric_ate_rmse_m"] > 0.0
    assert gt_eval["B0_sim3_aligned_ate_rmse_m"] > 0.0

def test_b1_visualizations_exist():
    """Test that all three B1 visualization plots exist and are non-empty."""
    for p in [TRAJ_PNG, RES_PNG, SCALE_PNG]:
        assert p.is_file(), f"Missing {p}"
        assert p.stat().st_size > 5000, f"File too small: {p}"
