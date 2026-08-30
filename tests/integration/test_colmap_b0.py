import pytest
import sqlite3
import struct
import math
import json
import csv
from pathlib import Path

B0_WORKSPACE = Path(r"D:\SIH26158\colmap_workspace\zurich_mav_b0")
B0_DB_PATH = B0_WORKSPACE / "database.db"
B0_SPARSE_DIR = B0_WORKSPACE / "sparse" / "0"
B0_REPORTS_DIR = Path("outputs/reports/zurich_mav/b0")

EXPECTED_PARAMS = [
    893.3901081378665,
    898.3264861625313,
    951.1310042974931,
    555.1335007742958,
    -0.2805251302544365,
    0.1158064134556822,
    -0.0009843367849156311,
    0.0001584792476978901,
    -0.027021503433937236,
    0.0,
    0.0,
    0.0
]

def test_350_images_available():
    """1. Test that 350 images exist in the workspace images directory."""
    img_dir = B0_WORKSPACE / "images"
    assert img_dir.is_dir(), f"Workspace image directory missing: {img_dir}"
    imgs = list(img_dir.glob("*.jpg"))
    assert len(imgs) == 350, f"Expected 350 images, found {len(imgs)}"

def test_database_exists_and_readable():
    """2. Test that B0 database exists and is readable in read-only mode."""
    assert B0_DB_PATH.is_file(), f"Database file missing: {B0_DB_PATH}"
    conn = sqlite3.connect(f"{B0_DB_PATH.as_uri()}?mode=ro", uri=True)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
    table_cnt = cursor.fetchone()[0]
    conn.close()
    assert table_cnt > 0, "Database contains no tables."

def test_database_contains_350_images():
    """3. Test that database contains exactly 350 image records."""
    conn = sqlite3.connect(f"{B0_DB_PATH.as_uri()}?mode=ro", uri=True)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM images")
    cnt = cursor.fetchone()[0]
    conn.close()
    assert cnt == 350, f"Expected 350 image records in database, got {cnt}"

def test_camera_model_and_calibration():
    """4 & 5. Test that camera model is FULL_OPENCV and parameters match unrounded calibration."""
    conn = sqlite3.connect(f"{B0_DB_PATH.as_uri()}?mode=ro", uri=True)
    cursor = conn.cursor()
    cursor.execute("SELECT camera_id, model, width, height, params, prior_focal_length FROM cameras")
    row = cursor.fetchone()
    conn.close()
    assert row is not None, "No camera record found in database."
    cam_id, model_id, width, height, params_blob, prior_focal = row
    assert width == 1920 and height == 1080, f"Invalid dimensions: {width}x{height}"
    assert prior_focal == 1, "Prior focal length should be 1."

    num_doubles = len(params_blob) // 8
    assert num_doubles == 12, f"Expected 12 parameters for FULL_OPENCV, got {num_doubles}"
    actual_params = list(struct.unpack(f"<{num_doubles}d", params_blob))

    for i, (act, exp) in enumerate(zip(actual_params, EXPECTED_PARAMS)):
        assert math.isclose(act, exp, rel_tol=1e-5, abs_tol=1e-5), f"Param {i} mismatch: {act} vs {exp}"

def test_features_and_descriptors_exist_for_all_images():
    """6. Test that keypoints and descriptors exist for all 350 images."""
    conn = sqlite3.connect(f"{B0_DB_PATH.as_uri()}?mode=ro", uri=True)
    cursor = conn.cursor()
    cursor.execute("SELECT image_id, rows FROM keypoints")
    kp = dict(cursor.fetchall())
    cursor.execute("SELECT image_id, rows FROM descriptors")
    desc = dict(cursor.fetchall())
    conn.close()

    assert len(kp) == 350, f"Expected 350 keypoint records, got {len(kp)}"
    assert len(desc) == 350, f"Expected 350 descriptor records, got {len(desc)}"
    for img_id in range(1, 351):
        assert kp[img_id] > 3000, f"Image {img_id} has too few keypoints: {kp[img_id]}"
        assert kp[img_id] == desc[img_id], f"Mismatch for image {img_id}: {kp[img_id]} vs {desc[img_id]}"

def test_matches_and_two_view_geometries_exist():
    """7 & 8. Test that exhaustive matches and two-view geometric inliers exist."""
    conn = sqlite3.connect(f"{B0_DB_PATH.as_uri()}?mode=ro", uri=True)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM matches WHERE rows > 0")
    matches_cnt = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*), MIN(rows), MAX(rows) FROM two_view_geometries WHERE rows >= 15")
    tv_cnt, min_inl, max_inl = cursor.fetchone()
    conn.close()

    assert matches_cnt > 50000, f"Expected >50,000 matches, got {matches_cnt}"
    assert tv_cnt > 50000, f"Expected >50,000 verified two-view geometries, got {tv_cnt}"
    assert min_inl >= 15, f"Expected minimum inliers >= 15, got {min_inl}"

def test_sparse_model_files_exist():
    """9. Test that binary sparse model files exist in sparse/0/."""
    assert (B0_SPARSE_DIR / "cameras.bin").is_file(), "cameras.bin missing"
    assert (B0_SPARSE_DIR / "images.bin").is_file(), "images.bin missing"
    assert (B0_SPARSE_DIR / "points3D.bin").is_file(), "points3D.bin missing"

def test_reconstruction_summary_schema_and_metrics():
    """10 & 11. Test that reconstruction_summary.json is valid and registered count is consistent."""
    summary_path = B0_REPORTS_DIR / "reconstruction_summary.json"
    assert summary_path.is_file(), f"reconstruction_summary.json missing: {summary_path}"
    with open(summary_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["status"] == "PASS"
    assert data["image_metrics"]["total_images"] == 350
    assert data["image_metrics"]["registered_images"] == 350
    assert data["image_metrics"]["registration_percentage"] == 100.0
    assert data["sparse_3d_metrics"]["total_sparse_3d_points"] > 40000
    assert data["sparse_3d_metrics"]["reprojection_error"]["mean_px"] < 1.5

def test_camera_poses_schema_and_integrity():
    """12. Test that camera_poses_colmap.csv has valid rows and non-null coordinates."""
    csv_path = B0_REPORTS_DIR / "camera_poses_colmap.csv"
    assert csv_path.is_file(), f"camera_poses_colmap.csv missing: {csv_path}"
    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 350, f"Expected 350 pose rows, got {len(rows)}"
    gt_count = sum(1 for r in rows if r["ground_truth_available"].lower() == "true")
    assert gt_count == 12, f"Expected 12 ground truth keyframe flags, got {gt_count}"

    for r in rows:
        assert r["registered"].lower() == "true"
        x = float(r["camera_center_x"])
        y = float(r["camera_center_y"])
        z = float(r["camera_center_z"])
        assert not math.isnan(x) and not math.isnan(y) and not math.isnan(z)
