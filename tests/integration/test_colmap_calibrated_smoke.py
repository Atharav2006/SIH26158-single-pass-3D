import pytest
import sqlite3
import struct
import math
from pathlib import Path

from src.reconstruction.colmap_wrapper import find_colmap_executable

CALIBRATED_DB_PATH = Path(r"D:\SIH26158\colmap_workspace\smoke_test_calibrated\database.db")
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

def test_colmap_executable_available():
    """1. Test that the COLMAP binary exists and is executable."""
    exe = find_colmap_executable()
    assert exe.is_file(), f"COLMAP executable not found: {exe}"

def test_database_exists_and_readable():
    """2. Test that calibrated smoke test database exists and is readable in read-only mode."""
    assert CALIBRATED_DB_PATH.is_file(), f"Database file missing: {CALIBRATED_DB_PATH}"
    conn = sqlite3.connect(f"{CALIBRATED_DB_PATH.as_uri()}?mode=ro", uri=True)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
    table_cnt = cursor.fetchone()[0]
    conn.close()
    assert table_cnt > 0, "Database contains no tables."

def test_ten_images_imported():
    """3. Test that exactly 10 representative images are present."""
    conn = sqlite3.connect(f"{CALIBRATED_DB_PATH.as_uri()}?mode=ro", uri=True)
    cursor = conn.cursor()
    cursor.execute("SELECT image_id, name FROM images ORDER BY image_id")
    rows = cursor.fetchall()
    conn.close()
    assert len(rows) == 10, f"Expected 10 images, found {len(rows)}"
    expected_names = [
        "00001.jpg", "00035.jpg", "00070.jpg", "00105.jpg", "00140.jpg",
        "00175.jpg", "00210.jpg", "00245.jpg", "00280.jpg", "00350.jpg"
    ]
    actual_names = [r[1] for r in rows]
    assert actual_names == expected_names, f"Image names mismatch: {actual_names}"

def test_camera_model_and_dimensions():
    """4 & 9. Test camera model, dimensions, and prior focal length."""
    conn = sqlite3.connect(f"{CALIBRATED_DB_PATH.as_uri()}?mode=ro", uri=True)
    cursor = conn.cursor()
    cursor.execute("SELECT camera_id, model, width, height, prior_focal_length FROM cameras")
    row = cursor.fetchone()
    conn.close()
    assert row is not None, "No camera record found in database."
    cam_id, model_id, width, height, prior_focal = row
    # In COLMAP, FULL_OPENCV model ID is 6 (or 8 depending on schema build, verified present with 12 params)
    assert width == 1920, f"Expected width 1920, got {width}"
    assert height == 1080, f"Expected height 1080, got {height}"
    assert prior_focal == 1, "Prior focal length flag should be enabled."

def test_camera_calibration_parameters():
    """5 & 10. Test that camera parameters match exact unrounded calibration and not 2304 px."""
    conn = sqlite3.connect(f"{CALIBRATED_DB_PATH.as_uri()}?mode=ro", uri=True)
    cursor = conn.cursor()
    cursor.execute("SELECT params FROM cameras")
    params_blob = cursor.fetchone()[0]
    conn.close()

    num_doubles = len(params_blob) // 8
    assert num_doubles == 12, f"Expected 12 parameters for FULL_OPENCV, got {num_doubles}"
    actual_params = list(struct.unpack(f"<{num_doubles}d", params_blob))

    # Assert fx and fy are close to ~893-898 and NOT 2304 px
    assert abs(actual_params[0] - 2304.0) > 1000.0, "Accidental 2304 px focal length detected!"
    assert abs(actual_params[1] - 2304.0) > 1000.0, "Accidental 2304 px focal length detected!"

    for i, (act, exp) in enumerate(zip(actual_params, EXPECTED_PARAMS)):
        assert math.isclose(act, exp, rel_tol=1e-6, abs_tol=1e-6), (
            f"Param {i} mismatch: actual {act} vs expected {exp}"
        )

def test_descriptors_and_keypoints_exist():
    """6. Test that all images have extracted keypoints and descriptors."""
    conn = sqlite3.connect(f"{CALIBRATED_DB_PATH.as_uri()}?mode=ro", uri=True)
    cursor = conn.cursor()
    cursor.execute("SELECT image_id, rows FROM keypoints")
    kp = dict(cursor.fetchall())
    cursor.execute("SELECT image_id, rows FROM descriptors")
    desc = dict(cursor.fetchall())
    conn.close()

    assert len(kp) == 10, f"Expected 10 keypoint records, got {len(kp)}"
    assert len(desc) == 10, f"Expected 10 descriptor records, got {len(desc)}"
    for i in range(1, 11):
        assert kp[i] > 5000, f"Image {i} has too few keypoints: {kp[i]}"
        assert desc[i] == kp[i], f"Descriptor count mismatch for image {i}: {desc[i]} vs {kp[i]}"

def test_matches_and_geometric_verification():
    """7 & 8. Test that matches and two-view geometric inliers exist and are well-connected."""
    conn = sqlite3.connect(f"{CALIBRATED_DB_PATH.as_uri()}?mode=ro", uri=True)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM matches WHERE rows > 0")
    matches_cnt = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*), MIN(rows), MAX(rows), AVG(rows) FROM two_view_geometries WHERE rows >= 15")
    verified_cnt, min_inl, max_inl, avg_inl = cursor.fetchone()
    conn.close()

    assert matches_cnt == 45, f"Expected 45 matched pairs, got {matches_cnt}"
    assert verified_cnt == 45, f"Expected 45 verified pairs with >=15 inliers, got {verified_cnt}"
    assert min_inl >= 30, f"Expected min inliers >= 30, got {min_inl}"
    assert max_inl > 1500, f"Expected max inliers > 1500, got {max_inl}"
