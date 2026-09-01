import pytest
import json
import csv
import math
from pathlib import Path

OUT_DIR = Path("outputs/reports/zurich_mav/b2")
CORR_CSV = OUT_DIR / "image_gps_imu_correspondence.csv"
JSON_PATH = OUT_DIR / "imu_quality.json"
ACC_PNG = OUT_DIR / "imu_acceleration.png"
GYRO_PNG = OUT_DIR / "imu_angular_velocity.png"
SAMP_PNG = OUT_DIR / "imu_sampling_interval.png"

def test_image_gps_imu_correspondence_csv():
    """Test that image_gps_imu_correspondence.csv exists and has 350 valid rows."""
    assert CORR_CSV.is_file(), f"Missing {CORR_CSV}"
    with open(CORR_CSV, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 350
    for r in rows:
        assert r["imgid"]
        assert float(r["image_timestamp"]) > 0.0
        assert float(r["gps_timestamp"]) > 0.0
        assert float(r["nearest_imu_timestamp"]) > 0.0
        assert float(r["nearest_imu_delta"]) < 0.10

def test_imu_quality_json_schema():
    """Test that imu_quality.json contains valid characterization metrics."""
    assert JSON_PATH.is_file(), f"Missing {JSON_PATH}"
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["status"] == "PASS"
    assert data["sampling_characteristics"]["total_imu_records"] == 27050
    assert 9.0 < data["sampling_characteristics"]["nominal_frequency_hz"] < 11.0
    assert data["sampling_characteristics"]["dropped_or_duplicate_timestamps_count"] == 0

    stat = data["stationary_characterization"]
    assert 8.5 < stat["stationary_acceleration_magnitude"] < 10.5
    assert math.isclose(stat["nominal_gravity"], 9.80665, abs_tol=1e-4)
    assert abs(stat["gravity_magnitude_difference"]) < 1.5
    assert len(stat["stationary_mean_acceleration"]) == 3
    assert len(stat["stationary_mean_gyro"]) == 3

def test_imu_visualizations_exist():
    """Test that all three IMU visualization PNGs exist on disk."""
    for p in [ACC_PNG, GYRO_PNG, SAMP_PNG]:
        assert p.is_file(), f"Missing {p}"
        assert p.stat().st_size > 5000, f"File too small: {p}"
