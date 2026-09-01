import pytest
import json
from pathlib import Path

OUT_DIR = Path("outputs/reports/zurich_mav/b2")
FRAME_JSON = OUT_DIR / "imu_frame_validation.json"
SANITY_JSON = OUT_DIR / "imu_preintegration_sanity.json"

def test_imu_frame_validation_json_schema():
    """Test that imu_frame_validation.json exists and validates frame transformations."""
    assert FRAME_JSON.is_file(), f"Missing {FRAME_JSON}"
    with open(FRAME_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["status"] == "PASS"
    semantics = data["verified_sensor_semantics"]
    assert semantics["native_sensor_frame"] == "Forward-Right-Down (FRD / NED body)"
    assert semantics["internal_target_frame"] == "Forward-Left-Up (FLU robotic body)"

    drift = data["stationary_gyro_drift_analysis"]["naive_integrated_orientation_drift"]
    assert drift["after_60_seconds_deg"] > 100.0

def test_imu_preintegration_sanity_json():
    """Test that imu_preintegration_sanity.json contains valid non-empty real-data sanity intervals."""
    assert SANITY_JSON.is_file(), f"Missing {SANITY_JSON}"
    with open(SANITY_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "PASS" in data["preintegration_status"]
    intervals = data["sanity_intervals"]
    assert len(intervals) >= 3

    for seg in intervals:
        assert seg["imu_sample_count"] > 0
        assert seg["integration_duration_s"] > 0.0
        assert seg["delta_rotation_angle_deg"] >= 0.0
        assert seg["delta_velocity_magnitude_m_s"] >= 0.0
        assert seg["delta_position_magnitude_m"] >= 0.0
