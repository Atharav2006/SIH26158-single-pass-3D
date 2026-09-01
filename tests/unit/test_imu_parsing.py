import pytest
import csv
import math
import numpy as np
from pathlib import Path

IMU_CSV = Path("outputs/reports/zurich_mav/imu.csv")

def test_imu_csv_schema_and_numerical_validity():
    """Test that imu.csv exists and contains valid numeric fields."""
    assert IMU_CSV.is_file(), f"Missing {IMU_CSV}"
    with open(IMU_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        expected_cols = {"timestamp_seconds", "accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"}
        assert set(reader.fieldnames) == expected_cols

        count = 0
        for r in reader:
            count += 1
            ts = float(r["timestamp_seconds"])
            ax = float(r["accel_x"])
            ay = float(r["accel_y"])
            az = float(r["accel_z"])
            gx = float(r["gyro_x"])
            gy = float(r["gyro_y"])
            gz = float(r["gyro_z"])

            assert not math.isnan(ts) and ts > 0.0
            assert -160.0 < ax < 160.0
            assert -160.0 < ay < 160.0
            assert -160.0 < az < 160.0
            assert -35.0 < gx < 35.0
            assert -35.0 < gy < 35.0
            assert -35.0 < gz < 35.0

        assert count == 27050, f"Expected 27,050 records, found {count}"

def test_imu_stationary_gravity_consistency():
    """Test that stationary acceleration magnitude is consistent with Earth gravity."""
    with open(IMU_CSV, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Takeoff pad dwell (first 10 records, t in [7.0, 8.0])
    stat_amags = []
    for r in rows:
        t = float(r["timestamp_seconds"])
        if 7.0 <= t <= 8.0:
            ax = float(r["accel_x"])
            ay = float(r["accel_y"])
            az = float(r["accel_z"])
            stat_amags.append(math.sqrt(ax**2 + ay**2 + az**2))

    assert len(stat_amags) > 5
    mean_g = float(np.mean(stat_amags))
    assert 8.5 < mean_g < 10.5, f"Gravity magnitude out of physical bounds: {mean_g} m/s^2"

def test_body_frame_flu_conversion_invariant():
    """Test that converting native FRD IMU vectors to FLU maintains vector norms."""
    a_native = np.array([0.24, -0.48, -9.61])
    # Mapping FRD -> FLU: [x, -y, -z]
    a_flu = np.array([a_native[0], -a_native[1], -a_native[2]])

    assert math.isclose(np.linalg.norm(a_native), np.linalg.norm(a_flu), rel_tol=1e-9)
    assert a_flu[2] > 0.0, "Upward reaction force in FLU must be positive"
