import pytest
import csv
import math
import numpy as np
from pathlib import Path

IMU_CSV = Path("outputs/reports/zurich_mav/imu.csv")
IMAGES_CSV = Path("outputs/reports/zurich_mav/images.csv")

def test_imu_strict_timestamp_monotonicity():
    """Test that all 27,050 IMU timestamps are strictly increasing."""
    ts_list = []
    with open(IMU_CSV, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            ts_list.append(float(r["timestamp_seconds"]))

    ts = np.array(ts_list)
    dt = np.diff(ts)

    assert np.all(dt > 0.0), f"Found non-positive timestamp deltas: min dt = {np.min(dt)}"
    assert np.sum(dt <= 0.0) == 0

def test_imu_sampling_frequency():
    """Test that IMU nominal sampling frequency is ~10 Hz (mean interval ~100 ms)."""
    ts_list = []
    with open(IMU_CSV, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            ts_list.append(float(r["timestamp_seconds"]))

    dt = np.diff(ts_list) * 1000.0  # in ms
    mean_dt = float(np.mean(dt))
    median_dt = float(np.median(dt))

    assert 90.0 < mean_dt < 110.0, f"Mean dt out of range: {mean_dt} ms"
    assert math.isclose(median_dt, 100.0, abs_tol=1.0), f"Median dt out of range: {median_dt} ms"

def test_image_imu_nearest_neighbor_timing_bound():
    """Test that image frames at 30 Hz have nearest-neighbor IMU samples within 100 ms."""
    imu_ts = []
    with open(IMU_CSV, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            imu_ts.append(float(r["timestamp_seconds"]))
    imu_arr = np.array(imu_ts)

    img_ts = []
    with open(IMAGES_CSV, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            img_ts.append(float(r["timestamp_seconds"]))

    max_delta = 0.0
    for t in img_ts:
        idx = np.argmin(np.abs(imu_arr - t))
        d = abs(imu_arr[idx] - t)
        if d > max_delta:
            max_delta = d

    assert max_delta < 0.10, f"Maximum nearest IMU delta exceeded 100 ms: {max_delta*1000:.2f} ms"
