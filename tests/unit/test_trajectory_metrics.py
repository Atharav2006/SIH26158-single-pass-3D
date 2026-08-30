import pytest
import numpy as np
import math

from src.metrics.trajectory_metrics import (
    compute_ate,
    compute_rpe,
    compute_trajectory_statistics
)

def test_ate_zero_error_for_identical_trajectories():
    """Test that identical trajectories produce zero ATE."""
    pts = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 2.0, 0.5],
        [2.0, 4.0, 1.0],
        [3.0, 6.0, 1.5]
    ])
    ate = compute_ate(pts, pts)
    assert ate["rmse_m"] == 0.0
    assert ate["mean_m"] == 0.0
    assert ate["max_m"] == 0.0
    assert all(e == 0.0 for e in ate["per_frame_errors_m"])

def test_ate_known_offset():
    """Test ATE with known constant displacement vector."""
    gt = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    est = gt + np.array([0.3, 0.4, 0.0])  # norm is 0.5
    ate = compute_ate(est, gt)
    assert math.isclose(ate["rmse_m"], 0.5, rel_tol=1e-5)
    assert math.isclose(ate["mean_m"], 0.5, rel_tol=1e-5)
    assert math.isclose(ate["median_m"], 0.5, rel_tol=1e-5)

def test_rpe_zero_for_identical_relative_motions():
    """Test that identical relative motions produce zero translational and rotational RPE."""
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    rots = [np.eye(3), np.eye(3), np.eye(3)]
    rpe = compute_rpe(pts, pts, rots, rots, delta=1)
    assert rpe["translational_rpe"]["rmse_m"] == 0.0
    assert rpe["rotational_rpe"]["rmse_deg"] == 0.0

def test_trajectory_statistics_length_and_scale():
    """Test trajectory path length, scale error, and endpoint error calculations."""
    gt = np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]])  # length 5.0
    raw = np.array([[0.0, 0.0, 0.0], [1.5, 2.0, 0.0]])  # length 2.5
    aligned = np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]])  # scale factor 2.0
    stats = compute_trajectory_statistics(raw, aligned, gt, scale_factor=2.0)

    assert stats["ground_truth_trajectory_length_m"] == 5.0
    assert stats["raw_colmap_trajectory_length_units"] == 2.5
    assert stats["aligned_colmap_trajectory_length_m"] == 5.0
    assert stats["scale_factor_s"] == 2.0
    assert stats["scale_error_percent"] == 100.0
    assert stats["endpoint_error_m"] == 0.0
