import pytest
import numpy as np
import math

from src.metrics.alignment import umeyama_alignment
from src.metrics.trajectory_metrics import compute_ate

def test_fixed_transform_determinism():
    """Test that applying a frozen Sim(3) transform is 100% deterministic and matches umeyama output."""
    np.random.seed(42)
    src = np.random.randn(10, 3) * 5.0
    dst = np.random.randn(10, 3) * 2.0

    s, R, t, aligned = umeyama_alignment(src, dst, with_scale=True)

    # Apply frozen transform directly
    frozen_aligned = s * (src @ R.T) + t
    assert np.allclose(aligned, frozen_aligned, atol=1e-12)

def test_leave_one_out_perfect_synthetic():
    """Test that leave-one-out cross validation yields zero error on a perfect synthetic similarity transform."""
    np.random.seed(99)
    src = np.random.randn(12, 3) * 10.0

    s_true = 2.5
    theta = math.radians(45.0)
    R_true = np.array([
        [math.cos(theta), -math.sin(theta), 0.0],
        [math.sin(theta),  math.cos(theta), 0.0],
        [0.0,              0.0,             1.0]
    ])
    t_true = np.array([5.0, -3.0, 8.0])

    dst = s_true * (src @ R_true.T) + t_true

    held_out_errors = []
    n = len(src)
    for i in range(n):
        train_mask = np.ones(n, dtype=bool)
        train_mask[i] = False

        s_est, R_est, t_est, _ = umeyama_alignment(src[train_mask], dst[train_mask], with_scale=True)
        pred = s_est * (src[i:i+1] @ R_est.T) + t_est
        err = np.linalg.norm(pred - dst[i:i+1])
        held_out_errors.append(err)

    assert all(math.isclose(e, 0.0, abs_tol=1e-6) for e in held_out_errors)
