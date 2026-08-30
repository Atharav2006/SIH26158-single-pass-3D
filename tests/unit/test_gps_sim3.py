import pytest
import numpy as np
import math

from src.metrics.alignment import (
    umeyama_alignment,
    rotation_matrix_to_quaternion,
    quaternion_to_rotation_matrix
)

def test_gps_sim3_identity_recovery():
    """Test that identical point clouds yield exact identity Sim(3) transform."""
    np.random.seed(42)
    pts = np.random.randn(30, 3) * 10.0
    s, R, t, aligned = umeyama_alignment(pts, pts, with_scale=True)

    assert math.isclose(s, 1.0, rel_tol=1e-7)
    assert np.allclose(R, np.eye(3), atol=1e-7)
    assert np.allclose(t, np.zeros(3), atol=1e-7)
    assert np.allclose(aligned, pts, atol=1e-7)

def test_gps_sim3_known_synthetic_transform():
    """Test exact recovery of scale, rotation, and translation on synthetic trajectory."""
    np.random.seed(101)
    src = np.random.randn(50, 3) * 15.0

    s_true = 0.14083
    theta = math.radians(72.5)
    R_true = np.array([
        [math.cos(theta), 0.0, math.sin(theta)],
        [0.0,             1.0, 0.0],
        [-math.sin(theta), 0.0, math.cos(theta)]
    ])
    t_true = np.array([-14.2, 55.8, 120.3])

    dst = s_true * (src @ R_true.T) + t_true

    s_est, R_est, t_est, aligned = umeyama_alignment(src, dst, with_scale=True)

    assert math.isclose(s_est, s_true, rel_tol=1e-5)
    assert np.allclose(R_est, R_true, atol=1e-5)
    assert np.allclose(t_est, t_true, atol=1e-5)
    assert np.allclose(aligned, dst, atol=1e-5)

def test_sim3_transform_invertibility():
    """Test that forward and inverse Sim(3) transformations roundtrip perfectly."""
    s = 0.140830
    theta = math.radians(33.0)
    R = np.array([
        [math.cos(theta), -math.sin(theta), 0.0],
        [math.sin(theta),  math.cos(theta), 0.0],
        [0.0,              0.0,             1.0]
    ])
    t = np.array([10.5, -4.2, 8.1])

    # Inverse definitions
    s_inv = 1.0 / s
    R_inv = R.T
    t_inv = - (s_inv * (R_inv @ t))

    def forward(p):
        return s * (p @ R.T) + t

    def inverse(p):
        return s_inv * (p @ R_inv.T) + t_inv

    np.random.seed(55)
    pts = np.random.randn(25, 3) * 50.0

    pts_fwd = forward(pts)
    pts_roundtrip = inverse(pts_fwd)

    assert np.allclose(pts, pts_roundtrip, atol=1e-7)
