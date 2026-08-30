import pytest
import numpy as np
import math

from src.metrics.alignment import (
    umeyama_alignment,
    rotation_matrix_to_quaternion,
    quaternion_to_rotation_matrix
)

def test_umeyama_identity_alignment():
    """Test that identical point clouds yield s=1, R=I, t=0 with zero error."""
    np.random.seed(42)
    pts = np.random.randn(20, 3) * 10.0
    s, R, t, aligned = umeyama_alignment(pts, pts, with_scale=True)

    assert math.isclose(s, 1.0, rel_tol=1e-6)
    assert np.allclose(R, np.eye(3), atol=1e-6)
    assert np.allclose(t, np.zeros(3), atol=1e-6)
    assert np.allclose(aligned, pts, atol=1e-6)

def test_umeyama_known_synthetic_similarity():
    """Test exact recovery of known scale, rotation, and translation."""
    np.random.seed(123)
    src = np.random.randn(15, 3) * 5.0

    # Known ground-truth transform
    s_gt = 3.75
    theta = math.radians(35.0)
    R_gt = np.array([
        [math.cos(theta), -math.sin(theta), 0.0],
        [math.sin(theta),  math.cos(theta), 0.0],
        [0.0,              0.0,             1.0]
    ])
    t_gt = np.array([12.5, -8.3, 4.1])

    dst = s_gt * (src @ R_gt.T) + t_gt

    s_est, R_est, t_est, aligned = umeyama_alignment(src, dst, with_scale=True)

    assert math.isclose(s_est, s_gt, rel_tol=1e-6)
    assert np.allclose(R_est, R_gt, atol=1e-6)
    assert np.allclose(t_est, t_gt, atol=1e-6)
    assert np.allclose(aligned, dst, atol=1e-6)
    assert math.isclose(float(np.linalg.det(R_est)), 1.0, rel_tol=1e-6)

def test_umeyama_se3_rigid_alignment():
    """Test pure SE(3) alignment enforcing scale s = 1.0."""
    np.random.seed(77)
    src = np.random.randn(10, 3)
    R_gt = np.array([
        [0.0, -1.0, 0.0],
        [1.0,  0.0, 0.0],
        [0.0,  0.0, 1.0]
    ])
    t_gt = np.array([2.0, 3.0, -1.0])
    dst = (src @ R_gt.T) + t_gt

    s_est, R_est, t_est, aligned = umeyama_alignment(src, dst, with_scale=False)

    assert s_est == 1.0
    assert np.allclose(R_est, R_gt, atol=1e-6)
    assert np.allclose(t_est, t_gt, atol=1e-6)
    assert np.allclose(aligned, dst, atol=1e-6)

def test_quaternion_matrix_roundtrip():
    """Test quaternion to rotation matrix and back conversion."""
    q_orig = np.array([0.2, -0.3, 0.5, 0.7745966692])
    q_orig /= np.linalg.norm(q_orig)

    R = quaternion_to_rotation_matrix(q_orig)
    assert math.isclose(float(np.linalg.det(R)), 1.0, rel_tol=1e-6)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-6)

    q_rec = rotation_matrix_to_quaternion(R)
    # Check sign ambiguity: q == -q represents same rotation
    if q_orig[3] * q_rec[3] < 0:
        q_rec = -q_rec

    assert np.allclose(q_orig, q_rec, atol=1e-6)
