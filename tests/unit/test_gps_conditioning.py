import pytest
import numpy as np
import math

def compute_point_cloud_conditioning(pts: np.ndarray):
    n = len(pts)
    mean = np.mean(pts, axis=0)
    demeaned = pts - mean
    cov = (demeaned.T @ demeaned) / (n - 1)

    evals, _ = np.linalg.eigh(cov)
    evals = np.sort(evals)[::-1]
    total_var = float(np.sum(evals))
    var_exp = (evals / total_var * 100.0) if total_var > 0 else np.zeros(3)

    _, s_vals, _ = np.linalg.svd(demeaned)
    cond_num = float(s_vals[0] / s_vals[-1]) if s_vals[-1] > 1e-12 else float("inf")

    return {
        "covariance": cov,
        "eigenvalues": evals,
        "singular_values": s_vals,
        "condition_number": cond_num,
        "explained_variance": var_exp,
        "rank": 3 if s_vals[-1] > 1e-3 else (2 if s_vals[1] > 1e-3 else 1),
        "is_planar": bool(var_exp[2] < 2.0),
        "is_linear": bool(var_exp[1] + var_exp[2] < 5.0)
    }

def test_spherical_isotropic_conditioning():
    """Test isotropic 3D point cloud has condition number ~1.0 and equal variance."""
    np.random.seed(42)
    pts = np.random.randn(500, 3) * 5.0
    res = compute_point_cloud_conditioning(pts)

    assert res["rank"] == 3
    assert res["condition_number"] < 1.5
    assert not res["is_planar"]
    assert not res["is_linear"]
    assert all(25.0 < v < 42.0 for v in res["explained_variance"])

def test_planar_degeneracy_detection():
    """Test strictly 2D flat point cloud is detected as planar degenerate."""
    np.random.seed(42)
    xy = np.random.randn(100, 2) * 5.0
    z = np.zeros((100, 1))  # Zero variance in Z
    pts = np.hstack([xy, z])
    res = compute_point_cloud_conditioning(pts)

    assert res["is_planar"]
    assert res["rank"] == 2
    assert math.isclose(res["explained_variance"][2], 0.0, abs_tol=1e-5)

def test_linear_collinear_degeneracy_detection():
    """Test 1D collinear point cloud is detected as linear degenerate."""
    t = np.linspace(0, 10, 50).reshape(-1, 1)
    pts = np.hstack([t, 2.0 * t, -0.5 * t])
    res = compute_point_cloud_conditioning(pts)

    assert res["is_linear"]
    assert res["is_planar"]
    assert res["rank"] == 1
    assert math.isclose(res["explained_variance"][0], 100.0, abs_tol=1e-4)
