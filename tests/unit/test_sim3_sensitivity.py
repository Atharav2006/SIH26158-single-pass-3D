import pytest
import numpy as np
import math

from src.metrics.alignment import umeyama_alignment

def test_sim3_noise_sensitivity_zero_noise():
    """Test zero noise perturbation yields zero error relative to reference."""
    np.random.seed(42)
    src = np.random.randn(20, 3) * 5.0
    s_true = 1.85
    R_true = np.eye(3)
    t_true = np.array([1.0, 2.0, 3.0])
    dst = s_true * (src @ R_true.T) + t_true

    s_ref, R_ref, t_ref, _ = umeyama_alignment(src, dst, with_scale=True)
    assert math.isclose(s_ref, s_true, rel_tol=1e-6)

    # Re-estimating with identical unperturbed data
    s_k, R_k, t_k, _ = umeyama_alignment(src, dst, with_scale=True)
    assert math.isclose(s_k, s_ref, rel_tol=1e-6)
    assert np.allclose(R_k, R_ref, atol=1e-6)
    assert np.allclose(t_k, t_ref, atol=1e-6)

def test_sim3_bounded_noise_monotonic_error():
    """Test that increasing noise levels strictly increase residual RMSE."""
    np.random.seed(123)
    src = np.random.randn(50, 3) * 10.0
    dst = 0.5 * src + np.array([2.0, -1.0, 4.0])

    rmses = []
    for sigma in [0.01, 0.10, 0.50]:
        pert_dst = dst + np.random.normal(0.0, sigma, size=dst.shape)
        _, _, _, aligned = umeyama_alignment(src, pert_dst, with_scale=True)
        res = float(np.sqrt(np.mean(np.sum((aligned - pert_dst)**2, axis=1))))
        rmses.append(res)

    assert rmses[0] < rmses[1] < rmses[2], f"Residuals not increasing with noise: {rmses}"
