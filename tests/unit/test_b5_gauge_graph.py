import pytest
import numpy as np
from src.depth_fusion.global_gauge_alignment import pose_aware_correspondences, fit_pairwise_affine

def test_pose_aware_correspondences():
    H, W = 100, 100
    D_i = np.ones((H, W), dtype=np.float32) * 5.0
    D_j = np.ones((H, W), dtype=np.float32) * 5.0
    
    K_rect = np.array([
        [100, 0, 50],
        [0, 100, 50],
        [0, 0, 1]
    ], dtype=np.float32)
    
    R_wc_i = np.eye(3)
    C_w_i = np.zeros(3)
    
    # Small translation
    R_wc_j = np.eye(3)
    C_w_j = np.array([0.1, 0, 0])
    
    D_i_vals, D_j_vals, weights = pose_aware_correspondences(
        D_i, K_rect, R_wc_i, C_w_i,
        D_j, R_wc_j, C_w_j,
        downsample_factor=2
    )
    
    assert len(D_i_vals) > 1000
    assert len(D_j_vals) == len(D_i_vals)
    assert np.all(D_i_vals == 5.0)
    assert np.all(D_j_vals == 5.0)

def test_fit_pairwise_affine():
    # D_j = 2.0 * D_i + 1.5
    D_i = np.random.rand(500) * 10
    D_j = 2.0 * D_i + 1.5 + np.random.randn(500)*0.01
    weights = np.ones_like(D_i)
    
    res = fit_pairwise_affine(D_i, D_j, weights)
    
    assert res['status'] == "SUCCESS"
    assert np.isclose(res['a'], 2.0, atol=0.1)
    assert np.isclose(res['b'], 1.5, atol=0.1)
    assert res['correlation'] > 0.99
    
def test_insufficient_overlap():
    res = fit_pairwise_affine(np.ones(50), np.ones(50), np.ones(50))
    assert res['status'] == "INSUFFICIENT_POINTS"

def test_zero_variance():
    res = fit_pairwise_affine(np.ones(500), np.ones(500), np.ones(500))
    assert res['status'] == "ZERO_VARIANCE"
