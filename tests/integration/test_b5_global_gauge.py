import pytest
import numpy as np
from src.depth_fusion.global_gauge_alignment import align_sequence, GaugeRepresentation

def test_align_sequence_synthetic():
    np.random.seed(42)
    H, W = 100, 100
    N = 10
    
    K_rect = np.array([
        [100, 0, 50],
        [0, 100, 50],
        [0, 0, 1]
    ], dtype=np.float32)
    
    depth_sequence = {}
    
    # Ground truth relative scales and shifts
    s_true = np.random.uniform(0.5, 2.0, N)
    t_true = np.random.uniform(5.0, 10.0, N)
    
    for i in range(N):
        # Base depth has variation to avoid ZERO_VARIANCE rejection
        D_true = np.ones((H, W), dtype=np.float32) * 20.0
        y, x = np.mgrid[0:H, 0:W]
        D_true += (x / 10.0).astype(np.float32)
        
        # Apply transformation to simulate relative depth D_inv
        D_inv = s_true[i] * D_true + t_true[i]
        
        depth_sequence[i] = {
            "D_inv": D_inv,
            "conf": np.ones((H, W), dtype=np.float32),
            "K_rect": K_rect,
            "R_wc": np.eye(3),
            "C_w": np.array([0.0, 0.0, 0.0]) # same pose so overlap is perfect
        }
        
    overlap_graph = [(i, i+1) for i in range(N-1)]
    
    res = align_sequence(depth_sequence, overlap_graph, reference_frame=0, representation=GaugeRepresentation.D_INV)
    
    assert res["status"] == "RELATIVE_GAUGE_ESTABLISHED"
    
    a_est = np.array([res["global_scales"][i] for i in range(N)])
    b_est = np.array([res["global_shifts"][i] for i in range(N)])
    
    true_a = s_true[0] / s_true
    true_b = t_true[0] - true_a * t_true
    
    np.testing.assert_allclose(a_est, true_a, rtol=1e-3)
    np.testing.assert_allclose(b_est, true_b, atol=1e-3)
    
    # Aligned depths should all be equal to D_inv_0
    D_0 = depth_sequence[0]["D_inv"]
    for i in range(N):
        np.testing.assert_allclose(res["aligned_depths"][i], D_0, atol=1e-3)

def test_missing_data_handling():
    res = align_sequence({}, [])
    assert res["status"] == "RELATIVE_GAUGE_NOT_IDENTIFIABLE"
