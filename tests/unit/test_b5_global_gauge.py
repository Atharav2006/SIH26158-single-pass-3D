import pytest
import numpy as np
from src.depth_fusion.global_gauge_alignment import GlobalGaugeSolver

def test_global_solver_synthetic_clean():
    np.random.seed(42)
    N = 50
    s_true = np.random.uniform(0.5, 2.0, N)
    t_true = np.random.uniform(-100, 100, N)

    edges = []
    # Build a simple chain
    for i in range(N-1):
        j = i + 1
        a_ij = s_true[j] / s_true[i]
        b_ij = t_true[j] - a_ij * t_true[i]
        edges.append({'i': i, 'j': j, 'a': a_ij, 'b': b_ij, 'w': 1.0})
        
    solver = GlobalGaugeSolver(ref_frame=0)
    a_est, b_est, status = solver.solve(edges, N)
    
    assert status == "SUCCESS"
    assert a_est is not None
    assert b_est is not None
    
    true_a = s_true[0] / s_true
    true_b = t_true[0] - true_a * t_true
    
    np.testing.assert_allclose(a_est, true_a, rtol=1e-5)
    np.testing.assert_allclose(b_est, true_b, atol=1e-5)

def test_global_solver_with_loops():
    np.random.seed(43)
    N = 20
    s_true = np.random.uniform(0.5, 2.0, N)
    t_true = np.random.uniform(-50, 50, N)

    edges = []
    # Build chain
    for i in range(N-1):
        j = i + 1
        a_ij = s_true[j] / s_true[i]
        b_ij = t_true[j] - a_ij * t_true[i]
        edges.append({'i': i, 'j': j, 'a': a_ij, 'b': b_ij, 'w': 1.0})
        
    # Add random cross-edges (loops)
    for _ in range(30):
        i = np.random.randint(0, N)
        j = np.random.randint(0, N)
        if i != j:
            a_ij = s_true[j] / s_true[i]
            b_ij = t_true[j] - a_ij * t_true[i]
            edges.append({'i': i, 'j': j, 'a': a_ij, 'b': b_ij, 'w': 0.5})
            
    solver = GlobalGaugeSolver(ref_frame=5) # use different ref frame
    a_est, b_est, status = solver.solve(edges, N)
    
    assert status == "SUCCESS"
    
    true_a = s_true[5] / s_true
    true_b = t_true[5] - true_a * t_true
    
    np.testing.assert_allclose(a_est, true_a, rtol=1e-5)
    np.testing.assert_allclose(b_est, true_b, atol=1e-5)
