import pytest
import numpy as np
from src.sensor_fusion.b2_optimizer import B2TrajectoryOptimizer
from src.sensor_fusion.sensor_factors import GPSFactor

def test_b2_optimizer_packing():
    N = 3
    init_rot = [np.eye(3) for _ in range(N)]
    init_pos = np.zeros((N, 3))
    init_vel = np.zeros((N, 3))
    
    opt = B2TrajectoryOptimizer(N, init_rot, init_pos, init_vel)
    
    # Random perturbation
    x = np.random.randn(N * 9) * 0.1
    R_list, p, v = opt._unpack_state_vector(x)
    
    assert len(R_list) == N
    assert p.shape == (N, 3)
    assert v.shape == (N, 3)
    
    # Test that delta_p maps exactly
    assert np.allclose(p[0], x[3:6])
    assert np.allclose(v[0], x[6:9])

def test_b2_optimizer_sparsity_structure():
    N = 4
    init_rot = [np.eye(3) for _ in range(N)]
    init_pos = np.zeros((N, 3))
    init_vel = np.zeros((N, 3))
    opt = B2TrajectoryOptimizer(N, init_rot, init_pos, init_vel)
    
    for i in range(N):
        opt.add_gps_factor(GPSFactor(i, np.zeros(3)))
        
    sparsity = opt._build_sparsity()
    assert sparsity.shape == (N * 3, N * 9)
    # Check that GPS factor i only touches state i's position
    for i in range(N):
        row = i * 3
        col_start = i * 9 + 3
        col_end = i * 9 + 6
        assert np.all(sparsity[row:row+3, col_start:col_end].toarray() == 1)
        assert np.sum(sparsity[row:row+3, :].toarray()) == 9 # 3 rows * 3 cols = 9 ones
