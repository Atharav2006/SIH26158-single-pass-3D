import numpy as np
import pytest
import csv
from pathlib import Path

def test_b2_poses_validity():
    poses_path = Path("outputs/reports/zurich_mav/b2/b2_fused_trajectory.csv")
    if not poses_path.exists():
        pytest.skip(f"Poses file not found: {poses_path}")
        
    with open(poses_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    # We should have 350 states
    assert len(rows) == 350
    
    frames_to_test = [1, 175, 349] # 0-indexed: frame 1, 175, 350
    
    for f_idx in frames_to_test:
        row = rows[f_idx]
        
        from scipy.spatial.transform import Rotation
        
        # B2 uses xyzw quaternions: qx, qy, qz, qw
        q = [float(row['qx']), float(row['qy']), float(row['qz']), float(row['qw'])]
        R_wc = Rotation.from_quat(q).as_matrix()
        
        # 1. Check determinant is +1 (proper rotation)
        det = np.linalg.det(R_wc)
        assert np.isclose(det, 1.0, atol=1e-4)
        
        # 2. Check R_wc * R_wc^T = I
        identity_test = R_wc @ R_wc.T
        assert np.allclose(identity_test, np.eye(3), atol=1e-4)
        
        # 3. Check C_world
        C_w = np.array([float(row['x']), float(row['y']), float(row['z'])])
        assert np.isfinite(C_w).all()
