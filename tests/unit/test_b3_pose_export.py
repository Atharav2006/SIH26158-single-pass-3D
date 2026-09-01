import pytest
import numpy as np
from pathlib import Path
from src.reconstruction.dense_mvs import b2_pose_to_colmap_pose, colmap_pose_to_b2_pose, ColmapWorkspace
from src.metrics.alignment import quaternion_to_rotation_matrix

def test_b2_to_colmap_conversion_invariants():
    # Synthetic B2 pose: Camera-to-World
    q_wc = np.array([0.0, 0.0, 0.70710678, 0.70710678]) # 90 deg yaw
    c_w = np.array([10.0, 20.0, 30.0])
    
    # 1. Convert to COLMAP
    q_cw, t_cw = b2_pose_to_colmap_pose(q_wc, c_w)
    
    R_wc = quaternion_to_rotation_matrix(q_wc)
    R_cw = quaternion_to_rotation_matrix(q_cw)
    
    # Orthogonality checks
    assert np.allclose(R_wc @ R_wc.T, np.eye(3))
    assert np.allclose(R_cw @ R_cw.T, np.eye(3))
    assert np.isclose(np.linalg.det(R_wc), 1.0)
    assert np.isclose(np.linalg.det(R_cw), 1.0)
    
    # R_cw should be transpose of R_wc
    assert np.allclose(R_cw, R_wc.T)
    
    # Forward check: X_w = [10, 20, 30] should map to X_c = [0, 0, 0]
    X_w = c_w
    X_c = R_cw @ X_w + t_cw
    assert np.allclose(X_c, np.zeros(3))
    
    # 2. Convert back to B2
    q_wc_rec, c_w_rec = colmap_pose_to_b2_pose(q_cw, t_cw)
    
    # Assert recovery
    assert np.allclose(c_w_rec, c_w)
    # Quaternions can be inverted (q and -q) but represent the same rotation.
    R_wc_rec = quaternion_to_rotation_matrix(q_wc_rec)
    assert np.allclose(R_wc_rec, R_wc)

def test_colmap_workspace_serialization(tmp_path):
    ws = ColmapWorkspace(tmp_path)
    b2_poses = [
        {
            "imgid": 1,
            "filename": "00001.jpg",
            "q_wc": np.array([0.0, 0.0, 0.70710678, 0.70710678]),
            "c_w": np.array([10.0, 20.0, 30.0])
        },
        {
            "imgid": 2,
            "filename": "00002.jpg",
            "q_wc": np.array([0.0, 0.0, 0.0, 1.0]),
            "c_w": np.array([-5.0, 5.0, 10.0])
        }
    ]
    
    ws.write_images_txt(b2_poses)
    
    recovered_poses = ws.read_images_txt()
    
    assert len(recovered_poses) == 2
    for orig, rec in zip(b2_poses, recovered_poses):
        assert orig["imgid"] == rec["imgid"]
        assert orig["filename"] == rec["filename"]
        assert np.allclose(orig["c_w"], rec["c_w"])
        
        R_orig = quaternion_to_rotation_matrix(orig["q_wc"])
        R_rec = quaternion_to_rotation_matrix(rec["q_wc"])
        assert np.allclose(R_orig, R_rec)
