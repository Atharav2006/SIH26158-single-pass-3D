import torch
import numpy as np
import pytest
from src.depth_fusion.unprojection import unproject_to_3d, project_to_pixels

def test_unprojection_safety_check():
    K_rect = np.eye(3)
    R_wc = np.eye(3)
    C_w = np.zeros(3)
    
    depth = torch.ones((10, 10))
    
    with pytest.raises(ValueError, match="not metrically calibrated"):
        # Without is_metric=True, unprojection must refuse to operate
        unproject_to_3d(depth, K_rect, R_wc, C_w, is_metric=False)

def test_roundtrip_projection():
    W, H = 1920, 1080
    K_rect = np.array([
        [800.0, 0.0, 960.0],
        [0.0, 800.0, 540.0],
        [0.0, 0.0, 1.0]
    ])
    
    # 90 deg rotation around Y
    R_wc = np.array([
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0]
    ])
    C_w = np.array([10.0, 20.0, 30.0])
    
    # Synthetic Metric Depth
    metric_depth = torch.rand((H, W)) * 50.0 + 10.0 # 10 to 60 meters
    
    # 1. Unproject
    points_w = unproject_to_3d(metric_depth, K_rect, R_wc, C_w, is_metric=True)
    
    # 2. Project back
    R_cw = R_wc.T
    pixels, reprojected_depth = project_to_pixels(points_w, K_rect, R_cw, C_w)
    
    # 3. Check pixel roundtrip
    # i, j grid
    i, j = torch.meshgrid(
        torch.arange(W, dtype=torch.float32), 
        torch.arange(H, dtype=torch.float32), 
        indexing='xy'
    )
    original_pixels = torch.stack([i, j], dim=-1)
    
    # Error should be near-zero float precision
    pixel_error = torch.norm(pixels - original_pixels, dim=-1)
    assert torch.max(pixel_error).item() < 1e-3
    
    # Check depth roundtrip
    depth_error = torch.abs(reprojected_depth.squeeze(-1) - metric_depth)
    assert torch.max(depth_error).item() < 1e-3
    
    # Explicitly test corners
    assert pixel_error[0, 0] < 1e-3
    assert pixel_error[0, W-1] < 1e-3
    assert pixel_error[H-1, 0] < 1e-3
    assert pixel_error[H-1, W-1] < 1e-3
    
    # Test Center
    cx, cy = int(K_rect[0, 2]), int(K_rect[1, 2])
    assert pixel_error[cy, cx] < 1e-3
