import torch
import numpy as np
from src.depth_fusion.rays import generate_rays
from src.depth_fusion.depth_semantics import inverse_to_relative_depth

def test_ray_generation():
    K_rect = np.array([
        [800.0, 0.0, 960.0],
        [0.0, 800.0, 540.0],
        [0.0, 0.0, 1.0]
    ])
    
    # Identity rotation -> Camera points down Z
    R_wc = np.eye(3)
    C_w = np.array([10.0, 20.0, 30.0])
    
    W, H = 1920, 1080
    
    rays_o, rays_d = generate_rays(K_rect, R_wc, C_w, W, H, torch.device('cpu'))
    
    assert rays_o.shape == (H, W, 3)
    assert rays_d.shape == (H, W, 3)
    
    # Check origin
    assert torch.allclose(rays_o[0, 0], torch.tensor([10.0, 20.0, 30.0]))
    
    # Center pixel should point exactly along +Z
    cx, cy = int(K_rect[0, 2]), int(K_rect[1, 2])
    center_ray = rays_d[cy, cx]
    assert torch.allclose(center_ray, torch.tensor([0.0, 0.0, 1.0]), atol=1e-5)
    
    # Normalization check
    norms = torch.norm(rays_d, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

def test_inverse_to_relative_depth():
    inv_depth = torch.tensor([[10.0, 0.0], [5.0, 100.0]])
    rel_depth, meta = inverse_to_relative_depth(inv_depth, epsilon=0.0)
    
    assert torch.allclose(rel_depth[0, 0], torch.tensor(0.1))
    assert torch.allclose(rel_depth[1, 0], torch.tensor(0.2))
    assert torch.allclose(rel_depth[1, 1], torch.tensor(0.01))
    
    assert meta["representation"] == "relative_depth"
    assert meta["metric"] == False
