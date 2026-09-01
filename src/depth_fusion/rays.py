import numpy as np
import torch
from typing import Tuple

def generate_rays(K_rect: np.ndarray, R_wc: np.ndarray, C_world: np.ndarray, width: int, height: int, device: torch.device = torch.device('cpu')) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates ray origins and directions in world coordinates for a rectified pinhole camera.
    
    Args:
        K_rect: 3x3 rectified intrinsic matrix
        R_wc: 3x3 Camera-to-World rotation matrix
        C_world: 3x1 or (3,) World camera center
        width: Image width
        height: Image height
        device: Torch device
        
    Returns:
        rays_o: [H, W, 3] ray origins in world frame
        rays_d: [H, W, 3] normalized ray directions in world frame
    """
    # Grid of pixel coordinates
    i, j = torch.meshgrid(
        torch.arange(width, dtype=torch.float32, device=device), 
        torch.arange(height, dtype=torch.float32, device=device), 
        indexing='xy'
    )
    
    fx = K_rect[0, 0]
    fy = K_rect[1, 1]
    cx = K_rect[0, 2]
    cy = K_rect[1, 2]
    
    # Camera rays (OpenCV convention: +X Right, +Y Down, +Z Forward)
    dirs_c = torch.stack([
        (i - cx) / fx,
        (j - cy) / fy,
        torch.ones_like(i)
    ], dim=-1)  # [H, W, 3]
    
    R_wc_t = torch.from_numpy(R_wc).float().to(device)
    C_world_t = torch.from_numpy(C_world).float().to(device).view(1, 1, 3)
    
    # Transform to world coordinates: ray_d_world = R_wc @ ray_d_cam
    rays_d = torch.einsum('ij,hwj->hwi', R_wc_t, dirs_c)
    
    # Normalize
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    
    rays_o = C_world_t.expand_as(rays_d)
    
    return rays_o, rays_d
