import torch
import numpy as np

def unproject_to_3d(depth: torch.Tensor, K_rect: np.ndarray, R_wc: np.ndarray, C_world: np.ndarray, is_metric: bool = False) -> torch.Tensor:
    """
    Unprojects a depth map into 3D world coordinates.
    
    Args:
        depth: [H, W] depth map.
        K_rect: 3x3 rectified intrinsic matrix.
        R_wc: 3x3 Camera-to-World rotation matrix.
        C_world: 3x1 or (3,) World camera center.
        is_metric: boolean flag explicitly verifying the depth has been metrically calibrated.
        
    Returns:
        points_w: [H, W, 3] 3D points in the world frame.
        
    Raises:
        ValueError if the depth is not explicitly marked as metric.
    """
    if not is_metric:
        raise ValueError("Depth map is not metrically calibrated! Cannot unproject to metric 3D space. Perform Phase 3 calibration first.")
        
    H, W = depth.shape
    device = depth.device
    
    # Grid of pixel coordinates
    i, j = torch.meshgrid(
        torch.arange(W, dtype=torch.float32, device=device), 
        torch.arange(H, dtype=torch.float32, device=device), 
        indexing='xy'
    )
    
    fx = K_rect[0, 0]
    fy = K_rect[1, 1]
    cx = K_rect[0, 2]
    cy = K_rect[1, 2]
    
    # Unproject to camera frame
    X_c = (i - cx) * depth / fx
    Y_c = (j - cy) * depth / fy
    Z_c = depth
    
    pts_c = torch.stack([X_c, Y_c, Z_c], dim=-1)  # [H, W, 3]
    
    # Transform to world frame
    R_wc_t = torch.from_numpy(R_wc).float().to(device)
    C_world_t = torch.from_numpy(C_world).float().to(device).view(1, 1, 3)
    
    # X_w = R_wc * X_c + C_world
    pts_w = torch.einsum('ij,hwj->hwi', R_wc_t, pts_c) + C_world_t
    
    return pts_w

def project_to_pixels(points_w: torch.Tensor, K_rect: np.ndarray, R_cw: np.ndarray, C_world: np.ndarray) -> torch.Tensor:
    """
    Projects world points back into camera pixel coordinates.
    Useful for round-trip validation and multi-view consistency.
    
    Args:
        points_w: [..., 3] points in world frame
        K_rect: 3x3 rectified intrinsic matrix
        R_cw: 3x3 World-to-Camera rotation matrix (inverse of R_wc)
        C_world: 3x1 World camera center
        
    Returns:
        pixels: [..., 2] (u, v) pixel coordinates
        depths: [..., 1] Z_c depth values
    """
    device = points_w.device
    
    R_cw_t = torch.from_numpy(R_cw).float().to(device)
    C_world_t = torch.from_numpy(C_world).float().to(device)
    
    # Translate to camera origin: X - C
    pts_translated = points_w - C_world_t
    
    # Rotate to camera frame: X_c = R_cw * (X_w - C_w)
    pts_c = torch.einsum('ij,...j->...i', R_cw_t, pts_translated)
    
    X_c, Y_c, Z_c = pts_c[..., 0], pts_c[..., 1], pts_c[..., 2]
    
    fx = K_rect[0, 0]
    fy = K_rect[1, 1]
    cx = K_rect[0, 2]
    cy = K_rect[1, 2]
    
    # Avoid division by zero
    Z_c_safe = torch.clamp(Z_c, min=1e-6)
    
    u = (X_c * fx / Z_c_safe) + cx
    v = (Y_c * fy / Z_c_safe) + cy
    
    pixels = torch.stack([u, v], dim=-1)
    
    return pixels, Z_c.unsqueeze(-1)
