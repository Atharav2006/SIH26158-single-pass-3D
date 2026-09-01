import torch
import torch.nn as nn
import time
from pathlib import Path

def sample_z_vals(near: float, far: float, num_samples: int, batch_size: int, device: torch.device):
    """
    Stratified sampling along the ray.
    """
    t_vals = torch.linspace(0., 1., num_samples, device=device)
    z_vals = near * (1. - t_vals) + far * t_vals
    z_vals = z_vals.expand(batch_size, num_samples)
    
    # Stratified sampling
    mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
    upper = torch.cat([mids, z_vals[..., -1:]], -1)
    lower = torch.cat([z_vals[..., :1], mids], -1)
    t_rand = torch.rand(z_vals.shape, device=device)
    z_vals = lower + (upper - lower) * t_rand
    
    return z_vals

class Trainer:
    def __init__(self, model, renderer, optimizer, device, near=0.1, far=10.0, num_samples=64, 
                 lambda_depth=0.0, lambda_smooth=0.0, lambda_reg=0.0):
        self.model = model
        self.renderer = renderer
        self.optimizer = optimizer
        self.device = device
        self.near = near
        self.far = far
        self.num_samples = num_samples
        self.loss_fn = nn.MSELoss()
        
        self.lambda_depth = lambda_depth
        self.lambda_smooth = lambda_smooth
        self.lambda_reg = lambda_reg

    def check_vram(self):
        if not torch.cuda.is_available():
            return
        mem = torch.cuda.max_memory_allocated() / 1e9
        if mem > 4.0:
            raise RuntimeError(f"VRAM exceeded safe abort limit: {mem:.2f} GB > 4.0 GB")
        elif mem > 3.8:
            print(f"WARNING: VRAM near limit: {mem:.2f} GB > 3.8 GB")

    def train_step(self, rays_o, rays_d, target_rgb, target_depth=None):
        self.model.train()
        self.optimizer.zero_grad()
        
        batch_size = rays_o.shape[0]
        z_vals = sample_z_vals(self.near, self.far, self.num_samples, batch_size, self.device)
        
        # Calculate 3D points
        pts = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_vals.unsqueeze(-1) # [B, S, 3]
        dirs = rays_d.unsqueeze(1).expand_as(pts)
        
        # Forward pass
        density, rgb = self.model(pts, dirs)
        
        # Render
        comp_rgb, depth_map, acc_map = self.renderer(density, rgb, z_vals)
        
        loss_rgb = self.loss_fn(comp_rgb, target_rgb)
        
        # Additional losses
        loss_depth = torch.tensor(0.0, device=self.device)
        loss_smooth = torch.tensor(0.0, device=self.device)
        loss_reg = torch.tensor(0.0, device=self.device)
        
        if self.lambda_depth > 0 and target_depth is not None:
            # Import dynamically to avoid circular dependencies
            from src.neural_reconstruction.depth_loss import scale_invariant_depth_loss
            # Flatten target_depth if it was passed in block shape
            target_depth_flat = target_depth.view(-1, 1) if target_depth.dim() > 1 else target_depth.unsqueeze(-1)
            loss_depth = scale_invariant_depth_loss(depth_map, target_depth_flat)
            
        if self.lambda_reg > 0:
            from src.neural_reconstruction.depth_loss import density_regularization
            loss_reg = density_regularization(density)
            
        # Smoothness usually requires 2D grid, but we can't easily compute it on random 1D rays.
        # If lambda_smooth > 0, we assume the batch represents a small grid or full image.
        if self.lambda_smooth > 0 and comp_rgb.dim() == 4: # [B, H, W, 3]
            from src.neural_reconstruction.depth_loss import edge_aware_smoothness_loss
            loss_smooth = edge_aware_smoothness_loss(depth_map.unsqueeze(0), target_rgb.unsqueeze(0))
            
        total_loss = loss_rgb + self.lambda_depth * loss_depth + self.lambda_smooth * loss_smooth + self.lambda_reg * loss_reg
        total_loss.backward()
        self.optimizer.step()
        
        self.check_vram()
        return total_loss.item(), loss_rgb.item(), loss_depth.item(), comp_rgb, depth_map

    @torch.no_grad()
    def render_image(self, rays_o, rays_d, chunk_size=4096):
        self.model.eval()
        H, W, _ = rays_o.shape
        rays_o = rays_o.view(-1, 3)
        rays_d = rays_d.view(-1, 3)
        
        comp_rgbs = []
        depth_maps = []
        
        for i in range(0, rays_o.shape[0], chunk_size):
            ro = rays_o[i:i+chunk_size]
            rd = rays_d[i:i+chunk_size]
            
            z_vals = sample_z_vals(self.near, self.far, self.num_samples, ro.shape[0], self.device)
            pts = ro.unsqueeze(1) + rd.unsqueeze(1) * z_vals.unsqueeze(-1)
            dirs = rd.unsqueeze(1).expand_as(pts)
            
            density, rgb = self.model(pts, dirs)
            comp_rgb, depth, _ = self.renderer(density, rgb, z_vals)
            
            comp_rgbs.append(comp_rgb)
            depth_maps.append(depth)
            
        final_rgb = torch.cat(comp_rgbs, dim=0).view(H, W, 3)
        final_depth = torch.cat(depth_maps, dim=0).view(H, W, 1)
        
        self.check_vram()
        return final_rgb, final_depth
