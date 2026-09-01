import torch
import torch.nn as nn
from typing import Tuple

class VolumetricRenderer(nn.Module):
    def __init__(self, bg_color: Tuple[float, float, float] = (0.0, 0.0, 0.0)):
        super().__init__()
        self.bg_color = bg_color

    def forward(self, 
                density: torch.Tensor, 
                rgb: torch.Tensor, 
                z_vals: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        density: [Batch, Samples, 1]
        rgb: [Batch, Samples, 3]
        z_vals: [Batch, Samples] distances along the ray
        
        Returns:
        comp_rgb: [Batch, 3]
        depth: [Batch, 1]
        acc_map: [Batch, 1]
        """
        # Calculate distances between samples
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        
        # Add a large distance for the last sample to represent infinity
        last_dist = torch.full((dists.shape[0], 1), 1e10, dtype=z_vals.dtype, device=z_vals.device)
        dists = torch.cat([dists, last_dist], dim=-1) # [Batch, Samples]
        
        # Multiply density by distance
        dists = dists.unsqueeze(-1) # [Batch, Samples, 1]
        
        # alpha = 1 - exp(-sigma * delta)
        # Add tiny eps for numerical stability
        alpha = 1.0 - torch.exp(-density * dists + 1e-10)
        
        # Transmittance T_i = prod_{j=1}^{i-1} (1 - alpha_j)
        # Using cumprod is standard, but we shift it right by 1
        trans = torch.cumprod(1.0 - alpha + 1e-10, dim=1)
        trans = torch.cat([torch.ones_like(trans[:, :1, :]), trans[:, :-1, :]], dim=1)
        
        # weights = alpha * T
        weights = alpha * trans # [Batch, Samples, 1]
        
        # Composite RGB
        comp_rgb = torch.sum(weights * rgb, dim=1) # [Batch, 3]
        
        # Depth Map (expected depth)
        depth_map = torch.sum(weights * z_vals.unsqueeze(-1), dim=1) # [Batch, 1]
        
        # Accumulated opacity (silhouette)
        acc_map = torch.sum(weights, dim=1) # [Batch, 1]
        
        # Background composite
        if sum(self.bg_color) > 0.0:
            bg = torch.tensor(self.bg_color, dtype=rgb.dtype, device=rgb.device)
            comp_rgb = comp_rgb + (1.0 - acc_map) * bg
            
        return comp_rgb, depth_map, acc_map
