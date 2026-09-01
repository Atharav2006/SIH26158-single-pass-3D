import pytest
import torch
import numpy as np

from src.neural_reconstruction.model import TinyNeRF
from src.neural_reconstruction.renderer import VolumetricRenderer
from src.neural_reconstruction.trainer import Trainer

def test_synthetic_sanity_overfit():
    """
    Synthetic Sanity Test:
    Ensures that TinyNeRF can perfectly overfit to a single batch of rays.
    This separates model/rendering bugs from dataset degeneracy issues.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(42)  # Deterministic seed for reproducibility
    
    # 1. Initialize
    model = TinyNeRF(hidden_dim=64, num_layers=4).to(device)
    # Use white background to prevent zero-density gradient trap on pure red targets
    renderer = VolumetricRenderer(bg_color=(1.0, 1.0, 1.0)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    trainer = Trainer(model, renderer, opt, device, near=2.0, far=6.0, num_samples=32)
    
    # 2. Create synthetic batch of rays
    batch_size = 1024
    rays_o = torch.zeros((batch_size, 3), device=device)
    rays_o[:, 2] = -4.0 # Camera at Z=-4
    
    # Random directions pointing roughly forward (+Z)
    rays_d = torch.randn((batch_size, 3), device=device)
    rays_d[:, 2] = torch.abs(rays_d[:, 2]) + 1.0
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    
    # Synthetic target: red solid object at Z=0. If ray intersects object, target is red, else white.
    # For simplicity, let's just make the target exactly red [1, 0, 0] for all rays to test overfit.
    target_rgb = torch.tensor([1.0, 0.0, 0.0], device=device).expand(batch_size, 3)
    
    # 3. Train for a few iterations
    initial_loss = None
    final_loss = None
    
    for i in range(200):
        loss, *_ = trainer.train_step(rays_o, rays_d, target_rgb)
        if i == 0:
            initial_loss = loss
        final_loss = loss
        
    # The loss should decrease significantly if the network works
    assert final_loss < initial_loss
    assert final_loss < 0.05, f"Failed to overfit synthetic data. Final loss: {final_loss}"
    
    if torch.cuda.is_available():
        mem = torch.cuda.max_memory_allocated() / 1e9
        assert mem < 3.5, f"VRAM exceeded safe limit during unit test: {mem} GB"
