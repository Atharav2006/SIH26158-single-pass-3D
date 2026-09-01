import pytest
import torch
from src.neural_reconstruction.renderer import VolumetricRenderer

def test_renderer_alpha_and_weights():
    renderer = VolumetricRenderer()
    
    # 2 rays, 3 samples each
    density = torch.tensor([
        [[0.0], [0.0], [0.0]],       # Ray 0: empty space
        [[100.0], [100.0], [0.0]]    # Ray 1: dense solid
    ], dtype=torch.float32)
    
    rgb = torch.ones(2, 3, 3, dtype=torch.float32)
    
    z_vals = torch.tensor([
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0]
    ], dtype=torch.float32)
    
    comp_rgb, depth_map, acc_map = renderer(density, rgb, z_vals)
    
    # Ray 0: Everything should be 0 because density is 0
    assert torch.allclose(comp_rgb[0], torch.zeros(3))
    assert torch.allclose(acc_map[0], torch.zeros(1))
    
    # Ray 1: Extremely dense, should hit at the first sample (z=1.0)
    # alpha will be 1.0, so weight for first sample = 1.0, others = 0.0
    assert torch.allclose(acc_map[1], torch.ones(1))
    assert torch.allclose(depth_map[1], torch.tensor([1.0]), atol=1e-2)
    assert torch.allclose(comp_rgb[1], torch.ones(3))

def test_renderer_finite():
    renderer = VolumetricRenderer()
    density = torch.randn(10, 64, 1) * 1000.0  # Random extreme values
    # Must be non-negative in practice, but let's just test absolute values
    density = torch.abs(density) 
    
    rgb = torch.rand(10, 64, 3)
    z_vals = torch.sort(torch.rand(10, 64))[0] * 100.0
    
    comp_rgb, depth_map, acc_map = renderer(density, rgb, z_vals)
    
    assert torch.all(torch.isfinite(comp_rgb))
    assert torch.all(torch.isfinite(depth_map))
    assert torch.all(torch.isfinite(acc_map))
    assert torch.all((acc_map >= 0.0) & (acc_map <= 1.0 + 1e-5))
