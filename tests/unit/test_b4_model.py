import pytest
import torch
from src.neural_reconstruction.model import TinyNeRF, PositionalEncoding

def test_positional_encoding():
    pe = PositionalEncoding(in_dim=3, num_freqs=10)
    x = torch.rand(10, 3)
    out = pe(x)
    # in_dim + 2 * in_dim * num_freqs = 3 + 2 * 3 * 10 = 63
    assert out.shape == (10, 63)
    
def test_tinynerf_forward():
    model = TinyNeRF(hidden_dim=64, num_layers=4)
    x = torch.rand(32, 64, 3) # [Batch, Samples, 3]
    d = torch.rand(32, 64, 3)
    
    # Normalize d
    d = d / torch.norm(d, dim=-1, keepdim=True)
    
    density, rgb = model(x, d)
    
    assert density.shape == (32, 64, 1)
    assert rgb.shape == (32, 64, 3)
    
    # Density must be non-negative
    assert torch.all(density >= 0)
    
    # RGB must be in [0, 1]
    assert torch.all(rgb >= 0) and torch.all(rgb <= 1)

def test_tinynerf_finite():
    model = TinyNeRF()
    x = torch.randn(10, 3) * 100.0 # Extreme coords
    d = torch.randn(10, 3)
    d = d / torch.norm(d, dim=-1, keepdim=True)
    
    density, rgb = model(x, d)
    assert torch.all(torch.isfinite(density))
    assert torch.all(torch.isfinite(rgb))
