import torch
import pytest
from src.neural_reconstruction.depth_prior import MiDaSDepthPrior

def test_midas_depth_prior():
    device = torch.device('cpu')
    try:
        prior = MiDaSDepthPrior(device)
    except Exception as e:
        pytest.skip(f"MiDaS could not load, skipping test: {e}")
        
    assert prior.metadata()['scale_invariant'] == True
    
    # Test batch prediction
    B, H, W = 2, 144, 256
    dummy_images = torch.rand(B, H, W, 3)
    
    depths = prior.predict_batch(dummy_images)
    
    assert depths.shape == (B, H, W)
    
    # Test normalization
    norm_depth = prior.normalize_depth(depths)
    
    mean = norm_depth.mean(dim=[-1,-2])
    std = norm_depth.std(dim=[-1,-2])
    
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)
    assert torch.allclose(std, torch.ones_like(std), atol=1e-5)
