import torch
import pytest
from src.depth_fusion.depth_prior import MiDaSDepthPrior

def test_midas_depth_prior_metadata():
    try:
        prior = MiDaSDepthPrior(torch.device('cpu'))
    except Exception as e:
        pytest.skip(f"MiDaS could not load, skipping test: {e}")
        
    meta = prior.metadata()
    assert meta["source_type"] == "MiDaS_small"
    assert meta["scale_type"] == "relative_inverse_depth"
    assert meta["metric"] == False

def test_midas_depth_prior_prediction():
    try:
        prior = MiDaSDepthPrior(torch.device('cpu'))
    except Exception as e:
        pytest.skip(f"MiDaS could not load, skipping test: {e}")
        
    dummy_img = torch.rand(1080, 1920, 3)
    res = prior.predict(dummy_img)
    
    assert res.depth.shape == (1080, 1920)
    assert res.confidence.shape == (1080, 1920)
    assert res.scale_type == "relative_inverse_depth"
    assert res.source_type == "MiDaS_small"
    assert "metric" not in res.scale_type
    
def test_midas_depth_prior_uncertainty():
    try:
        prior = MiDaSDepthPrior(torch.device('cpu'))
    except Exception as e:
        pytest.skip(f"MiDaS could not load, skipping test: {e}")
        
    dummy_img = torch.rand(100, 100, 3)
    unc = prior.estimate_uncertainty(dummy_img)
    
    assert unc.shape == (100, 100)
    # By default, MiDaS interface returns 0 (confident)
    assert torch.all(unc == 0)
