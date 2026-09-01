import torch
import pytest
from src.neural_reconstruction.depth_loss import scale_invariant_depth_loss, compute_scale_and_shift

def test_scale_invariant_loss():
    # If prediction is an exact linear scaling of target, loss should be ~0
    target = torch.rand(1024, 1) * 10 + 5.0
    
    # Scale and shift target
    s_true = 2.5
    t_true = -1.5
    prediction = (target - t_true) / s_true # So that s * prediction + t = target
    
    s, t = compute_scale_and_shift(prediction, target)
    
    # Verify exact recovery (within float precision)
    # The least squares or mean/std alignment should recover s, t perfectly.
    assert torch.isclose(s, torch.tensor(s_true), atol=1e-4)
    assert torch.isclose(t, torch.tensor(t_true), atol=1e-4)
    
    loss = scale_invariant_depth_loss(prediction, target)
    assert loss.item() < 1e-5

def test_scale_invariant_loss_with_mask():
    target = torch.rand(1024, 1)
    prediction = target * 3.0 + 1.0
    
    mask = torch.rand(1024, 1) > 0.5
    
    # Add huge noise to masked out regions
    prediction[~mask] += 1000.0
    
    s, t = compute_scale_and_shift(prediction, target, mask)
    loss = scale_invariant_depth_loss(prediction, target, mask)
    
    assert loss.item() < 1e-5
