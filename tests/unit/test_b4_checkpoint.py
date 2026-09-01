import pytest
import torch
import torch.nn as nn
from pathlib import Path
from src.neural_reconstruction.checkpoint import save_checkpoint, load_checkpoint

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)

def test_checkpoint_save_load(tmp_path):
    model = DummyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    
    # Modify weights to something specific
    with torch.no_grad():
        model.fc.weight.fill_(42.0)
        
    ckpt_path = tmp_path / "ckpt.pt"
    save_checkpoint(ckpt_path, 100, model, optimizer, {"loss": 0.5})
    
    # Create new model
    model2 = DummyModel()
    opt2 = torch.optim.Adam(model2.parameters(), lr=0.1)
    
    step, metrics = load_checkpoint(ckpt_path, model2, opt2)
    
    assert step == 100
    assert metrics["loss"] == 0.5
    assert torch.allclose(model2.fc.weight, torch.tensor(42.0))
