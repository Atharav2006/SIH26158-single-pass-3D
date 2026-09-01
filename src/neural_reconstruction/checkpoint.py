import torch
from pathlib import Path
import json

def save_checkpoint(path: Path, step: int, model: torch.nn.Module, optimizer: torch.optim.Optimizer, metrics: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    
    state = {
        'step': step,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'metrics': metrics
    }
    torch.save(state, str(path))
    
    # Save a lightweight json metadata file alongside
    meta_path = path.with_suffix('.json')
    with open(meta_path, 'w') as f:
        json.dump({'step': step, 'metrics': metrics}, f, indent=4)

def load_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    if not path.exists():
        return 0, {}
        
    state = torch.load(str(path), map_location='cpu')
    model.load_state_dict(state['model_state'])
    if optimizer is not None and 'optimizer_state' in state:
        optimizer.load_state_dict(state['optimizer_state'])
        
    return state['step'], state.get('metrics', {})
