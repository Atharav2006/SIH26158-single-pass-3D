import torch
from typing import Dict, Any, Tuple

def inverse_to_relative_depth(inverse_depth: torch.Tensor, epsilon: float = 1e-6) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Explicitly converts inverse relative depth (e.g., MiDaS output) to relative depth.
    
    WARNING: The output is NOT metric depth. It remains scale and shift ambiguous.
    It must never be treated as meters until explicitly calibrated.
    """
    
    # Safe inversion
    relative_depth = 1.0 / (inverse_depth + epsilon)
    
    metadata = {
        "representation": "relative_depth",
        "metric": False,
        "source": "inverse_inversion",
        "warning": "DO NOT USE AS METERS"
    }
    
    return relative_depth, metadata
