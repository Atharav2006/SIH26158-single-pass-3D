import pytest
import numpy as np
from src.depth_fusion.global_gauge_alignment import (
    evaluate_representations, GaugeRepresentation
)

def test_representation_selection():
    # Construct mock edges for D_inv and D_rel
    edges_inv = [
        {"correlation": 0.99, "norm_rmse": 0.05, "cond": 1e3, "valid_count": 10000}
    ]
    edges_rel = [
        {"correlation": 0.90, "norm_rmse": 0.20, "cond": 1e5, "valid_count": 9000}
    ]
    
    selected, metrics = evaluate_representations(edges_inv, edges_rel, 2)
    assert selected == GaugeRepresentation.D_INV
    assert metrics["selected"] == "D_inv"

def test_representation_ambiguous():
    edges_inv = [
        {"correlation": 0.95, "norm_rmse": 0.10, "cond": 1e4, "valid_count": 10000}
    ]
    edges_rel = [
        {"correlation": 0.95, "norm_rmse": 0.10, "cond": 1e4, "valid_count": 10000}
    ]
    
    selected, metrics = evaluate_representations(edges_inv, edges_rel, 2)
    assert selected is None
    assert metrics["selected"] == "AMBIGUOUS"
