import pytest
import numpy as np
from src.depth_fusion.global_gauge_alignment import GlobalGaugeSolver

def test_empty_graph():
    solver = GlobalGaugeSolver(ref_frame=0)
    a, b, status = solver.solve([], 0)
    assert status == "ZERO_FRAMES"
    assert a is None

def test_disconnected_graph():
    edges = [
        {'i': 0, 'j': 1, 'a': 1.0, 'b': 0.0, 'w': 1.0},
        # Node 2 and 3 are isolated from 0 and 1
        {'i': 2, 'j': 3, 'a': 1.0, 'b': 0.0, 'w': 1.0}
    ]
    solver = GlobalGaugeSolver(ref_frame=0)
    a, b, status = solver.solve(edges, 4)
    assert status == "DISCONNECTED_GRAPH"
    assert a is None

def test_isolated_node():
    edges = [
        {'i': 0, 'j': 1, 'a': 1.0, 'b': 0.0, 'w': 1.0}
    ]
    # Node 2 is completely isolated
    solver = GlobalGaugeSolver(ref_frame=0)
    a, b, status = solver.solve(edges, 3)
    assert status == "DISCONNECTED_GRAPH"
    assert a is None

def test_invalid_affine_scale():
    edges = [
        {'i': 0, 'j': 1, 'a': -1.0, 'b': 0.0, 'w': 1.0}
    ]
    solver = GlobalGaugeSolver(ref_frame=0)
    a, b, status = solver.solve(edges, 2)
    assert status == "INVALID_AFFINE_PARAMETER"
    assert a is None

    edges[0]['a'] = 0.0
    a, b, status = solver.solve(edges, 2)
    assert status == "INVALID_AFFINE_PARAMETER"
    
def test_rank_deficiency_handled():
    # We can force a singular matrix by providing a zero-weight edge (or some other structure if w matters)
    # Wait, with mask removing ref_frame and single component, it's structurally full rank.
    # If we pass w=0, the graph is effectively disconnected in the linear system but structurally connected.
    # Let's see if our AtW.dot(A) catches it.
    edges = [
        {'i': 0, 'j': 1, 'a': 1.0, 'b': 0.0, 'w': 0.0}
    ]
    solver = GlobalGaugeSolver(ref_frame=0)
    a, b, status = solver.solve(edges, 2)
    assert status == "RANK_DEFICIENCY"
