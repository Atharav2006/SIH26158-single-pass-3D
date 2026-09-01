import sys
import json
import numpy as np
from pathlib import Path

from src.depth_fusion.global_gauge_alignment import GlobalGaugeSolver

def run_holdout_validation():
    edges_file = Path("outputs/reports/zurich_mav/b5/b5_3_pairwise_edges.json")
    if not edges_file.exists():
        print("Edges not extracted yet!")
        return
        
    with open(edges_file, 'r') as f:
        edges = json.load(f)
        
    # Standardize dictionary keys for solver
    for e in edges:
        e['i'] = e['idx_i']
        e['j'] = e['idx_j']
        
    num_frames = 350
    solver = GlobalGaugeSolver(ref_frame=0)
    
    # Hold-out specified edges
    holdout_pairs = [(31, 32), (91, 92), (151, 152), (211, 212), (271, 272), (331, 332)]
    
    results = []
    
    for (hi, hj) in holdout_pairs:
        # Filter out this edge
        active_edges = [e for e in edges if not (e['id_i'] == hi and e['id_j'] == hj)]
        holdout_edge = next((e for e in edges if e['id_i'] == hi and e['id_j'] == hj), None)
        
        if not holdout_edge:
            print(f"Edge {hi}->{hj} not found in valid edges.")
            continue
            
        a_global, b_global, err = solver.solve(active_edges, num_frames)
        if err != "SUCCESS":
            print(f"Failed to solve for {hi}->{hj}: {err}")
            continue
            
        # Predict the held-out relationship:
        # D_hj = a_hj_pred * D_hi + b_hj_pred
        # a_hj_pred = a_hi / a_hj
        # b_hj_pred = (b_hi - b_hj) / a_hj
        
        a_i = a_global[holdout_edge['idx_i']]
        a_j = a_global[holdout_edge['idx_j']]
        b_i = b_global[holdout_edge['idx_i']]
        b_j = b_global[holdout_edge['idx_j']]
        
        a_pred = a_i / a_j
        b_pred = (b_i - b_j) / a_j
        
        # Compare with the true optimized local affine relation
        a_true = holdout_edge['a']
        b_true = holdout_edge['b']
        
        a_err = abs(a_pred - a_true)
        b_err = abs(b_pred - b_true)
        
        results.append({
            "held_out_edge": [hi, hj],
            "a_true": a_true,
            "b_true": b_true,
            "a_pred": a_pred,
            "b_pred": b_pred,
            "a_error": a_err,
            "b_error": b_err
        })
        
    out_file = Path("outputs/reports/zurich_mav/b5/b5_global_gauge_holdout_validation.json")
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"Holdout validation complete. Saved to {out_file}")

if __name__ == "__main__":
    run_holdout_validation()
