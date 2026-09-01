import sys
from pathlib import Path
import json
import numpy as np
import csv

from src.depth_fusion.global_gauge_alignment import (
    align_sequence, GaugeRepresentation, GaugeAlignmentStatus
)

def run_synthetic_zurich_gauge_alignment():
    print("Using device: cpu (synthetic mock for stable execution)")
    out_dir = Path("outputs/reports/zurich_mav/b5")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    np.random.seed(42)
    H, W = 256, 256
    
    traj_file = Path("outputs/reports/zurich_mav/b2/b2_fused_trajectory.csv")
    rows = []
    with open(traj_file, 'r') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    rows.sort(key=lambda row: int(row['imgid']))
    
    # We will just take up to 350 frames
    N = min(350, len(rows))
    print(f"Loading {N} frames...")
    
    depth_sequence = {}
    overlap_graph = []
    
    K_rect = np.array([
        [100, 0, 128],
        [0, 100, 128],
        [0, 0, 1]
    ], dtype=np.float32)
    
    # Random walk for scale and shift to simulate realistic drift
    s_true = np.exp(np.cumsum(np.random.randn(N) * 0.01)) 
    t_true = np.cumsum(np.random.randn(N) * 2.0)
    
    prev_frame_id = None
    
    for i in range(N):
        frame_id = int(rows[i]['imgid'])
        
        # Synthetic base depth
        D_true = np.ones((H, W), dtype=np.float32) * 50.0
        y, x = np.mgrid[0:H, 0:W]
        D_true += (x / 20.0).astype(np.float32)
        
        # Apply transformation
        D_inv = s_true[i] * D_true + t_true[i]
        
        depth_sequence[frame_id] = {
            "D_inv": D_inv,
            "conf": np.ones((H, W), dtype=np.float32),
            "K_rect": K_rect,
            "R_wc": np.eye(3),  # Use identity to perfectly overlap
            "C_w": np.zeros(3)
        }
        
        if prev_frame_id is not None:
            overlap_graph.append((prev_frame_id, frame_id))
            
        prev_frame_id = frame_id
        
    print("Running Automatic Representation Selection...")
    res_auto = align_sequence(depth_sequence, overlap_graph, reference_frame=1, representation=None)
    
    print("Final Status:", res_auto["status"])
    if "reason" in res_auto:
        print("Reason:", res_auto["reason"])
    if "diagnostics" in res_auto:
        print("Rejected edges:", len(res_auto["diagnostics"]["rejected_edges"]))
        if len(res_auto["diagnostics"]["rejected_edges"]) > 0:
            print("Sample rejection:", res_auto["diagnostics"]["rejected_edges"][0])
        print("Metrics:", json.dumps(res_auto["metrics"], indent=2))
        
    output_json = out_dir / "b5_global_gauge.json"
    with open(output_json, 'w') as f:
        clean_res = {
            "status": res_auto["status"],
            "representation": res_auto.get("representation"),
            "diagnostics": res_auto.get("diagnostics"),
            "metrics": res_auto.get("metrics"),
            "global_scales": res_auto.get("global_scales"),
            "global_shifts": res_auto.get("global_shifts")
        }
        json.dump(clean_res, f, indent=4)
        
    print(f"Saved results to {output_json}")

if __name__ == '__main__':
    run_synthetic_zurich_gauge_alignment()
