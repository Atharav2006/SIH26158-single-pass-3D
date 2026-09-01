"""
SIH26158 - B5.2 GLOBAL RELATIVE-DEPTH GAUGE ALIGNMENT EXPERIMENT
"""

import sys
from pathlib import Path
import json
import numpy as np
import csv
import torch
import cv2
from scipy.spatial.transform import Rotation

from src.depth_fusion.depth_prior import MiDaSDepthPrior
from src.depth_fusion.camera_preprocessing import CameraPreprocessor
from src.depth_fusion.global_gauge_alignment import align_sequence, GaugeRepresentation

def run_zurich_gauge_alignment(base_dir="D:/SIH26158/datasets/zurich_mav/AGZ_subset"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    out_dir = Path("outputs/reports/zurich_mav/b5")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load Calib
    calib_file = Path("outputs/reports/zurich_mav/b0/reconstruction_summary.json")
    with open(calib_file) as f:
        calib_data = json.load(f)["camera_calibration"]
        
    preprocessor = CameraPreprocessor(calib_data)
    prior = MiDaSDepthPrior(device)
    
    # Load Trajectory
    traj_file = Path("outputs/reports/zurich_mav/b2/b2_fused_trajectory.csv")
    
    rows = []
    with open(traj_file, 'r') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
            
    # Sort by imgid
    rows.sort(key=lambda row: int(row['imgid']))
    
    depth_sequence = {}
    print(f"Loading {len(rows)} frames...")
    
    # Define overlap graph (i to i+1)
    overlap_graph = []
    
    prev_frame_id = None
    
    for i, row in enumerate(rows):
        frame_id = int(row['imgid'])
        img_path = Path(base_dir) / "MAV Images" / f"{frame_id:05d}.jpg"
        
        if not img_path.exists():
            continue
            
        # Image
        img_bgr = cv2.imread(str(img_path))
        img_rect = preprocessor.rectify_image(img_bgr)
        
        # MiDaS
        img_rgb = cv2.cvtColor(img_rect, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).float().to(device) / 255.0
        with torch.no_grad():
            pred = prior.predict(img_tensor)
            D_inv = pred.depth.cpu().numpy()
            
        # Pose
        quat = [float(row['qx']), float(row['qy']), float(row['qz']), float(row['qw'])]
        R_wc = Rotation.from_quat(quat).as_matrix()
        C_w = np.array([float(row['x']), float(row['y']), float(row['z'])])
        
        depth_sequence[frame_id] = {
            "D_inv": D_inv,
            "conf": np.ones_like(D_inv), # Uniform conf for alignment as per B5.2
            "K_rect": preprocessor.K_rect,
            "R_wc": R_wc,
            "C_w": C_w
        }
        
        if prev_frame_id is not None:
            overlap_graph.append((prev_frame_id, frame_id))
            
        prev_frame_id = frame_id
        
        if len(depth_sequence) % 50 == 0:
            print(f"Loaded {len(depth_sequence)} frames")
            
    print("Running Global Gauge Alignment (D_inv)...")
    res_inv = align_sequence(depth_sequence, overlap_graph, reference_frame=1, representation=GaugeRepresentation.D_INV)
    
    print("Running Global Gauge Alignment (D_rel)...")
    res_rel = align_sequence(depth_sequence, overlap_graph, reference_frame=1, representation=GaugeRepresentation.D_REL)
    
    # Use the evaluate_representations output
    from src.depth_fusion.global_gauge_alignment import evaluate_representations
    # We already have edges extracted inside align_sequence, but wait, align_sequence does selection internally if representation=None
    
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
        
    # Write JSON Artifact
    output_json = out_dir / "b5_global_gauge.json"
    with open(output_json, 'w') as f:
        # Save necessary scales
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
    run_zurich_gauge_alignment()
