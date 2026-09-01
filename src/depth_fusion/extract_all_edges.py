import sys
from pathlib import Path
import json
import numpy as np
import cv2
import csv
import torch
from scipy.spatial.transform import Rotation

from src.depth_fusion.camera_preprocessing import CameraPreprocessor
from src.depth_fusion.depth_prior import MiDaSDepthPrior
from src.depth_fusion.global_gauge_alignment import pose_aware_correspondences, fit_pairwise_affine

def extract_all_edges():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    calib_file = Path("outputs/reports/zurich_mav/b0/reconstruction_summary.json")
    with open(calib_file) as f:
        calib_data = json.load(f)["camera_calibration"]
        
    preprocessor = CameraPreprocessor(calib_data)
    prior = MiDaSDepthPrior(device)
    
    traj_file = Path("outputs/reports/zurich_mav/b2/b2_fused_trajectory.csv")
    rows = []
    with open(traj_file, 'r') as f:
        for r in csv.DictReader(f):
            rows.append(r)
            
    rows.sort(key=lambda row: int(row['imgid']))
    
    depth_sequence = {}
    base_dir = Path("D:/SIH26158/datasets/zurich_mav/AGZ_subset")
    
    for row in rows:
        frame_id = int(row['imgid'])
        img_path = base_dir / "MAV Images" / f"{frame_id:05d}.jpg"
        if not img_path.exists(): continue
            
        img_bgr = cv2.imread(str(img_path))
        img_rect = preprocessor.rectify_image(img_bgr)
        img_rgb = cv2.cvtColor(img_rect, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).float().to(device) / 255.0
        
        with torch.no_grad():
            pred = prior.predict(img_tensor)
            D_inv = pred.depth.cpu().numpy()
            
        quat = [float(row['qx']), float(row['qy']), float(row['qz']), float(row['qw'])]
        R_wc = Rotation.from_quat(quat).as_matrix()
        C_w = np.array([float(row['x']), float(row['y']), float(row['z'])])
        
        depth_sequence[frame_id] = {
            "D_inv": D_inv,
            "R_wc": R_wc,
            "C_w": C_w
        }
        
        if len(depth_sequence) % 50 == 0:
            print(f"Loaded {len(depth_sequence)} frames")
            
    print("Computing edges...")
    frame_ids = sorted(list(depth_sequence.keys()))
    id_to_idx = {fid: i for i, fid in enumerate(frame_ids)}
    
    edges = []
    for step in [1, 2]:
        for i in range(len(frame_ids) - step):
            id_i, id_j = frame_ids[i], frame_ids[i+step]
            
            di, dj = depth_sequence[id_i], depth_sequence[id_j]
            
            D_i, D_j, w = pose_aware_correspondences(
                di["D_inv"], preprocessor.K_rect, di["R_wc"], di["C_w"],
                dj["D_inv"], dj["R_wc"], dj["C_w"],
                None, None, downsample_factor=4
            )
            
            fit = fit_pairwise_affine(D_i, D_j, w)
            if fit["status"] == "SUCCESS":
                edges.append({
                    "id_i": id_i,
                    "id_j": id_j,
                    "idx_i": id_to_idx[id_i],
                    "idx_j": id_to_idx[id_j],
                    "a": fit["a"],
                    "b": fit["b"],
                    "w": fit["valid_count"],
                    "residual": fit["residual"],
                    "correlation": fit["correlation"]
                })
            
    out_file = Path("outputs/reports/zurich_mav/b5/b5_3_pairwise_edges.json")
    with open(out_file, 'w') as f:
        json.dump(edges, f, indent=4)
        
    print(f"Extracted {len(edges)} edges to {out_file}")

if __name__ == "__main__":
    extract_all_edges()
