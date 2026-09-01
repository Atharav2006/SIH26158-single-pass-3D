import sys
import json
import numpy as np
import cv2
import csv
import torch
from pathlib import Path
from scipy.spatial.transform import Rotation

from src.depth_fusion.camera_preprocessing import CameraPreprocessor
from src.depth_fusion.depth_prior import MiDaSDepthPrior

def unproject_and_project(
    depth_i: np.ndarray, 
    K_rect: np.ndarray, 
    R_wc_i: np.ndarray, 
    C_w_i: np.ndarray,
    depth_j: np.ndarray, 
    R_wc_j: np.ndarray, 
    C_w_j: np.ndarray,
    downsample_factor: int = 4
):
    H, W = depth_i.shape
    y, x = np.mgrid[0:H:downsample_factor, 0:W:downsample_factor]
    u, v = x.flatten(), y.flatten()
    
    Z_i = depth_i[v, u]
    valid_i = Z_i > 0
    u, v, Z_i = u[valid_i], v[valid_i], Z_i[valid_i]
    
    fx, fy = K_rect[0, 0], K_rect[1, 1]
    cx, cy = K_rect[0, 2], K_rect[1, 2]
    
    X_c = (u - cx) * Z_i / fx
    Y_c = (v - cy) * Z_i / fy
    pts_c_i = np.stack([X_c, Y_c, Z_i], axis=-1)
    
    pts_w = pts_c_i @ R_wc_i.T + C_w_i.reshape(1, 3)
    pts_c_j = (pts_w - C_w_j.reshape(1, 3)) @ R_wc_j
    
    Z_c_j = pts_c_j[:, 2]
    valid_z = Z_c_j > 1e-6
    pts_c_j = pts_c_j[valid_z]
    Z_i_valid = Z_i[valid_z]
    
    u_j = (pts_c_j[:, 0] * fx / pts_c_j[:, 2]) + cx
    v_j = (pts_c_j[:, 1] * fy / pts_c_j[:, 2]) + cy
    
    u_j_int = np.round(u_j).astype(int)
    v_j_int = np.round(v_j).astype(int)
    
    in_bounds = (u_j_int >= 0) & (u_j_int < W) & (v_j_int >= 0) & (v_j_int < H)
    u_j_int, v_j_int = u_j_int[in_bounds], v_j_int[in_bounds]
    Z_i_valid = Z_i_valid[in_bounds]
    
    D_j_vals = depth_j[v_j_int, u_j_int]
    valid_j = D_j_vals > 0
    D_j_vals = D_j_vals[valid_j]
    D_i_vals = Z_i_valid[valid_j]
    
    return D_i_vals, D_j_vals

def run_crossframe_validation():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
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
    
    # Load b5_global_gauge.json
    with open("outputs/reports/zurich_mav/b5/b5_global_gauge.json") as f:
        gauge_data = json.load(f)
    
    global_scales = gauge_data["global_scales"]
    global_shifts = gauge_data["global_shifts"]
    
    test_pairs = [(10, 11), (50, 51), (100, 101), (150, 151), (200, 201), (250, 251), (300, 301)]
    base_dir = Path("D:/SIH26158/datasets/zurich_mav/AGZ_subset")
    
    results = []
    
    for (id_i, id_j) in test_pairs:
        row_i = next(r for r in rows if int(r['imgid']) == id_i)
        row_j = next(r for r in rows if int(r['imgid']) == id_j)
        
        # Load images and poses
        img_i = preprocessor.rectify_image(cv2.imread(str(base_dir / "MAV Images" / f"{id_i:05d}.jpg")))
        img_j = preprocessor.rectify_image(cv2.imread(str(base_dir / "MAV Images" / f"{id_j:05d}.jpg")))
        
        with torch.no_grad():
            D_inv_i = prior.predict(torch.from_numpy(cv2.cvtColor(img_i, cv2.COLOR_BGR2RGB)).float().to(device)/255.0).depth.cpu().numpy()
            D_inv_j = prior.predict(torch.from_numpy(cv2.cvtColor(img_j, cv2.COLOR_BGR2RGB)).float().to(device)/255.0).depth.cpu().numpy()
            
        R_i = Rotation.from_quat([float(row_i['qx']), float(row_i['qy']), float(row_i['qz']), float(row_i['qw'])]).as_matrix()
        C_i = np.array([float(row_i['x']), float(row_i['y']), float(row_i['z'])])
        
        R_j = Rotation.from_quat([float(row_j['qx']), float(row_j['qy']), float(row_j['qz']), float(row_j['qw'])]).as_matrix()
        C_j = np.array([float(row_j['x']), float(row_j['y']), float(row_j['z'])])
        
        # Original Reprojection Residuals
        # Since depth is arbitrary, we compute the affine alignment residual between D_i_vals and D_j_vals
        from src.depth_fusion.global_gauge_alignment import fit_pairwise_affine
        D_i_vals_raw, D_j_vals_raw = unproject_and_project(D_inv_i, preprocessor.K_rect, R_i, C_i, D_inv_j, R_j, C_j)
        
        raw_fit = fit_pairwise_affine(D_i_vals_raw, D_j_vals_raw, np.ones_like(D_i_vals_raw))
        
        # Global Gauge Reprojection
        a_i, b_i = global_scales[str(id_i)], global_shifts[str(id_i)]
        a_j, b_j = global_scales[str(id_j)], global_shifts[str(id_j)]
        
        D_aligned_i = a_i * D_inv_i + b_i
        D_aligned_j = a_j * D_inv_j + b_j
        
        # To strictly test the hypothesis, we unproject the aligned depths and expect them to match identically 1:1!
        # D_aligned_i -> rays -> project -> sample D_aligned_j
        D_i_vals_aligned, D_j_vals_aligned = unproject_and_project(D_aligned_i, preprocessor.K_rect, R_i, C_i, D_aligned_j, R_j, C_j)
        
        # If the global gauge worked, D_i_vals_aligned should equal D_j_vals_aligned directly (slope 1, intercept 0).
        # We compute the absolute residual assuming y = x
        gauge_residuals = np.abs(D_i_vals_aligned - D_j_vals_aligned)
        mae_gauge = float(np.mean(gauge_residuals))
        
        results.append({
            "pair": [id_i, id_j],
            "valid_correspondences": len(D_i_vals_raw),
            "original_affine_residual": raw_fit["residual"],
            "global_gauge_absolute_residual": mae_gauge,
            "improvement_ratio": raw_fit["residual"] / (mae_gauge + 1e-6)
        })
        
    out_file = Path("outputs/reports/zurich_mav/b5/b5_global_gauge_crossframe_validation.json")
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"Validation saved to {out_file}")

if __name__ == "__main__":
    run_crossframe_validation()
