import sys
import json
import numpy as np
import cv2
import csv
import torch
from pathlib import Path

def run_depth_stats():
    with open("outputs/reports/zurich_mav/b5/b5_global_gauge.json") as f:
        gauge_data = json.load(f)
        
    global_scales = gauge_data["global_scales"]
    global_shifts = gauge_data["global_shifts"]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    from src.depth_fusion.depth_prior import MiDaSDepthPrior
    from src.depth_fusion.camera_preprocessing import CameraPreprocessor
    
    calib_file = Path("outputs/reports/zurich_mav/b0/reconstruction_summary.json")
    with open(calib_file) as f:
        calib_data = json.load(f)["camera_calibration"]
    preprocessor = CameraPreprocessor(calib_data)
    prior = MiDaSDepthPrior(device)
    
    base_dir = Path("D:/SIH26158/datasets/zurich_mav/AGZ_subset")
    
    results = {
        "representation": "RELATIVE_INVERSE_DEPTH_GLOBAL_GAUGE",
        "metric": False,
        "frames_evaluated": 0,
        "invalid_negative_pixels_ratio": 0.0,
        "mean_magnitude": 0.0,
        "max_magnitude": 0.0
    }
    
    total_pixels = 0
    negative_pixels = 0
    magnitudes = []
    
    for i in range(1, 351, 10):
        frame_id = i
        img_path = base_dir / "MAV Images" / f"{frame_id:05d}.jpg"
        if not img_path.exists(): continue
            
        img_rect = preprocessor.rectify_image(cv2.imread(str(img_path)))
        with torch.no_grad():
            D_inv = prior.predict(torch.from_numpy(cv2.cvtColor(img_rect, cv2.COLOR_BGR2RGB)).float().to(device)/255.0).depth.cpu().numpy()
            
        a_i, b_i = global_scales[str(frame_id)], global_shifts[str(frame_id)]
        D_aligned = a_i * D_inv + b_i
        
        total_pixels += D_aligned.size
        negative_pixels += np.sum(D_aligned <= 0)
        magnitudes.append(np.mean(np.abs(D_aligned)))
        
        results["max_magnitude"] = max(results["max_magnitude"], float(np.max(np.abs(D_aligned))))
        results["frames_evaluated"] += 1
        
    results["invalid_negative_pixels_ratio"] = float(negative_pixels / total_pixels)
    results["mean_magnitude"] = float(np.mean(magnitudes))
    
    out_file = Path("outputs/reports/zurich_mav/b5/b5_global_gauge_depth_statistics.json")
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"Stats saved to {out_file}")

if __name__ == "__main__":
    run_depth_stats()
