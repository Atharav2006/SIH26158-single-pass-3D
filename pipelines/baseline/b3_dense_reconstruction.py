import sys
import os
import csv
import json
import time
import shutil
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict

sys.path.insert(0, str(Path("D:/SIH26158-single-pass-3D").resolve()))

from src.reconstruction.dense_mvs import (
    ColmapWorkspace,
    b2_pose_to_colmap_pose,
    colmap_pose_to_b2_pose,
    run_colmap_image_undistorter,
    run_colmap_patch_match_stereo,
    run_colmap_stereo_fusion
)

@dataclass
class B3Diagnostics:
    baseline: str = "B3 Dense Metric 3D Reconstruction"
    dataset: str = "Zurich Urban MAV Dataset"
    image_count: int = 0
    camera_model: str = "FULL_OPENCV"
    coordinate_frame: str = "Metric Local ENU"
    units: str = "meters"
    pose_source: str = "B2"
    dense_method: str = "COLMAP PatchMatch Stereo + Stereo Fusion"
    point_count: int = 0
    bounding_box: dict = None
    metric_scale_validation: dict = None
    camera_pose_consistency: dict = None
    gpu: dict = None
    runtime: dict = None
    smoke_test: dict = None
    validation: dict = None
    status: str = "PENDING"

def setup_b3_workspace(b3_dir: Path, source_images_dir: Path, images: list):
    """
    Copies images into the B3 workspace so COLMAP can work on them without polluting B0/B1/B2.
    """
    ws = ColmapWorkspace(b3_dir)
    print("Copying images to B3 workspace...")
    for img in images:
        src = source_images_dir / img["filename"]
        dst = ws.images_dir / img["filename"]
        if src.exists():
            shutil.copy2(src, dst)
        else:
            raise FileNotFoundError(f"Image not found: {src}")
    return ws

def run_camera_consistency_check(b2_poses: list, ws: ColmapWorkspace) -> dict:
    recovered = ws.read_images_txt()
    assert len(recovered) == len(b2_poses)
    
    rec_dict = {p["imgid"]: p for p in recovered}
    
    errors = []
    for orig in b2_poses:
        rec = rec_dict[orig["imgid"]]
        err = np.linalg.norm(orig["c_w"] - rec["c_w"])
        errors.append(err)
        
    errors = np.array(errors)
    return {
        "rmse_m": float(np.sqrt(np.mean(errors**2))),
        "mean_m": float(np.mean(errors)),
        "median_m": float(np.median(errors)),
        "max_m": float(np.max(errors))
    }

def run_b3_pipeline(smoke_test: bool = True):
    t_start = time.time()
    diags = B3Diagnostics()
    diags.runtime = {}
    
    reports_dir = Path("outputs/reports/zurich_mav/b3")
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load existing artifacts
    print("\n--- [1] Loading B2 Artifacts ---")
    b2_csv = Path("outputs/reports/zurich_mav/b2/b2_fused_trajectory.csv")
    b2_poses = []
    with open(b2_csv, "r") as f:
        for r in csv.DictReader(f):
            b2_poses.append({
                "imgid": int(r["imgid"]),
                "filename": r["timestamp"] + ".jpg", # Wait, we need actual filenames
                "timestamp": float(r["timestamp"]),
                "c_w": np.array([float(r["x"]), float(r["y"]), float(r["z"])]),
                "q_wc": np.array([float(r["qx"]), float(r["qy"]), float(r["qz"]), float(r["qw"])])
            })
            
    # Load actual image metadata to get accurate filenames
    img_meta_csv = Path("outputs/reports/zurich_mav/images.csv")
    img_dict = {}
    with open(img_meta_csv, "r") as f:
        for r in csv.DictReader(f):
            img_dict[int(r["imgid"])] = r["filename"]
            
    for p in b2_poses:
        p["filename"] = img_dict[p["imgid"]]
        
    diags.image_count = len(b2_poses)
    
    # Select images based on run mode
    if smoke_test:
        print(">>> SMOKE TEST MODE: Selecting 10 spaced images for sufficient baseline")
        indices = np.linspace(0, len(b2_poses)-1, 10, dtype=int)
        run_poses = [b2_poses[i] for i in indices]
        ws_dir = Path("colmap_workspace/zurich_mav_b3_smoke")
    else:
        print(">>> FULL RECONSTRUCTION MODE: Using all 350 images")
        run_poses = b2_poses
        ws_dir = Path("colmap_workspace/zurich_mav_b3_full")
        
    # 2. Setup Workspace
    print("\n--- [2] Setting up COLMAP Workspace ---")
    if ws_dir.exists():
        shutil.rmtree(ws_dir)
        
    ws = setup_b3_workspace(ws_dir, Path("D:/SIH26158/datasets/zurich_mav/AGZ_subset/MAV Images/"), run_poses)
    
    # Intrinsic parameters from B0
    camera_model = "FULL_OPENCV"
    w, h = 1920, 1080
    params = [893.3901, 898.3265, 951.1310, 555.1335, -0.2805, 0.1158, -0.00098, 0.000158, -0.0270, 0.0, 0.0, 0.0]
    
    ws.write_cameras_txt(camera_model, w, h, params)
    ws.write_images_txt(run_poses)
    ws.write_points3D_txt()
    
    # 3. Validation
    print("\n--- [3] Validating exported camera centers ---")
    consistency = run_camera_consistency_check(run_poses, ws)
    print(f"  Consistency Max Error: {consistency['max_m']:.6e} m")
    diags.camera_pose_consistency = consistency
    if consistency["max_m"] > 1e-3:
        diags.status = "VALIDATION FAILED: Camera center extraction mismatch"
        print("FAILURE: " + diags.status)
        return
        
    diags.runtime["preflight_and_export_s"] = time.time() - t_start
    t_start_undistort = time.time()
    
    # 4. Undistortion
    print("\n--- [4] Running Image Undistortion ---")
    res_undistort = run_colmap_image_undistorter(ws.workspace_dir, ws.images_dir)
    if res_undistort.returncode != 0:
        diags.status = "FAILED: Undistortion"
        print(res_undistort.stderr)
        return
    diags.runtime["undistort_s"] = time.time() - t_start_undistort
    
    # 4b. Generate custom patch-match.cfg (Image Undistorter writes empty sources without sparse points)
    cfg_path = ws.workspace_dir / "dense" / "stereo" / "patch-match.cfg"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg_path, "w", newline="\n") as f:
        for i, p in enumerate(run_poses):
            f.write(f"{p['filename']}\n")
            source_imgs = []
            for j in range(max(0, i-5), min(len(run_poses), i+6)):
                if i != j:
                    source_imgs.append(run_poses[j]['filename'])
            f.write(", ".join(source_imgs) + "\n")
            
    # 5. PatchMatch Stereo
    t_start_pm = time.time()
    print("\n--- [5] Running PatchMatch Stereo ---")
    # max_image_size = 960 (half res) to ensure safety on 4.29GB VRAM
    res_pm = run_colmap_patch_match_stereo(ws.workspace_dir, gpu_index="0", max_image_size=960)
    if res_pm.returncode != 0:
        diags.status = "FAILED: PatchMatch Stereo"
        print(res_pm.stderr)
        return
    diags.runtime["patchmatch_s"] = time.time() - t_start_pm
    
    # 6. Stereo Fusion
    t_start_fusion = time.time()
    print("\n--- [6] Running Stereo Fusion ---")
    ply_path = reports_dir / ("fused_smoke.ply" if smoke_test else "fused.ply")
    res_fusion = run_colmap_stereo_fusion(ws.workspace_dir, ply_path)
    if res_fusion.returncode != 0:
        diags.status = "FAILED: Stereo Fusion"
        print(res_fusion.stderr)
        return
    diags.runtime["fusion_s"] = time.time() - t_start_fusion
    
    # 7. Basic PLY validation
    if not ply_path.exists():
        diags.status = "B3_RECONSTRUCTION_FAILED: PLY file not created"
        print("FAILURE: " + diags.status)
        return
        
    import open3d as o3d
    try:
        pcd = o3d.io.read_point_cloud(str(ply_path))
        pts = np.asarray(pcd.points)
        diags.point_count = len(pts)
        if diags.point_count == 0:
            diags.status = "B3_RECONSTRUCTION_FAILED: 0 points fused (Degenerate baseline or pose noise)"
            print("FAILURE: " + diags.status)
        else:
            diags.status = "PASS"
    except Exception as e:
        diags.status = f"B3_RECONSTRUCTION_FAILED: Could not read PLY: {e}"
        print("FAILURE: " + diags.status)
    diags.runtime["total_reconstruction_s"] = time.time() - t_start
    
    diag_path = reports_dir / ("b3_smoke_diagnostics.json" if smoke_test else "b3_reconstruction_diagnostics.json")
    with open(diag_path, "w") as f:
        json.dump(asdict(diags), f, indent=4)
        
    print(f"\nReconstruction complete. PLY: {ply_path}")
    print(f"Total time: {diags.runtime['total_reconstruction_s']:.2f} s")
    
if __name__ == "__main__":
    mode = "smoke"
    if len(sys.argv) > 1 and sys.argv[1] == "--full":
        mode = "full"
        
    run_b3_pipeline(smoke_test=(mode == "smoke"))
