import argparse
import sys
import os
import csv
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from src.reconstruction.colmap_wrapper import COLMAPRunner, find_colmap_executable
from src.reconstruction.colmap_parser import (
    parse_colmap_cameras_txt,
    parse_colmap_images_txt,
    parse_colmap_points3D_txt,
    compute_colmap_metrics
)
from src.visualization.colmap_visualizer import (
    render_colmap_trajectory_plot,
    render_colmap_sparse_pointcloud_plot
)

def format_camera_params_for_colmap(camera_json_path: Path) -> Tuple[str, str]:
    """
    Format camera intrinsics for COLMAP ImageReader.
    Model: OPENCV (8 parameters: fx, fy, cx, cy, k1, k2, p1, p2)
    """
    with open(camera_json_path, "r", encoding="utf-8") as f:
        cam_meta = json.load(f)

    fx = cam_meta["fx"]
    fy = cam_meta["fy"]
    cx = cam_meta["cx"]
    cy = cam_meta["cy"]
    dist = cam_meta.get("distortion_parameters_if_available", [0.0, 0.0, 0.0, 0.0, 0.0])
    k1, k2, p1, p2 = dist[0], dist[1], dist[2], dist[3]

    param_str = f"{fx:.6f},{fy:.6f},{cx:.6f},{cy:.6f},{k1:.6f},{k2:.6f},{p1:.6f},{p2:.6f}"
    return "OPENCV", param_str

def main():
    parser = argparse.ArgumentParser(description="SIH26158 Baseline: Classical COLMAP Structure-from-Motion (b0).")
    parser.add_argument("--image-dir", "-i", default=r"D:\SIH26158\datasets\zurich_mav\AGZ_subset\MAV Images", help="Path to raw image directory.")
    parser.add_argument("--metadata-dir", "-m", default="outputs/reports/zurich_mav", help="Path to directory containing images.csv and camera.json.")
    parser.add_argument("--workspace-dir", "-w", default=r"D:\SIH26158\colmap_workspace\zurich_mav_b0", help="COLMAP workspace directory.")
    parser.add_argument("--output-dir", "-o", default="outputs/reports/zurich_mav/b0", help="Directory to save baseline reports and metrics.")
    parser.add_argument("--max-features", type=int, default=8192, help="Max SIFT features per image.")
    parser.add_argument("--use-gpu", type=int, default=1, help="Use CUDA GPU for extraction and matching (1/0).")

    args = parser.parse_args()

    image_dir = Path(args.image_dir).resolve()
    meta_dir = Path(args.metadata_dir).resolve()
    workspace_dir = Path(args.workspace_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    reports_parent = out_dir.parent

    workspace_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SIH26158: Classical COLMAP Structure-from-Motion Baseline (b0)")
    print("=" * 60)
    print(f"Image Directory:    {image_dir}")
    print(f"Metadata Directory: {meta_dir}")
    print(f"COLMAP Workspace:   {workspace_dir}")
    print(f"Output Directory:   {out_dir}")

    # 1. Validate Input Data
    if not image_dir.is_dir():
        print(f"[ERROR] Image directory not found: {image_dir}", file=sys.stderr)
        sys.exit(1)

    images_csv = meta_dir / "images.csv"
    camera_json = meta_dir / "camera.json"
    gt_assoc_csv = meta_dir / "image_groundtruth_associations.csv"

    if not camera_json.is_file():
        print(f"[ERROR] Camera calibration metadata not found: {camera_json}", file=sys.stderr)
        sys.exit(1)

    # 2. Prepare Camera Model & Parameters
    camera_model, camera_params = format_camera_params_for_colmap(camera_json)
    print(f"\nCamera Model: {camera_model}")
    print(f"Camera Parameters (fx,fy,cx,cy,k1,k2,p1,p2): {camera_params}")

    # 3. Initialize COLMAP Runner
    try:
        colmap_bin = find_colmap_executable()
        print(f"COLMAP Binary: {colmap_bin}")
    except Exception as e:
        print(f"[ERROR] Failed to find COLMAP executable: {e}", file=sys.stderr)
        sys.exit(1)

    runner = COLMAPRunner(colmap_bin=colmap_bin, workspace_dir=workspace_dir)
    database_path = workspace_dir / "database.db"

    # Reset existing database if clean run is requested
    if database_path.exists():
        try:
            database_path.unlink()
        except Exception:
            pass

    timing = {}
    total_start = time.perf_counter()

    # 4. Feature Extraction (SIFT GPU)
    print("\n--- 1. Feature Extraction (SIFT) ---")
    code_ext, elapsed_ext, log_ext = runner.extract_features(
        image_path=image_dir,
        database_path=database_path,
        camera_model=camera_model,
        camera_params=camera_params,
        single_camera=True,
        max_num_features=args.max_features,
        use_gpu=bool(args.use_gpu)
    )
    timing["feature_extraction_seconds"] = round(elapsed_ext, 2)
    print(f"Feature Extraction completed in {elapsed_ext:.2f}s (Log: {log_ext.name}, Exit: {code_ext})")
    if code_ext != 0:
        print(f"[ERROR] Feature extraction failed.", file=sys.stderr)
        sys.exit(1)

    # 5. Feature Matching (Exhaustive GPU)
    print("\n--- 2. Feature Matching (Exhaustive) ---")
    code_mat, elapsed_mat, log_mat = runner.match_exhaustive(
        database_path=database_path,
        use_gpu=bool(args.use_gpu)
    )
    timing["feature_matching_seconds"] = round(elapsed_mat, 2)
    print(f"Exhaustive Matching completed in {elapsed_mat:.2f}s (Log: {log_mat.name}, Exit: {code_mat})")
    if code_mat != 0:
        print(f"[ERROR] Feature matching failed.", file=sys.stderr)
        sys.exit(1)

    # 6. Sparse Reconstruction (Incremental Mapper)
    print("\n--- 3. Sparse Reconstruction (Mapper) ---")
    sparse_dir = workspace_dir / "sparse"
    code_map, elapsed_map, log_map = runner.run_mapper(
        image_path=image_dir,
        database_path=database_path,
        output_path=sparse_dir
    )
    timing["mapping_seconds"] = round(elapsed_map, 2)
    print(f"Mapper completed in {elapsed_map:.2f}s (Log: {log_map.name}, Exit: {code_map})")
    if code_map != 0:
        print(f"[ERROR] Mapping failed.", file=sys.stderr)
        sys.exit(1)

    # Check for reconstructed model folder (sparse/0)
    model_0_dir = sparse_dir / "0"
    if not model_0_dir.exists():
        # Check if model files are in sparse/ directly
        if (sparse_dir / "cameras.bin").exists() or (sparse_dir / "cameras.txt").exists():
            model_0_dir = sparse_dir
        else:
            print("[ERROR] No reconstructed sub-model found under sparse/ directory.", file=sys.stderr)
            sys.exit(1)

    # 7. Convert Model to TXT format
    txt_model_dir = workspace_dir / "sparse_txt"
    txt_model_dir.mkdir(parents=True, exist_ok=True)
    runner.convert_model(input_path=model_0_dir, output_path=txt_model_dir, output_type="TXT")

    # 8. Parse Reconstructed Model
    cameras = parse_colmap_cameras_txt(txt_model_dir / "cameras.txt")
    registered_images = parse_colmap_images_txt(txt_model_dir / "images.txt")
    sparse_points = parse_colmap_points3D_txt(txt_model_dir / "points3D.txt")

    total_duration = time.perf_counter() - total_start
    timing["total_runtime_seconds"] = round(total_duration, 2)

    # Count total input images from images.csv or directory
    total_images_count = 350
    if images_csv.exists():
        with open(images_csv, "r", encoding="utf-8") as f:
            total_images_count = len(list(csv.DictReader(f)))
    else:
        total_images_count = len(list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")))

    # 9. Compute Metrics
    metrics = compute_colmap_metrics(
        total_images_count=total_images_count,
        registered_images=registered_images,
        sparse_points=sparse_points
    )
    metrics["timing_seconds"] = timing

    # 10. Load Ground-Truth Associations
    gt_map = {}
    if gt_assoc_csv.exists():
        with open(gt_assoc_csv, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                imgid = int(row["imgid"])
                gt_map[imgid] = {
                    "ground_truth_imgid": int(row["ground_truth_imgid"]) if row.get("ground_truth_imgid") else None,
                    "ground_truth_pose_timestamp_seconds": float(row["ground_truth_pose_timestamp_seconds"]) if row.get("ground_truth_pose_timestamp_seconds") else None,
                    "association_method": row.get("association_method", "UNMATCHED"),
                    "matched": row.get("matched", "").lower() == "true"
                }

    # 11. Export Standardized Baseline Outputs
    # a. registered_images.csv
    reg_img_csv_path = out_dir / "registered_images.csv"
    with open(reg_img_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["imgid", "filename", "registered", "colmap_image_id", "num_points2D", "num_points3D", "ground_truth_available"])
        for i in range(1, total_images_count + 1):
            fname = f"{i:05d}.jpg"
            reg_info = next((v for v in registered_images.values() if v["imgid"] == i), None)
            is_reg = reg_info is not None
            has_gt = gt_map.get(i, {}).get("matched", False)
            writer.writerow([
                i,
                fname,
                str(is_reg).lower(),
                reg_info["colmap_image_id"] if reg_info else "",
                reg_info["num_points2D"] if reg_info else 0,
                reg_info["num_points3D"] if reg_info else 0,
                str(has_gt).lower()
            ])

    # b. camera_poses.csv
    cam_poses_csv_path = out_dir / "camera_poses.csv"
    with open(cam_poses_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "imgid",
            "filename",
            "registered",
            "x",
            "y",
            "z",
            "qx",
            "qy",
            "qz",
            "qw",
            "pose_convention",
            "ground_truth_available",
            "ground_truth_imgid"
        ])
        for i in range(1, total_images_count + 1):
            fname = f"{i:05d}.jpg"
            reg_info = next((v for v in registered_images.values() if v["imgid"] == i), None)
            has_gt = gt_map.get(i, {}).get("matched", False)
            gt_id = gt_map.get(i, {}).get("ground_truth_imgid")
            if reg_info:
                writer.writerow([
                    i,
                    fname,
                    "true",
                    reg_info["x_world"],
                    reg_info["y_world"],
                    reg_info["z_world"],
                    reg_info["qx_world"],
                    reg_info["qy_world"],
                    reg_info["qz_world"],
                    reg_info["qw_world"],
                    "T_world_camera (C_w optical center, R_wc attitude in COLMAP world frame)",
                    str(has_gt).lower(),
                    gt_id if gt_id is not None else ""
                ])
            else:
                writer.writerow([
                    i,
                    fname,
                    "false",
                    "", "", "", "", "", "", "",
                    "",
                    str(has_gt).lower(),
                    gt_id if gt_id is not None else ""
                ])

    # c. sparse_points.csv
    pts_csv_path = out_dir / "sparse_points.csv"
    with open(pts_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["point3D_id", "x", "y", "z", "r", "g", "b", "error", "track_length"])
        for p in sparse_points.values():
            writer.writerow([
                p["point3D_id"],
                p["x"],
                p["y"],
                p["z"],
                p["r"],
                p["g"],
                p["b"],
                p["error"],
                p["track_length"]
            ])

    # d. reconstruction_summary.json
    summary_path = out_dir / "reconstruction_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    # e. b0_reconstruction_report.json
    full_report = {
        "colmap_version": "COLMAP 4.1.1 with CUDA",
        "dataset_name": "Zurich Urban MAV Dataset (AGZ Sample)",
        "camera_calibration": {
            "camera_model": camera_model,
            "camera_params": camera_params,
            "image_width": 1920,
            "image_height": 1080
        },
        "configuration": {
            "feature_extractor": "SIFT",
            "max_num_features": args.max_features,
            "matcher": "exhaustive",
            "use_gpu": bool(args.use_gpu)
        },
        "metrics": metrics,
        "unregistered_imgids": [i for i in range(1, total_images_count + 1) if not any(v["imgid"] == i for v in registered_images.values())]
    }
    report_path = reports_parent / "b0_reconstruction_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=4)

    # 12. Render Visualizations
    traj_png_path = reports_parent / "b0_camera_trajectory.png"
    pts_png_path = reports_parent / "b0_sparse_pointcloud.png"

    render_colmap_trajectory_plot(registered_images, traj_png_path)
    render_colmap_sparse_pointcloud_plot(sparse_points, registered_images, pts_png_path)

    # 13. Print Telemetry Summary
    print("\n--- COLMAP Baseline (b0) Summary ---")
    print(f"  Images Processed:     {metrics['total_images']}")
    print(f"  Images Registered:    {metrics['registered_images']} ({metrics['registration_percentage']}%)")
    print(f"  Sparse 3D Points:     {metrics['sparse_3d_point_count']:,}")
    print(f"  Total Observations:   {metrics['total_observations']:,}")
    print(f"  Mean Track Length:    {metrics['mean_track_length']:.2f}")
    print(f"  Mean Reproj. Error:   {metrics['reprojection_error']['mean_px']:.4f} px")
    print(f"  Median Reproj. Error: {metrics['reprojection_error']['median_px']:.4f} px")
    print(f"  Extraction Time:      {timing['feature_extraction_seconds']:.2f} s")
    print(f"  Matching Time:        {timing['feature_matching_seconds']:.2f} s")
    print(f"  Mapping Time:         {timing['mapping_seconds']:.2f} s")
    print(f"  Total Runtime:        {timing['total_runtime_seconds']:.2f} s")
    print("=" * 60)

    if metrics['registered_images'] == 0:
        print("COLMAP B0 STATUS: BLOCKED", file=sys.stderr)
        sys.exit(1)
    else:
        print("COLMAP B0 STATUS: PASS")

if __name__ == "__main__":
    main()
