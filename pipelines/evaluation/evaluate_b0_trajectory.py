import sys
import os
import csv
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List

sys.path.insert(0, str(Path("D:/SIH26158-single-pass-3D").resolve()))

from src.metrics.alignment import (
    umeyama_alignment,
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion
)
from src.metrics.trajectory_metrics import (
    compute_ate,
    compute_rpe,
    compute_trajectory_statistics
)
from src.visualization.trajectory_eval_visualizer import (
    render_b0_gt_vs_colmap_topdown,
    render_b0_position_error_plot,
    render_b0_trajectory_comparison_3d
)

def evaluate_b0_trajectory(
    colmap_poses_csv: Path,
    gt_poses_csv: Path,
    assoc_csv: Path,
    output_dir: Path
) -> Dict[str, Any]:
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    colmap_records = {}
    with open(colmap_poses_csv, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["registered"].lower() == "true":
                imgid = int(r["imgid"])
                colmap_records[imgid] = {
                    "image_id": int(r["image_id"]),
                    "imgid": imgid,
                    "filename": r["filename"],
                    "x": float(r["camera_center_x"]),
                    "y": float(r["camera_center_y"]),
                    "z": float(r["camera_center_z"]),
                    "qx": float(r["q_wc_x"]),
                    "qy": float(r["q_wc_y"]),
                    "qz": float(r["q_wc_z"]),
                    "qw": float(r["q_wc_w"])
                }

    gt_records = {}
    with open(gt_poses_csv, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            imgid = int(r["imgid"])
            gt_records[imgid] = {
                "imgid": imgid,
                "timestamp": float(r["timestamp_seconds"]),
                "x": float(r["tx"]),
                "y": float(r["ty"]),
                "z": float(r["tz"]),
                "qx": float(r["qx"]),
                "qy": float(r["qy"]),
                "qz": float(r["qz"]),
                "qw": float(r["qw"])
            }

    assoc_records = []
    with open(assoc_csv, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["matched"].lower() == "true":
                assoc_records.append(int(r["imgid"]))

    # 2. Extract Exact Ground-Truth Intersection
    eval_pairs = []
    for imgid in sorted(assoc_records):
        if imgid in colmap_records and imgid in gt_records:
            c = colmap_records[imgid]
            g = gt_records[imgid]
            eval_pairs.append({
                "imgid": imgid,
                "filename": c["filename"],
                "gt_timestamp": g["timestamp"],
                "gt_x": g["x"],
                "gt_y": g["y"],
                "gt_z": g["z"],
                "gt_qx": g["qx"],
                "gt_qy": g["qy"],
                "gt_qz": g["qz"],
                "gt_qw": g["qw"],
                "colmap_x": c["x"],
                "colmap_y": c["y"],
                "colmap_z": c["z"],
                "colmap_qx": c["qx"],
                "colmap_qy": c["qy"],
                "colmap_qz": c["qz"],
                "colmap_qw": c["qw"]
            })

    print(f"Total Exact Ground-Truth Evaluation Pairs: {len(eval_pairs)}")

    # 3. Export Evaluation Pairs CSV
    eval_csv_path = output_dir / "b0_gt_evaluation_pairs.csv"
    with open(eval_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "imgid", "filename", "gt_timestamp",
            "gt_x", "gt_y", "gt_z", "gt_qx", "gt_qy", "gt_qz", "gt_qw",
            "colmap_x", "colmap_y", "colmap_z", "colmap_qx", "colmap_qy", "colmap_qz", "colmap_qw"
        ])
        for p in eval_pairs:
            w.writerow([
                p["imgid"], p["filename"], f"{p['gt_timestamp']:.6f}",
                f"{p['gt_x']:.6f}", f"{p['gt_y']:.6f}", f"{p['gt_z']:.6f}",
                f"{p['gt_qx']:.8f}", f"{p['gt_qy']:.8f}", f"{p['gt_qz']:.8f}", f"{p['gt_qw']:.8f}",
                f"{p['colmap_x']:.8f}", f"{p['colmap_y']:.8f}", f"{p['colmap_z']:.8f}",
                f"{p['colmap_qx']:.8f}", f"{p['colmap_qy']:.8f}", f"{p['colmap_qz']:.8f}", f"{p['colmap_qw']:.8f}"
            ])

    # 4. Coordinate Frame Normalization (Local Metric Frame)
    gt_utm_pts = np.array([[p["gt_x"], p["gt_y"], p["gt_z"]] for p in eval_pairs])
    gt_origin = gt_utm_pts[0].copy()
    gt_local_pts = gt_utm_pts - gt_origin  # Local ENU (meters) relative to first GT keyframe

    colmap_pts = np.array([[p["colmap_x"], p["colmap_y"], p["colmap_z"]] for p in eval_pairs])

    # Rotation matrices
    gt_rot_mats = [quaternion_to_rotation_matrix([p["gt_qx"], p["gt_qy"], p["gt_qz"], p["gt_qw"]]) for p in eval_pairs]
    colmap_rot_mats = [quaternion_to_rotation_matrix([p["colmap_qx"], p["colmap_qy"], p["colmap_qz"], p["colmap_qw"]]) for p in eval_pairs]

    # 5. Umeyama Similarity Alignment (Sim(3))
    scale_s, R_sim, t_sim, colmap_sim_aligned = umeyama_alignment(colmap_pts, gt_local_pts, with_scale=True)
    q_sim = rotation_matrix_to_quaternion(R_sim)

    # 6. Rigid Alignment without Scale (SE(3))
    _, R_se3, t_se3, colmap_se3_aligned = umeyama_alignment(colmap_pts, gt_local_pts, with_scale=False)

    # Aligned rotation matrices for COLMAP
    colmap_aligned_rot_mats = [R_sim @ R_c for R_c in colmap_rot_mats]

    # 7. Compute ATE
    ate_sim = compute_ate(colmap_sim_aligned, gt_local_pts)
    ate_se3 = compute_ate(colmap_se3_aligned, gt_local_pts)

    # 8. Compute RPE
    rpe_sim = compute_rpe(colmap_sim_aligned, gt_local_pts, colmap_aligned_rot_mats, gt_rot_mats, delta=1)

    # 9. Trajectory Statistics
    traj_stats = compute_trajectory_statistics(colmap_pts, colmap_sim_aligned, gt_local_pts, scale_s)

    # 10. Load Reprojection Metrics
    rec_summary_path = output_dir / "reconstruction_summary.json"
    reproj_stats = {}
    if rec_summary_path.exists():
        with open(rec_summary_path, "r", encoding="utf-8") as f:
            rec_data = json.load(f)
            reproj_stats = rec_data.get("sparse_3d_metrics", {}).get("reprojection_error", {})

    # 11. Build Evaluation JSON Artifact
    eval_results = {
        "dataset": "Zurich Urban MAV Dataset",
        "baseline": "B0 (Classical COLMAP SfM, FULL_OPENCV Camera)",
        "evaluation_subset": {
            "total_sample_images": 350,
            "registered_images": 350,
            "evaluated_ground_truth_keyframes": len(eval_pairs),
            "keyframe_imgids": [p["imgid"] for p in eval_pairs],
            "keyframe_timestamps_s": [round(p["gt_timestamp"], 4) for p in eval_pairs]
        },
        "coordinate_frames": {
            "ground_truth_source_frame": "WGS 84 / UTM zone 32N (Meters)",
            "ground_truth_local_frame": "Local ENU relative to keyframe 1 (Meters)",
            "local_origin_utm_xyz": [round(float(v), 6) for v in gt_origin],
            "colmap_source_frame": "COLMAP Reconstructed World Frame (Arbitrary Scale)",
            "quaternion_convention": "Hamilton scalar-last [qx, qy, qz, qw]"
        },
        "sim3_alignment": {
            "scale_factor_s": round(scale_s, 6),
            "rotation_matrix": [[round(float(v), 6) for v in row] for row in R_sim],
            "rotation_quaternion_wxyz": [round(float(q_sim[3]), 6), round(float(q_sim[0]), 6), round(float(q_sim[1]), 6), round(float(q_sim[2]), 6)],
            "rotation_quaternion_xyzw": [round(float(v), 6) for v in q_sim],
            "translation_vector_xyz_m": [round(float(v), 6) for v in t_sim],
            "pre_alignment_residual_m": round(float(np.sqrt(np.mean(np.sum((colmap_pts - gt_local_pts)**2, axis=1)))), 4),
            "post_alignment_residual_m": ate_sim["rmse_m"]
        },
        "se3_rigid_alignment_unscaled": {
            "scale_factor": 1.0,
            "translation_residual_rmse_m": ate_se3["rmse_m"],
            "translation_residual_mean_m": ate_se3["mean_m"]
        },
        "ate_metrics_m": ate_sim,
        "rpe_metrics": rpe_sim,
        "scale_metrics": {
            "scale_ratio": round(scale_s, 6),
            "scale_error_percent": traj_stats["scale_error_percent"],
            "trajectory_length_ratio": traj_stats["trajectory_length_ratio"]
        },
        "trajectory_statistics": traj_stats,
        "image_space_reprojection_metrics": reproj_stats,
        "limitations": [
            "Pure monocular Structure-from-Motion suffers from gauge and scale freedom, requiring Sim(3) alignment for trajectory comparison against metric ground truth.",
            "Ground truth poses are available only at 1 Hz keyframes (every 30 frames); intermediate 30 FPS frames are not interpolated to avoid fabricating ground truth.",
            "Scale-aligned ATE quantifies geometric shape consistency; it must not be conflated with absolute unscaled metric accuracy."
        ]
    }

    eval_json_path = output_dir / "b0_evaluation.json"
    with open(eval_json_path, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=4)

    print(f"Generated {eval_json_path}")

    # 12. Render Visualizations
    topdown_png = output_dir / "b0_gt_vs_colmap_topdown.png"
    pos_err_png = output_dir / "b0_position_error.png"
    traj_3d_png = output_dir / "b0_trajectory_comparison_3d.png"

    kf_ids = [p["imgid"] for p in eval_pairs]
    render_b0_gt_vs_colmap_topdown(gt_local_pts, colmap_pts, colmap_sim_aligned, kf_ids, topdown_png)
    render_b0_position_error_plot(kf_ids, ate_sim["per_frame_errors_m"], pos_err_png)
    render_b0_trajectory_comparison_3d(gt_local_pts, colmap_pts, colmap_sim_aligned, traj_3d_png)

    print("Generated evaluation visualizations:")
    print(f"  1. {topdown_png}")
    print(f"  2. {pos_err_png}")
    print(f"  3. {traj_3d_png}")

    return eval_results

if __name__ == "__main__":
    colmap_poses = Path("outputs/reports/zurich_mav/b0/camera_poses_colmap.csv")
    gt_poses = Path("outputs/reports/zurich_mav/pose.csv")
    assoc_file = Path("outputs/reports/zurich_mav/image_groundtruth_associations.csv")
    out_dir = Path("outputs/reports/zurich_mav/b0")

    res = evaluate_b0_trajectory(colmap_poses, gt_poses, assoc_file, out_dir)
    print("\n--- B0 Trajectory Evaluation Summary ---")
    print(f"  GT Evaluation Pairs:    {res['evaluation_subset']['evaluated_ground_truth_keyframes']}")
    print(f"  ATE RMSE:               {res['ate_metrics_m']['rmse_m']:.4f} m")
    print(f"  ATE Mean:               {res['ate_metrics_m']['mean_m']:.4f} m")
    print(f"  Translational RPE RMSE: {res['rpe_metrics']['translational_rpe']['rmse_m']:.4f} m")
    print(f"  Rotational RPE RMSE:    {res['rpe_metrics']['rotational_rpe']['rmse_deg']:.4f} deg")
    print(f"  Scale Factor (s):       {res['sim3_alignment']['scale_factor_s']:.6f}")
    print(f"  Scale Error:            {res['scale_metrics']['scale_error_percent']:.2f}%")
    print(f"  GT Trajectory Length:   {res['trajectory_statistics']['ground_truth_trajectory_length_m']:.4f} m")
    print(f"  Raw Trajectory Length:  {res['trajectory_statistics']['raw_colmap_trajectory_length_units']:.4f} units")
    print(f"  Aligned Traj Length:    {res['trajectory_statistics']['aligned_colmap_trajectory_length_m']:.4f} m")
