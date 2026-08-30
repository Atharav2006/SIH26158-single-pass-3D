import numpy as np
import math
from typing import Dict, Any, List, Tuple, Union

from src.metrics.alignment import quaternion_to_rotation_matrix

def compute_ate(
    estimated_pts: np.ndarray,
    ground_truth_pts: np.ndarray
) -> Dict[str, Any]:
    """
    Compute Absolute Trajectory Error (ATE) between aligned estimated and ground truth positions.
    
    Args:
        estimated_pts: (N, 3) aligned estimated positions in meters.
        ground_truth_pts: (N, 3) ground truth positions in meters.
        
    Returns:
        Dictionary containing ATE summary statistics (RMSE, mean, median, std, max, min) and per-frame error vector.
    """
    assert estimated_pts.shape == ground_truth_pts.shape, "Shape mismatch between estimated and ground truth"
    errors = np.linalg.norm(estimated_pts - ground_truth_pts, axis=1)

    rmse = float(np.sqrt(np.mean(errors ** 2)))
    mean_err = float(np.mean(errors))
    median_err = float(np.median(errors))
    std_err = float(np.std(errors))
    max_err = float(np.max(errors))
    min_err = float(np.min(errors))

    return {
        "rmse_m": round(rmse, 4),
        "mean_m": round(mean_err, 4),
        "median_m": round(median_err, 4),
        "std_m": round(std_err, 4),
        "max_m": round(max_err, 4),
        "min_m": round(min_err, 4),
        "per_frame_errors_m": [round(float(e), 4) for e in errors]
    }

def compute_rpe(
    estimated_pts: np.ndarray,
    ground_truth_pts: np.ndarray,
    estimated_rot_matrices: List[np.ndarray],
    ground_truth_rot_matrices: List[np.ndarray],
    delta: int = 1
) -> Dict[str, Any]:
    """
    Compute Relative Pose Error (RPE) for consecutive evaluated keyframe pairs.
    
    Relative translational error:
      || (p_est_{i+delta} - p_est_i) - (p_gt_{i+delta} - p_gt_i) ||
      
    Relative rotational error:
      Angle of (R_est_rel^T @ R_gt_rel) in degrees
    """
    n = len(estimated_pts)
    assert n == len(ground_truth_pts) == len(estimated_rot_matrices) == len(ground_truth_rot_matrices)

    trans_errors = []
    rot_errors_deg = []

    for i in range(n - delta):
        # Relative translation
        d_est = estimated_pts[i + delta] - estimated_pts[i]
        d_gt = ground_truth_pts[i + delta] - ground_truth_pts[i]
        t_err = np.linalg.norm(d_est - d_gt)
        trans_errors.append(float(t_err))

        # Relative rotation
        R_est_i = estimated_rot_matrices[i]
        R_est_j = estimated_rot_matrices[i + delta]
        R_gt_i = ground_truth_rot_matrices[i]
        R_gt_j = ground_truth_rot_matrices[i + delta]

        R_est_rel = R_est_i.T @ R_est_j
        R_gt_rel = R_gt_i.T @ R_gt_j

        # Error rotation matrix: delta_R = R_est_rel^T @ R_gt_rel
        delta_R = R_est_rel.T @ R_gt_rel
        cos_theta = (np.trace(delta_R) - 1.0) / 2.0
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        angle_rad = np.arccos(cos_theta)
        angle_deg = np.degrees(angle_rad)
        rot_errors_deg.append(float(angle_deg))

    trans_arr = np.array(trans_errors) if trans_errors else np.array([0.0])
    rot_arr = np.array(rot_errors_deg) if rot_errors_deg else np.array([0.0])

    return {
        "translational_rpe": {
            "rmse_m": round(float(np.sqrt(np.mean(trans_arr ** 2))), 4),
            "mean_m": round(float(np.mean(trans_arr)), 4),
            "median_m": round(float(np.median(trans_arr)), 4),
            "std_m": round(float(np.std(trans_arr)), 4),
            "max_m": round(float(np.max(trans_arr)), 4),
            "per_pair_errors_m": [round(float(e), 4) for e in trans_arr]
        },
        "rotational_rpe": {
            "rmse_deg": round(float(np.sqrt(np.mean(rot_arr ** 2))), 4),
            "mean_deg": round(float(np.mean(rot_arr)), 4),
            "median_deg": round(float(np.median(rot_arr)), 4),
            "std_deg": round(float(np.std(rot_arr)), 4),
            "max_deg": round(float(np.max(rot_arr)), 4),
            "per_pair_errors_deg": [round(float(e), 4) for e in rot_arr]
        }
    }

def compute_trajectory_statistics(
    raw_colmap_pts: np.ndarray,
    aligned_colmap_pts: np.ndarray,
    ground_truth_pts: np.ndarray,
    scale_factor: float
) -> Dict[str, Any]:
    """
    Compute trajectory path length, scale consistency, endpoint drift, and spatial extents.
    """
    def path_length(pts: np.ndarray) -> float:
        if len(pts) < 2:
            return 0.0
        diffs = np.diff(pts, axis=0)
        return float(np.sum(np.linalg.norm(diffs, axis=1)))

    gt_len = path_length(ground_truth_pts)
    raw_len = path_length(raw_colmap_pts)
    aligned_len = path_length(aligned_colmap_pts)

    endpoint_error_m = float(np.linalg.norm(aligned_colmap_pts[-1] - ground_truth_pts[-1]))
    normalized_trajectory_error_pct = (endpoint_error_m / gt_len * 100.0) if gt_len > 0 else 0.0

    scale_error_pct = abs(scale_factor - 1.0) * 100.0 if scale_factor != 1.0 else 0.0
    length_ratio = (aligned_len / gt_len) if gt_len > 0 else 1.0

    return {
        "ground_truth_trajectory_length_m": round(gt_len, 4),
        "raw_colmap_trajectory_length_units": round(raw_len, 4),
        "aligned_colmap_trajectory_length_m": round(aligned_len, 4),
        "scale_factor_s": round(scale_factor, 6),
        "scale_error_percent": round(scale_error_pct, 4),
        "trajectory_length_ratio": round(length_ratio, 6),
        "endpoint_error_m": round(endpoint_error_m, 4),
        "normalized_trajectory_error_percent": round(normalized_trajectory_error_pct, 4),
        "spatial_extents": {
            "ground_truth_xyz_span_m": [round(float(s), 4) for s in (np.ptp(ground_truth_pts, axis=0))],
            "aligned_colmap_xyz_span_m": [round(float(s), 4) for s in (np.ptp(aligned_colmap_pts, axis=0))]
        }
    }
