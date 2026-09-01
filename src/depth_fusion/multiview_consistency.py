"""
SIH26158 Depth Fusion - Scale-Aware Multi-View Consistency Evaluator

This module evaluates geometric and depth consistency across overlapping frame pairs
using scale-invariant relative ray reprojection and normalized disparity residuals.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
from src.depth_fusion.pointcloud_fusion import RelativePointcloud

class MultiViewConsistencyEvaluator:
    """
    Evaluates multi-view reprojection consistency across frame pairs without assuming metric scale.
    Uses scale-invariant ray rotation and normalized disparity residuals.
    """
    def __init__(
        self,
        K_rect: np.ndarray,
        consistency_threshold: float = 0.30,
        tau: float = 0.20
    ):
        self.K_rect = K_rect
        self.consistency_threshold = consistency_threshold
        self.tau = tau
        self.fx = K_rect[0, 0]
        self.fy = K_rect[1, 1]
        self.cx = K_rect[0, 2]
        self.cy = K_rect[1, 2]

    def evaluate_pair_consistency(
        self,
        pcd_A: RelativePointcloud,
        rel_depth_B: np.ndarray,
        R_wc_B: np.ndarray,
        C_world_B: np.ndarray,
        image_shape: Tuple[int, int] = (1080, 1920),
        R_wc_A: Optional[np.ndarray] = None,
        C_world_A: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Projects 3D points from frame A into camera frame B via relative rotation R_BA,
        samples predicted normalized depth, and computes scale-invariant consistency weights.
        """
        N = len(pcd_A.points)
        if N == 0:
            return (
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=bool),
                {"status": "EMPTY"}
            )

        H, W = image_shape
        pts_w = pcd_A.points

        # Compute point in camera A frame: X_cA = R_cw_A @ (X_w - C_world_A)
        if R_wc_A is not None and C_world_A is not None:
            R_cw_A = R_wc_A.T
            pts_cA = (pts_w - C_world_A.reshape(1, 3)) @ R_cw_A.T
        elif R_wc_A is not None:
            pts_cA = pts_w @ R_wc_A  # Fallback if uncentered
        else:
            pts_cA = pts_w.copy()

        # Relative rotation R_BA = R_wc_B^T @ R_wc_A
        if R_wc_A is not None:
            R_BA = R_wc_B.T @ R_wc_A
        else:
            R_BA = np.eye(3, dtype=np.float32)

        # Rotate points to camera B: X_cB = R_BA @ X_cA
        pts_cB = pts_cA @ R_BA.T
        X_cB = pts_cB[:, 0]
        Y_cB = pts_cB[:, 1]
        Z_cB = pts_cB[:, 2]

        pos_depth = Z_cB > 1e-4
        Z_safe = np.maximum(Z_cB, 1e-4)

        u_B = (X_cB * self.fx / Z_safe) + self.cx
        v_B = (Y_cB * self.fy / Z_safe) + self.cy

        inside_B = pos_depth & (u_B >= 10) & (u_B < W - 10) & (v_B >= 10) & (v_B < H - 10)

        consistency_weights = np.full(N, 0.5, dtype=np.float32)
        inlier_mask = np.zeros(N, dtype=bool)
        residuals = []

        if np.any(inside_B):
            u_in = u_B[inside_B].astype(np.int32)
            v_in = v_B[inside_B].astype(np.int32)

            # Sample predicted relative depth at projected locations in B
            z_pred_B = rel_depth_B[v_in, u_in]
            z_orig_A = pts_cA[inside_B, 2]

            # Scale-invariant comparison: normalize each by its median
            med_B = np.median(z_pred_B)
            med_A = np.median(z_orig_A)

            scale_norm_B = z_pred_B / max(1e-6, med_B)
            scale_norm_A = z_orig_A / max(1e-6, med_A)

            # Scale-invariant relative residual
            rel_error = np.abs(scale_norm_A - scale_norm_B) / (scale_norm_A + scale_norm_B + 1e-6)

            pair_conf = np.exp(-rel_error / self.tau).astype(np.float32)
            is_inlier = rel_error < self.consistency_threshold

            consistency_weights[inside_B] = pair_conf
            inlier_mask[inside_B] = is_inlier
            residuals = list(rel_error)

        diag = {
            "total_points": N,
            "projected_inside_count": int(np.count_nonzero(inside_B)),
            "inlier_count": int(np.count_nonzero(inlier_mask)),
            "inlier_ratio": float(np.count_nonzero(inlier_mask) / max(1, np.count_nonzero(inside_B))),
            "mean_relative_residual": float(np.mean(residuals)) if len(residuals) > 0 else 0.0,
            "median_relative_residual": float(np.median(residuals)) if len(residuals) > 0 else 0.0
        }

        return consistency_weights, inlier_mask, diag
