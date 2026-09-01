"""
SIH26158 Depth Fusion - Robust Metric Scale Alignment Engine

This module implements robust affine inverse-depth alignment (1/Z = a * D_inv + b),
cross-validation, leave-one-out stability testing, and fallback handling.
"""

from typing import List, Dict, Tuple, Optional, Any
import numpy as np
from src.depth_fusion.metric_anchor import (
    MetricAnchor,
    AnchorSource,
    CalibrationStatus,
    MetricDepthOutput
)
from src.depth_fusion.scale_alignment import affine_inverse_depth_transform

class RobustMetricAlignmentEngine:
    """
    Robust estimator and validation engine for metric scale alignment.
    """
    def __init__(
        self,
        min_anchors: int = 15,
        min_unique_frames: int = 3,
        min_correlation: float = 0.20,
        max_condition_number: float = 1e4,
        huber_delta: float = 0.01,
        ransac_threshold: float = 0.05,
        max_ransac_iters: int = 100,
        random_seed: int = 42
    ):
        self.min_anchors = min_anchors
        self.min_unique_frames = min_unique_frames
        self.min_correlation = min_correlation
        self.max_condition_number = max_condition_number
        self.huber_delta = huber_delta
        self.ransac_threshold = ransac_threshold
        self.max_ransac_iters = max_ransac_iters
        self.random_seed = random_seed

    def filter_and_validate_anchors(self, anchors: List[MetricAnchor]) -> Tuple[bool, List[MetricAnchor], Dict[str, Any]]:
        """
        Validates anchor collection for production calibration.
        Rejects GROUND_TRUTH_EVALUATION_ONLY anchors.
        """
        valid_anchors = []
        for anc in anchors:
            # Rule: GROUND_TRUTH_EVALUATION_ONLY must NEVER be accepted by production calibration
            if anc.source == AnchorSource.GROUND_TRUTH_EVALUATION_ONLY:
                continue
            if anc.metric_depth_m > 0 and anc.inv_depth_predicted > 0:
                valid_anchors.append(anc)

        diag = {
            "total_provided": len(anchors),
            "valid_count": len(valid_anchors),
            "unique_frames": len(set(a.frame_id for a in valid_anchors))
        }

        if len(valid_anchors) < self.min_anchors:
            diag["rejection_reason"] = f"Insufficient anchors: {len(valid_anchors)} < {self.min_anchors}"
            return False, valid_anchors, diag

        if diag["unique_frames"] < self.min_unique_frames:
            diag["rejection_reason"] = f"Insufficient unique frames: {diag['unique_frames']} < {self.min_unique_frames}"
            return False, valid_anchors, diag

        d_invs = np.array([a.inv_depth_predicted for a in valid_anchors])
        inv_zs = np.array([1.0 / a.metric_depth_m for a in valid_anchors])

        # Check variance / rank
        A = np.column_stack([d_invs, np.ones_like(d_invs)])
        cond = float(np.linalg.cond(A))
        diag["condition_number"] = cond
        diag["d_inv_std"] = float(np.std(d_invs))
        diag["inv_z_std"] = float(np.std(inv_zs))

        if cond > self.max_condition_number or np.isnan(cond):
            diag["rejection_reason"] = f"Ill-conditioned design matrix: cond={cond:.2e} > {self.max_condition_number}"
            return False, valid_anchors, diag

        if diag["d_inv_std"] < 1e-4 or diag["inv_z_std"] < 1e-4:
            diag["rejection_reason"] = "Near-zero variance in anchor depth"
            return False, valid_anchors, diag

        corr = float(np.corrcoef(d_invs, inv_zs)[0, 1])
        diag["correlation"] = corr
        if corr < self.min_correlation or np.isnan(corr):
            diag["rejection_reason"] = f"Insufficient correlation between D_inv and 1/Z: corr={corr:.4f} < {self.min_correlation}"
            return False, valid_anchors, diag

        return True, valid_anchors, diag

    def fit_ransac_huber(
        self,
        d_invs: np.ndarray,
        inv_zs: np.ndarray,
        weights: Optional[np.ndarray] = None
    ) -> Tuple[float, float, np.ndarray, Dict[str, Any]]:
        """
        Fits 1/Z = a * D_inv + b with RANSAC inlier selection followed by Huber/IRLS refinement.
        Constrains a > 0.
        """
        N = len(d_invs)
        if weights is None:
            weights = np.ones(N)

        np.random.seed(self.random_seed)
        best_inliers = np.ones(N, dtype=bool)
        best_inlier_count = 0
        best_a, best_b = 0.0, 0.0

        # RANSAC iterations
        for _ in range(self.max_ransac_iters):
            idx = np.random.choice(N, size=2, replace=False)
            d1, d2 = d_invs[idx[0]], d_invs[idx[1]]
            z1, z2 = inv_zs[idx[0]], inv_zs[idx[1]]

            if abs(d1 - d2) < 1e-6:
                continue

            a_cand = (z1 - z2) / (d1 - d2)
            if a_cand <= 0:  # Constraint a > 0
                continue

            b_cand = z1 - a_cand * d1
            residuals = np.abs(inv_zs - (a_cand * d_invs + b_cand))
            inliers = residuals < self.ransac_threshold
            inlier_count = np.sum(inliers)

            if inlier_count > best_inlier_count:
                best_inlier_count = inlier_count
                best_inliers = inliers
                best_a, best_b = a_cand, b_cand

        # If RANSAC found too few inliers, fallback to full WLS
        if best_inlier_count < max(5, int(0.2 * N)):
            best_inliers = np.ones(N, dtype=bool)

        # Refinement via Iteratively Reweighted Least Squares (IRLS) with Huber loss
        inlier_d = d_invs[best_inliers]
        inlier_z = inv_zs[best_inliers]
        inlier_w = weights[best_inliers]

        A = np.column_stack([inlier_d, np.ones_like(inlier_d)])
        
        # Initial WLS
        W_mat = np.diag(inlier_w)
        try:
            x, _, _, _ = np.linalg.lstsq(A.T @ W_mat @ A, A.T @ W_mat @ inlier_z, rcond=None)
            a, b = float(x[0]), float(x[1])
        except np.linalg.LinAlgError:
            a, b = best_a, best_b

        # Enforce positive scale constraint a > 0
        if a <= 0:
            a = max(1e-6, best_a)

        # IRLS Huber refinement
        for _ in range(5):
            res = np.abs(inlier_z - (a * inlier_d + b))
            huber_w = np.where(res <= self.huber_delta, 1.0, self.huber_delta / np.maximum(res, 1e-6))
            total_w = inlier_w * huber_w
            W_mat = np.diag(total_w)
            try:
                x_new, _, _, _ = np.linalg.lstsq(A.T @ W_mat @ A, A.T @ W_mat @ inlier_z, rcond=None)
                if x_new[0] > 0:
                    a, b = float(x_new[0]), float(x_new[1])
            except np.linalg.LinAlgError:
                break

        final_res = inv_zs - (a * d_invs + b)
        rmse = float(np.sqrt(np.mean(final_res**2)))
        mae = float(np.mean(np.abs(final_res)))

        stats = {
            "inlier_count": int(best_inlier_count),
            "inlier_ratio": float(best_inlier_count / N),
            "rmse_inv_depth": rmse,
            "mae_inv_depth": mae
        }
        return a, b, best_inliers, stats

    def calibrate_depth(
        self,
        raw_inv_depth: np.ndarray,
        anchors: List[MetricAnchor],
        confidence_map: Optional[np.ndarray] = None
    ) -> MetricDepthOutput:
        """
        Calibrates raw inverse depth map using metric anchors.
        Falls back cleanly to relative depth mode if anchors fail validation.
        """
        if confidence_map is None:
            confidence_map = np.ones_like(raw_inv_depth, dtype=np.float32)

        is_valid, valid_anchors, diag = self.filter_and_validate_anchors(anchors)

        if not is_valid:
            # Safe Fallback to relative depth mode
            rel_depth = 1.0 / np.maximum(raw_inv_depth, 1e-6)
            return MetricDepthOutput(
                depth=rel_depth,
                confidence=confidence_map,
                metric=False,
                scale_a=None,
                shift_b=None,
                calibration_status=CalibrationStatus.METRIC_SCALE_NOT_IDENTIFIABLE,
                metadata={"reason": diag.get("rejection_reason", "Validation failed"), "diagnostics": diag}
            )

        d_invs = np.array([a.inv_depth_predicted for a in valid_anchors])
        inv_zs = np.array([1.0 / a.metric_depth_m for a in valid_anchors])
        weights = np.array([a.confidence for a in valid_anchors])

        a, b, inliers, stats = self.fit_ransac_huber(d_invs, inv_zs, weights)

        metric_depth = affine_inverse_depth_transform(raw_inv_depth, a, b)

        return MetricDepthOutput(
            depth=metric_depth,
            confidence=confidence_map,
            metric=True,
            scale_a=a,
            shift_b=b,
            calibration_status=CalibrationStatus.METRIC_ALIGNMENT_VALID,
            metadata={
                "anchor_count": len(valid_anchors),
                "inlier_stats": stats,
                "diagnostics": diag
            }
        )

    def run_leave_one_frame_out(self, anchors: List[MetricAnchor]) -> Dict[str, Any]:
        """
        Performs Leave-One-Frame-Out cross-validation across all unique frames.
        """
        unique_frames = sorted(list(set(a.frame_id for a in anchors)))
        if len(unique_frames) < self.min_unique_frames:
            return {"status": "FAILED", "reason": "Insufficient unique frames"}

        fold_results = []
        for held_out_frame in unique_frames:
            train_anchors = [a for a in anchors if a.frame_id != held_out_frame]
            val_anchors = [a for a in anchors if a.frame_id == held_out_frame]

            is_valid, v_train, _ = self.filter_and_validate_anchors(train_anchors)
            if not is_valid:
                fold_results.append({"held_out_frame": held_out_frame, "status": "REJECTED"})
                continue

            d_train = np.array([a.inv_depth_predicted for a in v_train])
            z_train = np.array([1.0 / a.metric_depth_m for a in v_train])
            a, b, _, _ = self.fit_ransac_huber(d_train, z_train)

            # Evaluate on held out frame
            d_val = np.array([a_val.inv_depth_predicted for a_val in val_anchors])
            z_val = np.array([1.0 / a_val.metric_depth_m for a_val in val_anchors])
            pred_z = a * d_val + b
            val_rmse = float(np.sqrt(np.mean((z_val - pred_z)**2)))

            fold_results.append({
                "held_out_frame": held_out_frame,
                "scale_a": a,
                "shift_b": b,
                "val_rmse_inv_depth": val_rmse,
                "status": "PASS"
            })

        passed_folds = [f for f in fold_results if f["status"] == "PASS"]
        if not passed_folds:
            return {"status": "FAILED", "passed_folds": 0, "folds": fold_results}

        a_vals = [f["scale_a"] for f in passed_folds]
        b_vals = [f["shift_b"] for f in passed_folds]

        return {
            "status": "PASS",
            "total_folds": len(unique_frames),
            "passed_folds": len(passed_folds),
            "mean_scale_a": float(np.mean(a_vals)),
            "std_scale_a": float(np.std(a_vals)),
            "mean_shift_b": float(np.mean(b_vals)),
            "std_shift_b": float(np.std(b_vals)),
            "folds": fold_results
        }
