"""
SIH26158 Depth Fusion - Metric Scale Alignment Design and Identifiability Module

This module implements the mathematical models, scale estimators, degeneracy detectors,
and identifiability checks for transforming relative monocular depth to metric depth.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import numpy as np
import torch

class MetricIdentifiabilityStatus(Enum):
    NOT_IDENTIFIABLE = "METRIC_SCALE_NOT_IDENTIFIABLE"
    PARTIALLY_IDENTIFIABLE = "METRIC_SCALE_PARTIALLY_IDENTIFIABLE"
    IDENTIFIABLE = "METRIC_SCALE_IDENTIFIABLE"

@dataclass
class ScaleAlignmentResult:
    status: MetricIdentifiabilityStatus
    scale_a: Optional[float] = None
    shift_b: Optional[float] = None
    condition_number: Optional[float] = None
    baseline_to_depth_ratio: Optional[float] = None
    residual_rmse: Optional[float] = None
    inlier_ratio: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

def affine_inverse_depth_transform(inv_depth: np.ndarray, a: float, b: float, epsilon: float = 1e-6) -> np.ndarray:
    """
    Transforms relative inverse depth (disparity) to metric depth using affine inverse parameters:
    1 / Z_metric = a * D_inv + b
    => Z_metric = 1 / max(a * D_inv + b, epsilon)
    """
    inv_metric = a * inv_depth + b
    # Ensure positive depth
    inv_metric_clamped = np.maximum(inv_metric, epsilon)
    return 1.0 / inv_metric_clamped

def affine_direct_depth_transform(rel_depth: np.ndarray, s: float, t: float = 0.0) -> np.ndarray:
    """
    Transforms relative depth to metric depth using direct affine scaling:
    Z_metric = s * D_rel + t
    """
    return s * rel_depth + t

class RobustScaleEstimator:
    """
    Robust estimator for affine inverse-depth alignment:
    min_{a, b} sum_i w_i * rho( 1 / Z_metric_i - (a * D_inv_i + b) )
    """
    def __init__(self, min_points: int = 5, min_parallax_ratio: float = 0.05, huber_delta: float = 0.01):
        self.min_points = min_points
        self.min_parallax_ratio = min_parallax_ratio
        self.huber_delta = huber_delta

    def estimate_from_anchors(
        self,
        inv_depth_samples: np.ndarray,
        metric_depth_samples: np.ndarray,
        weights: Optional[np.ndarray] = None
    ) -> ScaleAlignmentResult:
        """
        Fits (a, b) mapping relative inverse depth to metric inverse depth.
        """
        N = len(inv_depth_samples)
        if N < self.min_points:
            return ScaleAlignmentResult(
                status=MetricIdentifiabilityStatus.NOT_IDENTIFIABLE,
                metadata={"reason": f"Insufficient anchor points: {N} < {self.min_points}"}
            )

        inv_metric = 1.0 / np.maximum(metric_depth_samples, 1e-6)
        
        # Design matrix A = [D_inv, 1]
        A = np.column_stack([inv_depth_samples, np.ones(N)])
        
        # Check condition number / depth variance
        cond = np.linalg.cond(A)
        if cond > 1e4 or np.isnan(cond):
            return ScaleAlignmentResult(
                status=MetricIdentifiabilityStatus.NOT_IDENTIFIABLE,
                condition_number=float(cond),
                metadata={"reason": f"Degenerate anchor geometry (condition number {cond:.2e} > 1e4)"}
            )

        if weights is None:
            W = np.ones(N)
        else:
            W = np.asarray(weights)
            
        # Weighted Least Squares: (A^T W A) x = A^T W y
        W_diag = np.diag(W)
        try:
            x, residuals, rank, s = np.linalg.lstsq(A.T @ W_diag @ A, A.T @ W_diag @ inv_metric, rcond=None)
            a, b = float(x[0]), float(x[1])
        except np.linalg.LinAlgError:
            return ScaleAlignmentResult(
                status=MetricIdentifiabilityStatus.NOT_IDENTIFIABLE,
                metadata={"reason": "Linear solver failed"}
            )

        # Residuals
        pred_inv = a * inv_depth_samples + b
        res = inv_metric - pred_inv
        rmse = float(np.sqrt(np.mean(res**2)))

        return ScaleAlignmentResult(
            status=MetricIdentifiabilityStatus.IDENTIFIABLE,
            scale_a=a,
            shift_b=b,
            condition_number=float(cond),
            residual_rmse=rmse,
            inlier_ratio=1.0,
            metadata={"anchor_count": N, "solver": "Weighted_Linear_Least_Squares"}
        )

    def check_motion_observability(
        self,
        baseline_m: float,
        mean_scene_depth_m: float
    ) -> Tuple[MetricIdentifiabilityStatus, float]:
        """
        Checks if camera motion baseline is sufficient to uniquely identify depth scale.
        """
        if mean_scene_depth_m <= 0:
            return MetricIdentifiabilityStatus.NOT_IDENTIFIABLE, 0.0
            
        b_over_z = baseline_m / mean_scene_depth_m
        if b_over_z < self.min_parallax_ratio:
            return MetricIdentifiabilityStatus.NOT_IDENTIFIABLE, b_over_z
        elif b_over_z < 0.2:
            return MetricIdentifiabilityStatus.PARTIALLY_IDENTIFIABLE, b_over_z
        else:
            return MetricIdentifiabilityStatus.IDENTIFIABLE, b_over_z
