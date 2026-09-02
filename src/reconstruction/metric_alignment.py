import numpy as np
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum

class AlignmentStatus(Enum):
    SUCCESS = "SUCCESS"
    INSUFFICIENT_POINTS = "INSUFFICIENT_POINTS"
    INVALID_COORDINATES = "INVALID_COORDINATES"
    COLLINEAR_DEGENERATE = "COLLINEAR_DEGENERATE"
    UNSTABLE_SCALE = "UNSTABLE_SCALE"
    DUPLICATE_POINTS = "DUPLICATE_POINTS"

@dataclass
class AlignmentResult:
    status: AlignmentStatus
    scale: Optional[float] = None
    rotation_matrix: Optional[np.ndarray] = None
    translation: Optional[np.ndarray] = None
    transformed_points: Optional[np.ndarray] = None
    control_residuals: Optional[np.ndarray] = None
    rms_control_residual: Optional[float] = None
    inlier_ids: Optional[List[Any]] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    failure_reason: Optional[str] = None

    def transform(self, points: np.ndarray) -> np.ndarray:
        """Apply the frozen metric transformation to a set of points (e.g., checkpoints or point cloud)."""
        if self.status != AlignmentStatus.SUCCESS:
            raise RuntimeError("Cannot apply transform: Alignment was not successful.")
        
        if len(points) == 0:
            return np.empty((0, 3))
            
        pts = np.asarray(points, dtype=np.float64)
        if pts.shape[1] != 3:
            raise ValueError(f"Points must have shape (N, 3), got {pts.shape}")
            
        return self.scale * (pts @ self.rotation_matrix.T) + self.translation

class MetricAligner:
    """
    Estimates a 7-DoF similarity transformation (scale, rotation, translation) 
    from source reconstruction coordinates to target metric GCP coordinates.
    
    Uses Umeyama's algorithm for Absolute Orientation.
    """
    
    @staticmethod
    def align(
        source_points: np.ndarray, 
        target_points: np.ndarray, 
        point_ids: Optional[List[Any]] = None
    ) -> AlignmentResult:
        """
        source_points: (N, 3) relative reconstruction coordinates
        target_points: (N, 3) metric control point coordinates
        point_ids: (N,) optional list of IDs to track which points are used
        
        Returns an AlignmentResult with fail-closed status.
        """
        source_points = np.asarray(source_points, dtype=np.float64)
        target_points = np.asarray(target_points, dtype=np.float64)
        
        # 1. Validate Shape & Length
        if source_points.ndim != 2 or source_points.shape[1] != 3:
            return AlignmentResult(status=AlignmentStatus.INVALID_COORDINATES, failure_reason="Source points must be (N, 3).")
        if target_points.ndim != 2 or target_points.shape[1] != 3:
            return AlignmentResult(status=AlignmentStatus.INVALID_COORDINATES, failure_reason="Target points must be (N, 3).")
        if len(source_points) != len(target_points):
            return AlignmentResult(status=AlignmentStatus.INVALID_COORDINATES, failure_reason="Number of source and target points must match.")
            
        N = len(source_points)
        if N < 3:
            return AlignmentResult(status=AlignmentStatus.INSUFFICIENT_POINTS, failure_reason=f"Need at least 3 control points, got {N}.")
            
        # 2. Check for NaN/Inf
        if not (np.isfinite(source_points).all() and np.isfinite(target_points).all()):
            return AlignmentResult(status=AlignmentStatus.INVALID_COORDINATES, failure_reason="NaN or Inf detected in coordinates.")
            
        # 3. Check for exact duplicate source or target points
        # If rank is severely deficient, it's caught later, but exact duplicates can be rejected early.
        if len(np.unique(source_points, axis=0)) < 3 or len(np.unique(target_points, axis=0)) < 3:
            return AlignmentResult(status=AlignmentStatus.DUPLICATE_POINTS, failure_reason="Not enough unique points (duplicates detected).")
            
        # 4. Centroids
        mu_s = np.mean(source_points, axis=0)
        mu_t = np.mean(target_points, axis=0)
        
        # Center points
        P = source_points - mu_s
        Q = target_points - mu_t
        
        # 5. Collinearity / Degeneracy Check
        # If the points lie on a line or plane, the covariance matrix rank < 2
        cov_s = P.T @ P / N
        cov_t = Q.T @ Q / N
        rank_s = np.linalg.matrix_rank(cov_s)
        rank_t = np.linalg.matrix_rank(cov_t)
        
        if rank_s < 2 or rank_t < 2:
            return AlignmentResult(status=AlignmentStatus.COLLINEAR_DEGENERATE, failure_reason="Points are collinear or degenerate. Rank < 2.")
            
        # 6. Umeyama Covariance & SVD
        # H = sum (source * target^T)
        H = P.T @ Q / N
        U, D, Vt = np.linalg.svd(H)
        
        # Rotation
        # R = V * S * U^T
        S = np.eye(3)
        if np.linalg.det(U) * np.linalg.det(Vt) < 0:
            S[2, 2] = -1
            
        R = Vt.T @ S @ U.T
        
        # Scale
        # s = tr(D S) / variance(source)
        var_s = np.trace(cov_s)
        if var_s < 1e-12:
            return AlignmentResult(status=AlignmentStatus.UNSTABLE_SCALE, failure_reason="Source point variance is too small (scale unidentifiable).")
            
        scale = float(np.trace(np.diag(D) @ S) / var_s)
        
        if scale <= 1e-8 or not np.isfinite(scale):
            return AlignmentResult(status=AlignmentStatus.UNSTABLE_SCALE, failure_reason=f"Estimated scale {scale} is invalid/unstable.")
            
        # Translation
        t = mu_t - scale * (R @ mu_s)
        
        # 7. Evaluate residuals on the control points
        transformed = scale * (source_points @ R.T) + t
        residuals = transformed - target_points
        sq_dists = np.sum(residuals**2, axis=1)
        rms = float(np.sqrt(np.mean(sq_dists)))
        
        inlier_ids = point_ids if point_ids is not None else list(range(N))
        
        return AlignmentResult(
            status=AlignmentStatus.SUCCESS,
            scale=scale,
            rotation_matrix=R,
            translation=t,
            transformed_points=transformed,
            control_residuals=residuals,
            rms_control_residual=rms,
            inlier_ids=inlier_ids,
            diagnostics={"num_points": N, "source_variance": float(var_s)}
        )
