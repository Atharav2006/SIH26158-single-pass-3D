import numpy as np
from typing import Tuple, Dict, Any, Union

def umeyama_alignment(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    with_scale: bool = True
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Closed-form Umeyama similarity alignment (Sim(3) or SE(3)) between two point sets.
    
    Finds s, R, t that minimizes:
      1/N * sum_i || dst_pts[i] - (s * R * src_pts[i] + t) ||^2
      
    Args:
        src_pts: (N, 3) numpy array of source points (e.g. COLMAP camera centers).
        dst_pts: (N, 3) numpy array of target ground-truth points.
        with_scale: If True, computes optimal scale s (Sim(3)); if False, enforces s = 1.0 (SE(3)).
        
    Returns:
        s: Optimal scale factor (float).
        R: (3, 3) optimal rotation matrix with det(R) = +1.
        t: (3,) optimal translation vector.
        aligned_pts: (N, 3) array of transformed source points: s * src_pts @ R.T + t.
    """
    assert src_pts.shape == dst_pts.shape, f"Shape mismatch: {src_pts.shape} vs {dst_pts.shape}"
    n, m = src_pts.shape
    assert m == 3, f"Expected 3D points, got shape {src_pts.shape}"
    assert n >= 3, f"Need at least 3 non-collinear points for 3D alignment, got {n}"

    # 1. Compute Centroids
    src_mean = np.mean(src_pts, axis=0)
    dst_mean = np.mean(dst_pts, axis=0)

    # 2. De-meaned point clouds
    src_d = src_pts - src_mean
    dst_d = dst_pts - dst_mean

    # 3. Variance of source points
    src_var = np.mean(np.sum(src_d ** 2, axis=1))
    if src_var < 1e-12:
        raise ValueError("Source points are degenerate (zero variance).")

    # 4. Cross-covariance matrix
    # Sigma_xy = 1/N * (dst_d^T * src_d)
    sigma_xy = (dst_d.T @ src_d) / n

    # 5. Singular Value Decomposition
    U, D, Vt = np.linalg.svd(sigma_xy)

    # 6. Proper rotation matrix (ensuring det(R) = +1 to prevent reflection)
    S = np.eye(m)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1.0

    R = U @ S @ Vt

    # 7. Scale computation
    if with_scale:
        s = float(np.trace(np.diag(D) @ S) / src_var)
    else:
        s = 1.0

    # 8. Translation
    t = dst_mean - s * (R @ src_mean)

    # 9. Transform points: s * (R @ x) + t  <=>  s * src_pts @ R.T + t
    aligned_pts = s * (src_pts @ R.T) + t

    return s, R, t, aligned_pts

def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """
    Convert 3x3 rotation matrix to quaternion in Hamilton scalar-last format: [qx, qy, qz, qw].
    """
    tr = np.trace(R)
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s

    q = np.array([qx, qy, qz, qw], dtype=float)
    norm = np.linalg.norm(q)
    return q / norm if norm > 1e-12 else np.array([0.0, 0.0, 0.0, 1.0])

def quaternion_to_rotation_matrix(q: Union[np.ndarray, list, tuple]) -> np.ndarray:
    """
    Convert quaternion in Hamilton scalar-last format [qx, qy, qz, qw] to 3x3 rotation matrix.
    """
    qx, qy, qz, qw = q
    norm = np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    if norm < 1e-12:
        return np.eye(3)
    qx, qy, qz, qw = qx/norm, qy/norm, qz/norm, qw/norm

    return np.array([
        [1.0 - 2.0*(qy*qy + qz*qz), 2.0*(qx*qy - qz*qw),       2.0*(qx*qz + qy*qw)],
        [2.0*(qx*qy + qz*qw),       1.0 - 2.0*(qx*qx + qz*qz), 2.0*(qy*qz - qx*qw)],
        [2.0*(qx*qz - qy*qw),       2.0*(qy*qz + qx*qw),       1.0 - 2.0*(qx*qx + qy*qy)]
    ], dtype=float)
