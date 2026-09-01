"""
SIH26158 Depth Fusion - Confidence-Aware Relative Point Cloud Fusion

This module implements relative 3D unprojection, structured point provenance,
voxel-based multi-frame fusion, duplicate suppression, and PLY export.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import numpy as np

@dataclass
class RelativePointcloud:
    """
    Structured point cloud container explicitly tagged as relative geometry in B2 gauge.
    """
    points: np.ndarray        # [N, 3] float32
    colors: np.ndarray        # [N, 3] uint8
    confidences: np.ndarray   # [N] float32 in [0, 1]
    frame_ids: np.ndarray     # [N] int32
    support_counts: np.ndarray# [N] int32
    scale_type: str = "relative"
    metric: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.points)

    def is_empty(self) -> bool:
        return len(self.points) == 0

def unproject_relative_frame(
    rgb: np.ndarray,
    rel_depth: np.ndarray,
    confidence_map: np.ndarray,
    K_rect: np.ndarray,
    R_wc: np.ndarray,
    C_world: np.ndarray,
    frame_id: int,
    subsample_step: int = 2,
    min_confidence: float = 0.05,
    valid_mask: Optional[np.ndarray] = None
) -> RelativePointcloud:
    """
    Unprojects a single rectified frame with relative depth into 3D world space (in B2 gauge).
    
    Formula:
        X_c = (u - cx) * Z_rel / fx
        Y_c = (v - cy) * Z_rel / fy
        Z_c = Z_rel
        X_w = R_wc @ X_c + C_world
    """
    H, W = rel_depth.shape
    fx = K_rect[0, 0]
    fy = K_rect[1, 1]
    cx = K_rect[0, 2]
    cy = K_rect[1, 2]

    # Subsample grid for computational efficiency and dense coverage balance
    u_idx = np.arange(0, W, subsample_step, dtype=np.int32)
    v_idx = np.arange(0, H, subsample_step, dtype=np.int32)
    uu, vv = np.meshgrid(u_idx, v_idx)

    depth_sub = rel_depth[vv, uu]
    conf_sub = confidence_map[vv, uu]
    rgb_sub = rgb[vv, uu]

    # Filter mask
    valid = np.isfinite(depth_sub) & (depth_sub > 0) & (conf_sub >= min_confidence)
    if valid_mask is not None:
        valid &= valid_mask[vv, uu]

    u_val = uu[valid]
    v_val = vv[valid]
    z_val = depth_sub[valid]
    c_val = conf_sub[valid]
    rgb_val = rgb_sub[valid]

    if len(z_val) == 0:
        return RelativePointcloud(
            points=np.zeros((0, 3), dtype=np.float32),
            colors=np.zeros((0, 3), dtype=np.uint8),
            confidences=np.zeros((0,), dtype=np.float32),
            frame_ids=np.zeros((0,), dtype=np.int32),
            support_counts=np.zeros((0,), dtype=np.int32),
            scale_type="relative",
            metric=False,
            metadata={"frame_id": frame_id, "status": "EMPTY"}
        )

    # Unproject to camera frame
    x_c = (u_val - cx) * z_val / fx
    y_c = (v_val - cy) * z_val / fy
    z_c = z_val
    pts_c = np.stack([x_c, y_c, z_c], axis=-1).astype(np.float32)

    # Transform to world coordinates: X_w = pts_c @ R_wc.T + C_world
    pts_w = (pts_c @ R_wc.T) + C_world.reshape(1, 3).astype(np.float32)

    return RelativePointcloud(
        points=pts_w.astype(np.float32),
        colors=rgb_val.astype(np.uint8),
        confidences=c_val.astype(np.float32),
        frame_ids=np.full(len(z_val), frame_id, dtype=np.int32),
        support_counts=np.ones(len(z_val), dtype=np.int32),
        scale_type="relative",
        metric=False,
        metadata={"frame_id": frame_id, "point_count": len(z_val)}
    )

class VoxelGridFusion:
    """
    Streaming voxel-based fusion engine with confidence weighting,
    support counting, and duplicate suppression.
    """
    def __init__(self, voxel_size: float = 0.05):
        self.voxel_size = voxel_size
        self.voxel_map: Dict[Tuple[int, int, int], Dict[str, Any]] = {}

    def add_pointcloud(self, pcd: RelativePointcloud):
        """
        Integrates a RelativePointcloud into the spatial voxel hash grid.
        Vectorized with np.unique and np.bincount for high-performance streaming.
        """
        if pcd.is_empty():
            return

        inv_vox = 1.0 / self.voxel_size
        voxel_indices = np.floor(pcd.points * inv_vox).astype(np.int64)
        f_id = int(pcd.frame_ids[0]) if len(pcd.frame_ids) > 0 else 0

        # Unique voxels in current frame point cloud
        unq_keys, inverse_idx = np.unique(voxel_indices, axis=0, return_inverse=True)
        K = len(unq_keys)
        weights = pcd.confidences.astype(np.float64)

        # Vectorized weighted sums per unique voxel
        sum_w_k = np.bincount(inverse_idx, weights=weights, minlength=K)
        sum_wx_k = np.bincount(inverse_idx, weights=weights * pcd.points[:, 0], minlength=K)
        sum_wy_k = np.bincount(inverse_idx, weights=weights * pcd.points[:, 1], minlength=K)
        sum_wz_k = np.bincount(inverse_idx, weights=weights * pcd.points[:, 2], minlength=K)

        sum_wr_k = np.bincount(inverse_idx, weights=weights * pcd.colors[:, 0], minlength=K)
        sum_wg_k = np.bincount(inverse_idx, weights=weights * pcd.colors[:, 1], minlength=K)
        sum_wb_k = np.bincount(inverse_idx, weights=weights * pcd.colors[:, 2], minlength=K)

        pts_count_k = np.bincount(inverse_idx, minlength=K)

        for k in range(K):
            key = (int(unq_keys[k, 0]), int(unq_keys[k, 1]), int(unq_keys[k, 2]))
            w_sum = sum_w_k[k]
            n_pts = int(pts_count_k[k])
            pos_sum = np.array([sum_wx_k[k], sum_wy_k[k], sum_wz_k[k]], dtype=np.float64)
            col_sum = np.array([sum_wr_k[k], sum_wg_k[k], sum_wb_k[k]], dtype=np.float64)

            if key not in self.voxel_map:
                self.voxel_map[key] = {
                    "sum_w": w_sum,
                    "weighted_pos": pos_sum,
                    "weighted_col": col_sum,
                    "max_conf": w_sum / max(1, n_pts),
                    "total_points": n_pts,
                    "support_count": 1,
                    "frame_ids": {f_id}
                }
            else:
                vox = self.voxel_map[key]
                vox["sum_w"] += w_sum
                vox["weighted_pos"] += pos_sum
                vox["weighted_col"] += col_sum
                vox["total_points"] += n_pts
                vox["frame_ids"].add(f_id)
                vox["support_count"] = len(vox["frame_ids"])

    def extract_fused_pointcloud(
        self,
        min_support: int = 1,
        min_confidence: float = 0.0,
        mode: str = "RELATIVE_CONFIDENT"
    ) -> RelativePointcloud:
        """
        Extracts aggregated RelativePointcloud with confidence and support filters.
        """
        fused_pts = []
        fused_cols = []
        fused_confs = []
        fused_supports = []

        for key, vox in self.voxel_map.items():
            if vox["support_count"] < min_support:
                continue
            total_pts = vox.get("total_points", vox["support_count"])
            mean_conf = float(np.clip(vox["sum_w"] / max(1, total_pts), 0.0, 1.0))
            if mean_conf < min_confidence:
                continue

            sum_w = max(1e-6, vox["sum_w"])
            pos = (vox["weighted_pos"] / sum_w).astype(np.float32)
            col = np.clip(vox["weighted_col"] / sum_w, 0, 255).astype(np.uint8)

            fused_pts.append(pos)
            fused_cols.append(col)
            fused_confs.append(mean_conf)
            fused_supports.append(vox["support_count"])

        if len(fused_pts) == 0:
            return RelativePointcloud(
                points=np.zeros((0, 3), dtype=np.float32),
                colors=np.zeros((0, 3), dtype=np.uint8),
                confidences=np.zeros((0,), dtype=np.float32),
                frame_ids=np.zeros((0,), dtype=np.int32),
                support_counts=np.zeros((0,), dtype=np.int32),
                scale_type="relative",
                metric=False,
                metadata={"mode": mode, "point_count": 0}
            )

        return RelativePointcloud(
            points=np.array(fused_pts, dtype=np.float32),
            colors=np.array(fused_cols, dtype=np.uint8),
            confidences=np.array(fused_confs, dtype=np.float32),
            frame_ids=np.zeros(len(fused_pts), dtype=np.int32),
            support_counts=np.array(fused_supports, dtype=np.int32),
            scale_type="relative",
            metric=False,
            metadata={
                "mode": mode,
                "voxel_size": self.voxel_size,
                "total_voxels": len(self.voxel_map),
                "fused_point_count": len(fused_pts),
                "min_support_applied": min_support,
                "min_confidence_applied": min_confidence
            }
        )

def save_pointcloud_ply(
    path: Path,
    pcd: RelativePointcloud,
    extra_comments: Optional[List[str]] = None
):
    """
    Saves a RelativePointcloud as an ASCII/binary PLY file with explicit scale metadata headers.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    N = len(pcd.points)

    header = [
        "ply",
        "format ascii 1.0",
        "# SCALE_TYPE: RELATIVE_3D",
        f"# METRIC_CALIBRATED: {pcd.metric}",
        "# PROVENANCE: SIH26158_B5_RELATIVE_RECONSTRUCTION",
        f"# TOTAL_POINTS: {N}"
    ]
    if extra_comments:
        header.extend([f"# {c}" for c in extra_comments])

    header.extend([
        f"element vertex {N}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "property float confidence",
        "property int support_count",
        "end_header\n"
    ])

    with open(path, "w") as f:
        f.write("\n".join(header))
        for i in range(N):
            p = pcd.points[i]
            c = pcd.colors[i]
            conf = pcd.confidences[i]
            sup = pcd.support_counts[i]
            f.write(f"{p[0]:.5f} {p[1]:.5f} {p[2]:.5f} {int(c[0])} {int(c[1])} {int(c[2])} {conf:.4f} {int(sup)}\n")
