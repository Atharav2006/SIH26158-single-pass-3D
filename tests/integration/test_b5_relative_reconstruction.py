import pytest
import json
import numpy as np
import torch
import cv2
from pathlib import Path
from src.depth_fusion.pointcloud_fusion import (
    unproject_relative_frame,
    VoxelGridFusion,
    RelativePointcloud
)
from src.depth_fusion.depth_quality import compute_depth_confidence
from src.depth_fusion.multiview_consistency import MultiViewConsistencyEvaluator

def test_relative_reconstruction_smoke():
    """
    Integration smoke test:
    Validates end-to-end relative unprojection, multi-view consistency, and voxel fusion
    on a lightweight synthetic multi-frame sequence.
    """
    K_rect = np.array([
        [50.0, 0.0, 25.0],
        [0.0, 50.0, 25.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)

    fusion = VoxelGridFusion(voxel_size=0.05)
    evaluator = MultiViewConsistencyEvaluator(K_rect=K_rect)

    # 3 frames with small camera translation
    pcds = []
    depth_maps = []
    poses = []

    for f_id in range(1, 4):
        rgb = np.full((50, 50, 3), 100 + f_id * 20, dtype=np.uint8)
        # Synthetic inverse depth (plane at depth ~2.0 relative units)
        inv_depth = np.ones((50, 50), dtype=np.float32) * 0.5
        rel_depth = np.ones((50, 50), dtype=np.float32) * 2.0
        depth_maps.append(rel_depth)

        conf, mask, stats = compute_depth_confidence(rgb, inv_depth)

        R_wc = np.eye(3, dtype=np.float32)
        C_w = np.array([f_id * 0.05, 0.0, 0.0], dtype=np.float32)
        poses.append((R_wc, C_w))

        pcd = unproject_relative_frame(
            rgb=rgb,
            rel_depth=rel_depth,
            confidence_map=conf,
            K_rect=K_rect,
            R_wc=R_wc,
            C_world=C_w,
            frame_id=f_id,
            subsample_step=2
        )
        pcds.append(pcd)

    # Evaluate multi-view consistency between frame 1 and frame 2
    weights, inliers, diag = evaluator.evaluate_pair_consistency(
        pcds[0], depth_maps[1], poses[1][0], poses[1][1], image_shape=(50, 50)
    )
    assert diag["projected_inside_count"] > 0

    # Add all to fusion
    for p in pcds:
        fusion.add_pointcloud(p)

    fused = fusion.extract_fused_pointcloud(min_support=1)
    assert len(fused.points) > 0
    assert fused.scale_type == "relative"
    assert fused.metric is False
    assert np.all(np.isfinite(fused.points))
