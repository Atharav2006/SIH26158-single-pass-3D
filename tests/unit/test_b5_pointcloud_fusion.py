import pytest
import numpy as np
from pathlib import Path
from src.depth_fusion.pointcloud_fusion import (
    RelativePointcloud,
    VoxelGridFusion,
    save_pointcloud_ply
)

def test_voxel_grid_fusion_aggregation():
    """
    Verifies that VoxelGridFusion merges duplicate/nearby points, averages confidence,
    and increments support counts accurately.
    """
    fusion = VoxelGridFusion(voxel_size=0.1)

    # Point from Frame 1 at (1.02, 1.03, 1.01) with conf=0.8
    pcd1 = RelativePointcloud(
        points=np.array([[1.02, 1.03, 1.01]], dtype=np.float32),
        colors=np.array([[200, 0, 0]], dtype=np.uint8),
        confidences=np.array([0.8], dtype=np.float32),
        frame_ids=np.array([1], dtype=np.int32),
        support_counts=np.array([1], dtype=np.int32)
    )

    # Point from Frame 2 at (1.04, 1.01, 1.02) with conf=0.6 (same voxel [10, 10, 10])
    pcd2 = RelativePointcloud(
        points=np.array([[1.04, 1.01, 1.02]], dtype=np.float32),
        colors=np.array([[100, 0, 0]], dtype=np.uint8),
        confidences=np.array([0.6], dtype=np.float32),
        frame_ids=np.array([2], dtype=np.int32),
        support_counts=np.array([1], dtype=np.int32)
    )

    fusion.add_pointcloud(pcd1)
    fusion.add_pointcloud(pcd2)

    fused = fusion.extract_fused_pointcloud(min_support=2)

    assert len(fused.points) == 1
    assert fused.support_counts[0] == 2
    # Confidence-weighted position: (1.02*0.8 + 1.04*0.6) / 1.4 = 1.02857
    assert np.isclose(fused.points[0, 0], 1.02857, atol=1e-3)
    assert fused.scale_type == "relative"
    assert fused.metric is False

def test_save_and_verify_ply_header(tmp_path):
    """
    Verifies that saved PLY files contain the explicit non-metric and relative headers.
    """
    pcd = RelativePointcloud(
        points=np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
        colors=np.array([[255, 128, 64]], dtype=np.uint8),
        confidences=np.array([0.9], dtype=np.float32),
        frame_ids=np.array([1], dtype=np.int32),
        support_counts=np.array([3], dtype=np.int32),
        scale_type="relative",
        metric=False
    )

    ply_path = tmp_path / "test_relative.ply"
    save_pointcloud_ply(ply_path, pcd)

    assert ply_path.exists()
    content = ply_path.read_text()
    assert "# SCALE_TYPE: RELATIVE_3D" in content
    assert "# METRIC_CALIBRATED: False" in content
    assert "element vertex 1" in content
