import pytest
import numpy as np
from src.depth_fusion.pointcloud_fusion import (
    unproject_relative_frame,
    RelativePointcloud
)
from src.depth_fusion.depth_quality import compute_depth_confidence

def test_unproject_relative_frame_basic():
    """
    Verifies that unproject_relative_frame correctly transforms relative depth to 3D points in B2 gauge.
    """
    H, W = 100, 100
    rgb = np.ones((H, W, 3), dtype=np.uint8) * 128
    inv_depth = np.ones((H, W), dtype=np.float32) * 500.0
    rel_depth = 1.0 / inv_depth  # 0.002
    conf = np.ones((H, W), dtype=np.float32)

    K_rect = np.array([
        [50.0, 0.0, 50.0],
        [0.0, 50.0, 50.0],
        [0.0, 0.0, 1.0]
    ])
    R_wc = np.eye(3, dtype=np.float32)
    C_world = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    pcd = unproject_relative_frame(
        rgb=rgb,
        rel_depth=rel_depth,
        confidence_map=conf,
        K_rect=K_rect,
        R_wc=R_wc,
        C_world=C_world,
        frame_id=1,
        subsample_step=1,
        min_confidence=0.1
    )

    assert isinstance(pcd, RelativePointcloud)
    assert len(pcd.points) == H * W
    assert pcd.scale_type == "relative"
    assert pcd.metric is False
    assert np.all(np.isfinite(pcd.points))

    # Center pixel (50, 50) in camera frame has X_c=0, Y_c=0, Z_c=0.002
    # In world frame: (0 + 1.0, 0 + 2.0, 0.002 + 3.0) = (1.0, 2.0, 3.002)
    center_idx = 50 * W + 50
    assert np.allclose(pcd.points[center_idx], np.array([1.0, 2.0, 3.002]), atol=1e-5)

def test_unproject_relative_frame_confidence_filtering():
    """
    Verifies that low confidence or non-positive depth points are filtered out.
    """
    H, W = 10, 10
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    rel_depth = np.ones((H, W), dtype=np.float32) * 0.5
    # Set half points to zero confidence
    conf = np.ones((H, W), dtype=np.float32)
    conf[:5, :] = 0.01  # below min_confidence=0.1

    K_rect = np.eye(3, dtype=np.float32) * 50.0
    K_rect[2, 2] = 1.0
    R_wc = np.eye(3, dtype=np.float32)
    C_world = np.zeros(3, dtype=np.float32)

    pcd = unproject_relative_frame(
        rgb=rgb,
        rel_depth=rel_depth,
        confidence_map=conf,
        K_rect=K_rect,
        R_wc=R_wc,
        C_world=C_world,
        frame_id=1,
        subsample_step=1,
        min_confidence=0.1
    )

    # Exactly 50 points should remain
    assert len(pcd.points) == 50
