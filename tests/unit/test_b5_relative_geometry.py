import pytest
import numpy as np
from src.depth_fusion.pointcloud_fusion import RelativePointcloud
from src.depth_fusion.unprojection import unproject_to_3d

def test_relative_geometry_metadata_and_metric_guardrail():
    """
    Verifies that RelativePointcloud explicitly marks metric=False and scale_type='relative',
    and that uncalibrated relative depth CANNOT be passed into metric-only unprojection functions.
    """
    pcd = RelativePointcloud(
        points=np.ones((10, 3), dtype=np.float32),
        colors=np.zeros((10, 3), dtype=np.uint8),
        confidences=np.ones(10, dtype=np.float32),
        frame_ids=np.ones(10, dtype=np.int32),
        support_counts=np.ones(10, dtype=np.int32),
        scale_type="relative",
        metric=False
    )

    assert pcd.metric is False
    assert pcd.scale_type == "relative"

    # Verify that unproject_to_3d strictly blocks uncalibrated relative depth
    import torch
    dummy_depth_t = torch.ones((5, 5), dtype=torch.float32)
    K = np.eye(3)
    R = np.eye(3)
    C = np.zeros(3)

    with pytest.raises(ValueError, match="not metrically calibrated"):
        unproject_to_3d(dummy_depth_t, K, R, C, is_metric=False)
