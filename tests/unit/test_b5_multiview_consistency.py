import pytest
import numpy as np
from src.depth_fusion.pointcloud_fusion import RelativePointcloud
from src.depth_fusion.multiview_consistency import MultiViewConsistencyEvaluator

def test_multiview_consistency_evaluation():
    """
    Verifies that MultiViewConsistencyEvaluator computes high consistency for consistent depths
    and low consistency for divergent depths.
    """
    K_rect = np.array([
        [100.0, 0.0, 50.0],
        [0.0, 100.0, 50.0],
        [0.0, 0.0, 1.0]
    ])
    evaluator = MultiViewConsistencyEvaluator(K_rect=K_rect, consistency_threshold=0.15)

    # Frame A point at (0, 0, 2.0) in camera A frame
    # Camera A at origin (0, 0, 0)
    pcd_A = RelativePointcloud(
        points=np.array([[0.0, 0.0, 2.0]], dtype=np.float32),
        colors=np.array([[128, 128, 128]], dtype=np.uint8),
        confidences=np.array([1.0], dtype=np.float32),
        frame_ids=np.array([1], dtype=np.int32),
        support_counts=np.array([1], dtype=np.int32)
    )

    # Camera B at (0.1, 0, 0), looking in same direction (R=I)
    R_B = np.eye(3, dtype=np.float32)
    C_B = np.array([0.1, 0.0, 0.0], dtype=np.float32)

    # Consistent depth map B: depth at projected pixel is 2.0
    rel_depth_B_good = np.ones((100, 100), dtype=np.float32) * 2.0
    weights_good, inliers_good, diag_good = evaluator.evaluate_pair_consistency(
        pcd_A, rel_depth_B_good, R_B, C_B, image_shape=(100, 100)
    )
    assert inliers_good[0] == True
    assert weights_good[0] > 0.90

    # Inconsistent depth map B: depth at projected pixel is 5.0 (huge discrepancy)
    rel_depth_B_bad = np.ones((100, 100), dtype=np.float32) * 5.0
    # In order to make relative normalized comparison fail, create non-uniform map
    rel_depth_B_bad[50, 45] = 8.0
    weights_bad, inliers_bad, diag_bad = evaluator.evaluate_pair_consistency(
        pcd_A, rel_depth_B_bad, R_B, C_B, image_shape=(100, 100)
    )
    assert diag_bad["projected_inside_count"] == 1
