import pytest
import numpy as np
import torch
from src.depth_fusion.metric_anchor import (
    MetricAnchor,
    AnchorSource,
    CalibrationStatus
)
from src.depth_fusion.depth_scale_alignment import RobustMetricAlignmentEngine
from tests.unit.test_b5_metric_alignment import generate_synthetic_anchors

def test_leave_one_frame_out_cross_validation():
    """
    Integration test:
    Validates leave-one-frame-out parameter stability across multi-frame anchors.
    """
    true_a, true_b = 0.0006, 0.015
    anchors = generate_synthetic_anchors(N=120, a=true_a, b=true_b, num_frames=6, noise_std=0.0003)

    engine = RobustMetricAlignmentEngine(min_anchors=15, min_unique_frames=3)
    loo_report = engine.run_leave_one_frame_out(anchors)

    assert loo_report["status"] == "PASS"
    assert loo_report["total_folds"] == 6
    assert loo_report["passed_folds"] == 6

    # Parameter stability across folds: std should be very small
    assert loo_report["std_scale_a"] < 1e-4
    assert loo_report["std_shift_b"] < 2e-3
    assert np.isclose(loo_report["mean_scale_a"], true_a, atol=1e-4)

def test_train_val_split_evaluation():
    """
    Integration test:
    Splits anchors into 80% train, 20% validation and evaluates independent generalization.
    """
    true_a, true_b = 0.0004, 0.03
    anchors = generate_synthetic_anchors(N=100, a=true_a, b=true_b, num_frames=5, noise_std=0.0004)

    np.random.seed(99)
    indices = np.random.permutation(len(anchors))
    split_idx = int(0.8 * len(anchors))
    train_anchors = [anchors[i] for i in indices[:split_idx]]
    val_anchors = [anchors[i] for i in indices[split_idx:]]

    engine = RobustMetricAlignmentEngine(min_anchors=20)
    raw_inv = np.ones((50, 50), dtype=np.float32) * 500.0

    # Fit on train only
    out = engine.calibrate_depth(raw_inv, train_anchors)
    assert out.metric is True

    # Evaluate on validation only
    d_val = np.array([a.inv_depth_predicted for a in val_anchors])
    inv_z_val = np.array([1.0 / a.metric_depth_m for a in val_anchors])

    pred_inv_val = out.scale_a * d_val + out.shift_b
    pred_z_val = 1.0 / pred_inv_val

    val_rmse_inv = float(np.sqrt(np.mean((inv_z_val - pred_inv_val)**2)))
    val_rmse_m = float(np.sqrt(np.mean((1.0/inv_z_val - pred_z_val)**2)))

    assert val_rmse_inv < 0.01
    assert val_rmse_m < 0.5  # Sub-meter generalization on synthetic validation
