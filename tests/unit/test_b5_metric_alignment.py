import pytest
import numpy as np
from src.depth_fusion.metric_anchor import (
    MetricAnchor,
    AnchorSource,
    CalibrationStatus
)
from src.depth_fusion.depth_scale_alignment import RobustMetricAlignmentEngine

def generate_synthetic_anchors(
    N=50,
    a=0.0005,
    b=0.02,
    num_frames=5,
    noise_std=0.001,
    outlier_ratio=0.0,
    source=AnchorSource.B0_SPARSE_REPROJECTION
):
    np.random.seed(42)
    d_invs = np.random.uniform(100.0, 800.0, size=N)
    inv_zs = a * d_invs + b + np.random.normal(0, noise_std, size=N)
    inv_zs = np.maximum(inv_zs, 1e-4)

    # Add outliers
    if outlier_ratio > 0:
        num_outliers = int(N * outlier_ratio)
        outlier_indices = np.random.choice(N, size=num_outliers, replace=False)
        inv_zs[outlier_indices] += np.random.uniform(0.05, 0.20, size=num_outliers)

    anchors = []
    for i in range(N):
        frame = (i % num_frames) + 1
        anchors.append(MetricAnchor(
            pixel_u=float(np.random.uniform(50, 1870)),
            pixel_v=float(np.random.uniform(50, 1030)),
            frame_id=frame,
            metric_depth_m=float(1.0 / inv_zs[i]),
            inv_depth_predicted=float(d_invs[i]),
            source=source
        ))
    return anchors

def test_robust_recovery_synthetic():
    """
    Verifies that RobustMetricAlignmentEngine recovers true (a, b) on clean and noisy synthetic data.
    """
    true_a, true_b = 0.0005, 0.02
    anchors = generate_synthetic_anchors(N=60, a=true_a, b=true_b, noise_std=0.0005)

    engine = RobustMetricAlignmentEngine(min_anchors=20, min_unique_frames=3)
    raw_inv = np.ones((100, 100), dtype=np.float32) * 400.0
    output = engine.calibrate_depth(raw_inv, anchors)

    assert output.metric is True
    assert output.calibration_status == CalibrationStatus.METRIC_ALIGNMENT_VALID
    assert np.isclose(output.scale_a, true_a, atol=1e-4)
    assert np.isclose(output.shift_b, true_b, atol=2e-3)

def test_ransac_outlier_rejection():
    """
    Verifies that RANSAC filters out up to 30% severe depth outliers.
    """
    true_a, true_b = 0.0005, 0.02
    anchors = generate_synthetic_anchors(N=100, a=true_a, b=true_b, outlier_ratio=0.30)

    engine = RobustMetricAlignmentEngine(min_anchors=30)
    raw_inv = np.ones((100, 100), dtype=np.float32) * 500.0
    output = engine.calibrate_depth(raw_inv, anchors)

    assert output.metric is True
    assert np.isclose(output.scale_a, true_a, atol=1e-4)
    assert np.isclose(output.shift_b, true_b, atol=5e-3)
    assert output.metadata["inlier_stats"]["inlier_ratio"] >= 0.65

def test_ground_truth_source_rejection():
    """
    Verifies that GROUND_TRUTH_EVALUATION_ONLY anchors are strictly ignored
    and rejected in production calibration path.
    """
    gt_anchors = generate_synthetic_anchors(N=50, source=AnchorSource.GROUND_TRUTH_EVALUATION_ONLY)

    engine = RobustMetricAlignmentEngine(min_anchors=10)
    raw_inv = np.ones((50, 50), dtype=np.float32) * 300.0
    output = engine.calibrate_depth(raw_inv, gt_anchors)

    assert output.metric is False
    assert output.calibration_status == CalibrationStatus.METRIC_SCALE_NOT_IDENTIFIABLE
    assert "Insufficient anchors" in output.metadata["reason"]

def test_insufficient_anchors_and_frames():
    """
    Verifies rejection when too few anchors or unique frames are provided.
    """
    engine = RobustMetricAlignmentEngine(min_anchors=20, min_unique_frames=3)
    raw_inv = np.ones((50, 50), dtype=np.float32) * 300.0

    # Few anchors
    few_anchors = generate_synthetic_anchors(N=5, num_frames=3)
    out_few = engine.calibrate_depth(raw_inv, few_anchors)
    assert out_few.metric is False

    # Single frame
    single_frame_anchors = generate_synthetic_anchors(N=30, num_frames=1)
    out_single = engine.calibrate_depth(raw_inv, single_frame_anchors)
    assert out_single.metric is False

def test_low_correlation_rejection():
    """
    Verifies rejection when anchors exhibit near-zero correlation between D_inv and 1/Z.
    """
    np.random.seed(42)
    # Random uncorrelated inverse depths and depths
    anchors = []
    for i in range(50):
        anchors.append(MetricAnchor(
            pixel_u=100.0,
            pixel_v=100.0,
            frame_id=(i % 4) + 1,
            metric_depth_m=float(np.random.uniform(5.0, 30.0)),
            inv_depth_predicted=float(np.random.uniform(100.0, 900.0))
        ))

    engine = RobustMetricAlignmentEngine(min_anchors=20, min_correlation=0.30)
    raw_inv = np.ones((50, 50), dtype=np.float32) * 300.0
    output = engine.calibrate_depth(raw_inv, anchors)

    assert output.metric is False
    assert "correlation" in output.metadata["reason"].lower()

def test_deterministic_behavior():
    """
    Verifies that engine produces bitwise identical results on repeated calls.
    """
    anchors = generate_synthetic_anchors(N=50)
    engine1 = RobustMetricAlignmentEngine(random_seed=123)
    engine2 = RobustMetricAlignmentEngine(random_seed=123)
    raw_inv = np.ones((50, 50), dtype=np.float32) * 400.0

    out1 = engine1.calibrate_depth(raw_inv, anchors)
    out2 = engine2.calibrate_depth(raw_inv, anchors)

    assert out1.scale_a == out2.scale_a
    assert out1.shift_b == out2.shift_b
    assert np.array_equal(out1.depth, out2.depth)
