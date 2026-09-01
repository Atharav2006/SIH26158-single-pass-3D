import pytest
import numpy as np
from src.depth_fusion.scale_alignment import (
    MetricIdentifiabilityStatus,
    ScaleAlignmentResult,
    affine_inverse_depth_transform,
    affine_direct_depth_transform,
    RobustScaleEstimator
)

def test_scale_shift_ambiguity_synthetic():
    """
    Synthetic Scale Ambiguity Test:
    Demonstrates that multiple scale/shift parameters (a, b) yield different metric depth
    magnitudes while preserving identical ordinal disparity structures.
    """
    # Synthetic ground truth metric depths (5m to 25m)
    Z_true = np.linspace(5.0, 25.0, 50)
    inv_Z_true = 1.0 / Z_true
    
    # Model 1: Ground truth relationship (a1=1.0, b1=0.0) -> D_inv1 = inv_Z_true
    D_inv1 = inv_Z_true
    
    # Model 2: Arbitrary scale and shift (a2=2.5, b2=0.01) ensuring positive inverse depth
    # inv_Z = a2 * D_inv2 + b2 => D_inv2 = (inv_Z - b2) / a2
    a2, b2 = 2.5, 0.01
    D_inv2 = (inv_Z_true - b2) / a2
    
    # Inverted relative depth
    D_rel1 = 1.0 / (D_inv1 + 1e-6)
    D_rel2 = 1.0 / (D_inv2 + 1e-6)
    
    # Correlation between inverse depths is exactly 1.0 (affine linear relationship)
    corr_inv = np.corrcoef(D_inv1, D_inv2)[0, 1]
    assert np.isclose(corr_inv, 1.0, atol=1e-5)
    
    # Monotonic order of relative depths is strictly preserved
    assert np.all(np.diff(D_rel1) > 0)
    assert np.all(np.diff(D_rel2) > 0)
    
    # But direct unscaled depth is completely wrong
    assert not np.allclose(D_rel2, Z_true)

def test_affine_depth_transform():
    """
    Tests mathematical correctness of affine inverse depth and direct depth transformations.
    """
    D_inv = np.array([100.0, 200.0, 500.0])
    a = 0.001
    b = 0.05
    
    # 1/Z = a * D_inv + b
    # 1/Z = [0.1 + 0.05, 0.2 + 0.05, 0.5 + 0.05] = [0.15, 0.25, 0.55]
    # Z = [6.6667, 4.0, 1.8182]
    Z_expected = 1.0 / np.array([0.15, 0.25, 0.55])
    Z_calc = affine_inverse_depth_transform(D_inv, a, b)
    assert np.allclose(Z_calc, Z_expected, atol=1e-4)

    # Direct affine: Z = s * D_rel + t
    D_rel = np.array([1.0, 2.0, 5.0])
    s = 3.0
    t = 0.5
    Z_direct = affine_direct_depth_transform(D_rel, s, t)
    assert np.allclose(Z_direct, np.array([3.5, 6.5, 15.5]))

def test_robust_scale_estimation_synthetic():
    """
    Verifies that RobustScaleEstimator accurately recovers known (a, b) from anchor points.
    """
    np.random.seed(42)
    estimator = RobustScaleEstimator(min_points=10)
    
    # Ground truth parameters
    true_a = 0.0005
    true_b = 0.02
    
    # Sample synthetic inverse depths
    D_inv_samples = np.random.uniform(50.0, 800.0, size=50)
    inv_Z_samples = true_a * D_inv_samples + true_b
    metric_Z_samples = 1.0 / inv_Z_samples
    
    # Add small observation noise to depth
    noise = np.random.normal(0, 0.001, size=50)
    noisy_inv_Z = np.maximum(inv_Z_samples + noise, 1e-4)
    noisy_Z = 1.0 / noisy_inv_Z
    
    result = estimator.estimate_from_anchors(D_inv_samples, noisy_Z)
    
    assert result.status == MetricIdentifiabilityStatus.IDENTIFIABLE
    assert np.isclose(result.scale_a, true_a, atol=1e-4)
    assert np.isclose(result.shift_b, true_b, atol=1e-3)
    assert result.residual_rmse < 0.01

def test_degeneracy_detection_collinear_points():
    """
    Verifies that degenerate anchor geometry (all anchors at identical depth)
    triggers NOT_IDENTIFIABLE status due to ill-conditioned system.
    """
    estimator = RobustScaleEstimator(min_points=5)
    
    # All anchor points have identical inverse depth (zero variance)
    D_inv_samples = np.array([300.0, 300.0, 300.0, 300.0, 300.0, 300.0])
    metric_Z_samples = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
    
    result = estimator.estimate_from_anchors(D_inv_samples, metric_Z_samples)
    assert result.status == MetricIdentifiabilityStatus.NOT_IDENTIFIABLE
    assert "Degenerate" in result.metadata["reason"]

def test_insufficient_anchor_count():
    """
    Verifies that fewer anchor points than min_points triggers NOT_IDENTIFIABLE.
    """
    estimator = RobustScaleEstimator(min_points=10)
    D_inv_samples = np.array([100.0, 200.0, 300.0])
    metric_Z_samples = np.array([5.0, 10.0, 15.0])
    
    result = estimator.estimate_from_anchors(D_inv_samples, metric_Z_samples)
    assert result.status == MetricIdentifiabilityStatus.NOT_IDENTIFIABLE
    assert "Insufficient anchor points" in result.metadata["reason"]

def test_insufficient_baseline_detection():
    """
    Verifies baseline-to-depth ratio checks for multi-view motion observability.
    """
    estimator = RobustScaleEstimator(min_parallax_ratio=0.05)
    
    # Tiny baseline (0.1m for 20m depth -> B/Z = 0.005 < 0.05) -> NOT_IDENTIFIABLE
    status, b_over_z = estimator.check_motion_observability(baseline_m=0.1, mean_scene_depth_m=20.0)
    assert status == MetricIdentifiabilityStatus.NOT_IDENTIFIABLE
    assert b_over_z == pytest.approx(0.005)
    
    # Moderate baseline (2.0m for 20m depth -> B/Z = 0.10) -> PARTIALLY_IDENTIFIABLE
    status, b_over_z = estimator.check_motion_observability(baseline_m=2.0, mean_scene_depth_m=20.0)
    assert status == MetricIdentifiabilityStatus.PARTIALLY_IDENTIFIABLE
    assert b_over_z == pytest.approx(0.10)
    
    # Strong baseline (6.0m for 20m depth -> B/Z = 0.30) -> IDENTIFIABLE
    status, b_over_z = estimator.check_motion_observability(baseline_m=6.0, mean_scene_depth_m=20.0)
    assert status == MetricIdentifiabilityStatus.IDENTIFIABLE
    assert b_over_z == pytest.approx(0.30)

def test_deterministic_estimation():
    """
    Verifies that estimation is 100% deterministic given identical inputs.
    """
    estimator = RobustScaleEstimator(min_points=5)
    D_inv = np.array([100.0, 250.0, 400.0, 550.0, 700.0])
    Z_metric = np.array([20.0, 15.0, 10.0, 7.0, 5.0])
    
    res1 = estimator.estimate_from_anchors(D_inv, Z_metric)
    res2 = estimator.estimate_from_anchors(D_inv, Z_metric)
    
    assert res1.scale_a == res2.scale_a
    assert res1.shift_b == res2.shift_b
    assert res1.residual_rmse == res2.residual_rmse
