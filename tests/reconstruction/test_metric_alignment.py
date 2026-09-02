import numpy as np
import pytest

from src.reconstruction.metric_alignment import MetricAligner, AlignmentStatus

def test_exact_similarity_transform():
    # A. exact known similarity transform
    # Scale = 2.0, Translation = [10, -5, 3]
    # Rotation: 90 deg around Z
    scale = 2.0
    R = np.array([
        [0, -1, 0],
        [1,  0, 0],
        [0,  0, 1]
    ], dtype=np.float64)
    t = np.array([10.0, -5.0, 3.0])
    
    source = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [1.0, 0.0, 0.0]
    ])
    
    target = scale * (source @ R.T) + t
    
    result = MetricAligner.align(source, target)
    assert result.status == AlignmentStatus.SUCCESS
    assert np.allclose(result.scale, scale)
    assert np.allclose(result.rotation_matrix, R)
    assert np.allclose(result.translation, t)
    assert np.allclose(result.transformed_points, target)
    assert result.rms_control_residual < 1e-12

def test_noisy_control_points():
    # B. noisy control points
    np.random.seed(42)
    source = np.random.rand(10, 3) * 100
    
    scale = 0.5
    R = np.eye(3)
    t = np.array([1.0, 2.0, 3.0])
    
    target = scale * (source @ R.T) + t
    # Add noise
    noise = np.random.randn(10, 3) * 0.01
    target_noisy = target + noise
    
    result = MetricAligner.align(source, target_noisy)
    assert result.status == AlignmentStatus.SUCCESS
    assert abs(result.scale - scale) < 0.05
    assert result.rms_control_residual < 0.05

def test_minimum_valid_control():
    # C. minimum valid control configuration (3 points)
    source = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0]
    ])
    target = source * 3.0 + np.array([1.0, 1.0, 1.0])
    
    result = MetricAligner.align(source, target)
    assert result.status == AlignmentStatus.SUCCESS
    assert np.allclose(result.scale, 3.0)

def test_insufficient_control():
    # D. insufficient control points (<3)
    source = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    target = np.array([[1.0, 1.0, 1.0], [2.0, 1.0, 1.0]])
    
    result = MetricAligner.align(source, target)
    assert result.status == AlignmentStatus.INSUFFICIENT_POINTS

def test_collinear_degenerate():
    # E. collinear/degenerate control points
    # 4 points, but all on the X axis
    source = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [3.0, 0.0, 0.0]
    ])
    target = source * 2.0
    
    result = MetricAligner.align(source, target)
    assert result.status == AlignmentStatus.COLLINEAR_DEGENERATE

def test_duplicate_points():
    # F. duplicate points
    source = np.array([
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0]
    ])
    target = source * 2.0
    
    result = MetricAligner.align(source, target)
    # The unique check will catch this
    assert result.status == AlignmentStatus.DUPLICATE_POINTS

def test_nan_inf():
    # G. NaN/Inf
    source = np.array([
        [0.0, 0.0, 0.0],
        [1.0, np.nan, 0.0],
        [0.0, 1.0, 0.0]
    ])
    target = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0]
    ])
    result = MetricAligner.align(source, target)
    assert result.status == AlignmentStatus.INVALID_COORDINATES

def test_invalid_scale():
    # H. invalid scale (all source points map to extremely low variance)
    source = np.array([
        [0.0, 0.0, 0.0],
        [1e-7, 0.0, 0.0],
        [0.0, 1e-7, 0.0],
        [0.0, 0.0, 1e-7]
    ])
    # Target points spread out, meaning scale would blow up, but variance of source is < 1e-12.
    target = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])
    result = MetricAligner.align(source, target)
    assert result.status == AlignmentStatus.UNSTABLE_SCALE

def test_checkpoint_leakage_prevention():
    # I. checkpoint leakage prevention
    # Verify that transforming points works correctly and that 
    # checkpoint points passed to `transform` did not influence the result.
    source_control = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])
    target_control = source_control * 2.0
    
    result = MetricAligner.align(source_control, target_control)
    assert result.status == AlignmentStatus.SUCCESS
    
    # Checkpoints: these are totally wildly scaled in the "ground truth" 
    # but the transform must apply the *control* scale (2.0)
    source_checkpoint = np.array([[10.0, 10.0, 10.0]])
    
    transformed_checkpoint = result.transform(source_checkpoint)
    
    assert np.allclose(transformed_checkpoint, [[20.0, 20.0, 20.0]])
