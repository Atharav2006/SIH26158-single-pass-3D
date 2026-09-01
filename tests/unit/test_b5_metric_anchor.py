import pytest
import numpy as np
from src.depth_fusion.metric_anchor import (
    MetricAnchor,
    AnchorSource,
    CalibrationStatus,
    MetricDepthOutput
)

def test_metric_anchor_valid_creation():
    """
    Verifies valid MetricAnchor construction with all required fields.
    """
    anc = MetricAnchor(
        pixel_u=960.0,
        pixel_v=540.0,
        frame_id=1,
        metric_depth_m=18.5,
        inv_depth_predicted=450.0,
        confidence=0.95,
        source=AnchorSource.B0_SPARSE_REPROJECTION
    )
    assert anc.pixel_u == 960.0
    assert anc.metric_depth_m == 18.5
    assert anc.source == AnchorSource.B0_SPARSE_REPROJECTION

def test_metric_anchor_invalid_depth_rejection():
    """
    Verifies that negative or zero depths are strictly rejected.
    """
    with pytest.raises(ValueError, match="strictly positive"):
        MetricAnchor(
            pixel_u=100.0,
            pixel_v=100.0,
            frame_id=1,
            metric_depth_m=-5.0,
            inv_depth_predicted=200.0
        )

    with pytest.raises(ValueError, match="strictly positive"):
        MetricAnchor(
            pixel_u=100.0,
            pixel_v=100.0,
            frame_id=1,
            metric_depth_m=10.0,
            inv_depth_predicted=-10.0
        )

def test_metric_depth_output_contract():
    """
    Verifies typed contract of MetricDepthOutput.
    """
    dummy_depth = np.ones((100, 100), dtype=np.float32) * 15.0
    dummy_conf = np.ones((100, 100), dtype=np.float32)

    # Relative uncalibrated output
    out_rel = MetricDepthOutput(
        depth=dummy_depth,
        confidence=dummy_conf,
        metric=False,
        calibration_status=CalibrationStatus.METRIC_SCALE_NOT_IDENTIFIABLE
    )
    assert not out_rel.is_calibrated()
    assert out_rel.scale_a is None

    # Calibrated metric output
    out_metric = MetricDepthOutput(
        depth=dummy_depth,
        confidence=dummy_conf,
        metric=True,
        scale_a=0.0005,
        shift_b=0.02,
        calibration_status=CalibrationStatus.METRIC_ALIGNMENT_VALID
    )
    assert out_metric.is_calibrated()
    assert out_metric.scale_a == 0.0005
