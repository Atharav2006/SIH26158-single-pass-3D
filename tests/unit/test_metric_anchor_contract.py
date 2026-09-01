import pytest
from src.reconstruction.reconstruction_result import ReconstructionResult, MetricAnchorCategory

def test_metric_reconstruction_valid():
    res = ReconstructionResult(
        geometry_path="path/to/geom.ply",
        metric=True,
        scale_type="metric",
        coordinate_frame="Local_ENU",
        status="METRIC_RECONSTRUCTION_READY",
        anchor_source=MetricAnchorCategory.CALIBRATED_STEREO,
        provenance="Hardware-synced stereo pair baseline=0.15m"
    )
    assert res.metric is True
    assert res.scale_type == "metric"
    assert res.anchor_source == MetricAnchorCategory.CALIBRATED_STEREO

def test_metric_reconstruction_fails_closed_missing_anchor():
    with pytest.raises(ValueError, match="MUST fail closed if anchor_source is missing"):
        ReconstructionResult(
            geometry_path="path/to/geom.ply",
            metric=True,
            scale_type="metric",
            coordinate_frame="Local_ENU",
            status="METRIC_RECONSTRUCTION_READY",
            provenance="Some provenance"
        )

def test_metric_reconstruction_fails_closed_missing_provenance():
    with pytest.raises(ValueError, match="MUST fail closed if provenance is missing"):
        ReconstructionResult(
            geometry_path="path/to/geom.ply",
            metric=True,
            scale_type="metric",
            coordinate_frame="Local_ENU",
            status="METRIC_RECONSTRUCTION_READY",
            anchor_source=MetricAnchorCategory.LIDAR
        )

def test_metric_reconstruction_rejects_invalid_anchor():
    with pytest.raises(ValueError, match="must be a legitimate MetricAnchorCategory"):
        ReconstructionResult(
            geometry_path="path/to/geom.ply",
            metric=True,
            scale_type="metric",
            coordinate_frame="Local_ENU",
            status="METRIC_RECONSTRUCTION_READY",
            anchor_source="MiDaS_with_GPS", # Invalid enum
            provenance="Not allowed"
        )
