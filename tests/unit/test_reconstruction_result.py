import pytest
from src.reconstruction.reconstruction_result import ReconstructionResult, MetricAnchorCategory

def test_relative_reconstruction_valid():
    res = ReconstructionResult(
        geometry_path="path/to/geom.ply",
        metric=False,
        scale_type="relative",
        coordinate_frame="relative_world_gauge",
        status="RELATIVE_RECONSTRUCTION_READY"
    )
    assert res.metric is False
    assert res.scale_type == "relative"

def test_relative_reconstruction_rejects_metric_anchor():
    with pytest.raises(ValueError, match="cannot declare a metric anchor source"):
        ReconstructionResult(
            geometry_path="path/to/geom.ply",
            metric=False,
            scale_type="relative",
            coordinate_frame="relative_world_gauge",
            status="RELATIVE_RECONSTRUCTION_READY",
            anchor_source=MetricAnchorCategory.LIDAR
        )

def test_relative_reconstruction_rejects_metric_scale_type():
    with pytest.raises(ValueError, match="must specify scale_type='relative'"):
        ReconstructionResult(
            geometry_path="path/to/geom.ply",
            metric=False,
            scale_type="metric",
            coordinate_frame="relative_world_gauge",
            status="RELATIVE_RECONSTRUCTION_READY"
        )
