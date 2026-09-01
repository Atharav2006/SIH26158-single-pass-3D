import pytest
import json
from pathlib import Path
from src.reconstruction.reconstruction_result import ReconstructionResult, MetricAnchorCategory
from src.reconstruction.session import ReconstructionSession
import shutil

@pytest.fixture
def temp_workspace(tmp_path):
    ws = tmp_path / "integration_ws"
    ws.mkdir()
    yield str(ws)
    shutil.rmtree(ws, ignore_errors=True)

def test_integration_relative_mode_session(temp_workspace):
    # Simulating Video A (No metric anchors)
    session = ReconstructionSession("VideoA", temp_workspace)
    
    # Process geometry...
    geom_file = session.get_path("geometry/pointcloud.ply")
    geom_file.parent.mkdir(parents=True, exist_ok=True)
    geom_file.write_text("dummy ply content")
    
    # Generate Contract Result
    result = ReconstructionResult(
        geometry_path=str(geom_file),
        metric=False,
        scale_type="relative",
        coordinate_frame="relative_world_gauge",
        status="RELATIVE_RECONSTRUCTION_READY"
    )
    
    assert result.metric is False
    assert "geometry" in result.geometry_path

def test_integration_metric_mode_session(temp_workspace):
    # Simulating Video B (Has LiDAR anchors)
    session = ReconstructionSession("VideoB", temp_workspace)
    
    geom_file = session.get_path("geometry/pointcloud.ply")
    geom_file.parent.mkdir(parents=True, exist_ok=True)
    geom_file.write_text("dummy ply content")
    
    result = ReconstructionResult(
        geometry_path=str(geom_file),
        metric=True,
        scale_type="metric",
        coordinate_frame="Local_ENU",
        status="METRIC_RECONSTRUCTION_READY",
        anchor_source=MetricAnchorCategory.LIDAR,
        provenance="Registered point cloud from Velodyne VLP-16"
    )
    
    assert result.metric is True
    assert result.anchor_source == MetricAnchorCategory.LIDAR

def test_integration_b5_2_fails_metric_contract():
    # Prove that B5.2 Global Gauge cannot masquerade as metric
    with pytest.raises(ValueError, match="MUST fail closed if anchor_source is missing"):
        ReconstructionResult(
            geometry_path="path/to/b5.2_gauge.ply",
            metric=True,
            scale_type="metric",
            coordinate_frame="Local_ENU",
            status="METRIC_RECONSTRUCTION_READY",
            anchor_source=None, # Global gauge lacks an external anchor
            provenance="B5.2 Global Scale (Drifted)"
        )
