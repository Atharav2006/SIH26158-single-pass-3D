import os
import sys
import csv
import json
import pytest
from pathlib import Path

# Ensure project root is on sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.ingestion.datasets.zurich_mav import ZurichMAVAdapter
from src.ingestion.synchronization import TemporalSynchronizer
from src.ingestion.dataset_validator import DatasetValidator

@pytest.fixture
def zurich_mav_root():
    root = Path(r"D:\SIH26158\datasets\zurich_mav")
    if not root.exists():
        pytest.skip(f"Zurich MAV dataset root not found at: {root}")
    return root

@pytest.fixture
def parsed_adapter(zurich_mav_root):
    adapter = ZurichMAVAdapter(zurich_mav_root)
    adapter.parse()
    return adapter

def test_dataset_root_validation(zurich_mav_root, tmp_path):
    # 1. Valid root
    adapter = ZurichMAVAdapter(zurich_mav_root)
    assert adapter.validate_root() is True

    # Invalid root
    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()
    bad_adapter = ZurichMAVAdapter(empty_dir)
    with pytest.raises(FileNotFoundError):
        bad_adapter.validate_root()

def test_image_discovery_and_readability(parsed_adapter):
    # 2. Image discovery & 9. Image readability
    assert len(parsed_adapter.images) > 0
    first_img = parsed_adapter.images[0]
    assert "image_id" in first_img
    assert "filename" in first_img
    assert "timestamp_seconds" in first_img
    assert first_img["width"] == 1920
    assert first_img["height"] == 1080

    validator = DatasetValidator(parsed_adapter)
    report = validator.validate()
    img_checks = report["checks"]["images"]
    assert img_checks["readable_images"] > 0
    assert img_checks["corrupted_files"] == 0
    assert img_checks["missing_files"] == 0

def test_gps_parsing(parsed_adapter):
    # 3. GPS parsing
    assert len(parsed_adapter.gps) > 0
    first_gps = parsed_adapter.gps[0]
    assert "timestamp_seconds" in first_gps
    assert "latitude" in first_gps
    assert "longitude" in first_gps
    assert "altitude_if_available" in first_gps

    # Zurich coordinates: Lat ~ 47.38, Lon ~ 8.54
    assert 45.0 <= first_gps["latitude"] <= 50.0
    assert 5.0 <= first_gps["longitude"] <= 10.0
    assert first_gps["altitude_if_available"] is not None

def test_imu_parsing(parsed_adapter):
    # 4. IMU parsing
    assert len(parsed_adapter.imu) > 0
    first_imu = parsed_adapter.imu[0]
    assert "timestamp_seconds" in first_imu
    assert "accel_x" in first_imu
    assert "accel_y" in first_imu
    assert "accel_z" in first_imu
    assert "gyro_x" in first_imu
    assert "gyro_y" in first_imu
    assert "gyro_z" in first_imu

def test_pose_parsing(parsed_adapter):
    # 5. Pose parsing
    assert len(parsed_adapter.pose) > 0
    first_pose = parsed_adapter.pose[0]
    assert "timestamp_seconds" in first_pose
    assert "tx" in first_pose
    assert "ty" in first_pose
    assert "tz" in first_pose
    assert "qx" in first_pose
    assert "qy" in first_pose
    assert "qz" in first_pose
    assert "qw" in first_pose

    # Check quaternion normalization
    qx, qy, qz, qw = first_pose["qx"], first_pose["qy"], first_pose["qz"], first_pose["qw"]
    norm = (qx**2 + qy**2 + qz**2 + qw**2)**0.5
    assert abs(norm - 1.0) < 1e-3

def test_camera_calibration(parsed_adapter):
    # 6. Camera calibration loading
    cam = parsed_adapter.camera
    assert cam is not None
    assert cam["model"] == "pinhole_radial_tangential"
    assert abs(cam["fx"] - 893.39) < 0.1
    assert abs(cam["fy"] - 898.32) < 0.1
    assert abs(cam["cx"] - 951.13) < 0.1
    assert abs(cam["cy"] - 555.13) < 0.1
    assert len(cam["distortion_parameters_if_available"]) == 5

def test_timestamp_parsing_and_monotonicity(parsed_adapter):
    # 7. Timestamp parsing & stream progression
    for stream_name, stream in [("gps", parsed_adapter.gps), ("imu", parsed_adapter.imu), ("pose", parsed_adapter.pose)]:
        assert len(stream) > 0
        first_ts = stream[0]["timestamp_seconds"]
        last_ts = stream[-1]["timestamp_seconds"]
        assert isinstance(first_ts, float)
        assert isinstance(last_ts, float)
        assert first_ts >= 0.0
        assert last_ts > first_ts

        # Verify that >99.9% of timestamps are monotonic (accounting for real-world GPS packet jitter)
        monotonic_count = 0
        prev_ts = -1.0
        for item in stream:
            ts = item["timestamp_seconds"]
            assert isinstance(ts, float)
            if ts >= prev_ts:
                monotonic_count += 1
            prev_ts = ts
        assert (monotonic_count / len(stream)) > 0.999

def test_synchronization(parsed_adapter):
    # 8. Synchronization
    images = parsed_adapter.images
    gps = parsed_adapter.gps

    synced = TemporalSynchronizer.synchronize(images, gps, max_tolerance=0.05)
    assert len(synced) > 0
    for assoc in synced:
        assert "source_timestamp" in assoc
        assert "matched_timestamp" in assoc
        assert "time_difference" in assoc
        assert assoc["time_difference"] <= 0.05

def test_normalized_output_schema(parsed_adapter, tmp_path):
    # 10. Normalized output schema
    out_dir = tmp_path / "normalized_export"
    exported = parsed_adapter.export_normalized(out_dir)

    assert "dataset" in exported and exported["dataset"].exists()
    assert "images" in exported and exported["images"].exists()
    assert "gps" in exported and exported["gps"].exists()
    assert "imu" in exported and exported["imu"].exists()
    assert "pose" in exported and exported["pose"].exists()
    assert "camera" in exported and exported["camera"].exists()

    with open(exported["dataset"], "r", encoding="utf-8") as f:
        meta = json.load(f)
        assert meta["dataset_name"] == "Zurich Urban MAV Dataset (AGZ)"
        assert meta["record_counts"]["gps"] == len(parsed_adapter.gps)
