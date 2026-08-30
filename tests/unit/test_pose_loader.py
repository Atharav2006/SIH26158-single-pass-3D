import os
import sys
import csv
import pytest
from pathlib import Path

# Ensure project root is on sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.pose.models import Position, Quaternion, Pose
from src.pose.pose_loader import (
    load_poses_from_csv,
    load_image_metadata,
    associate_poses_to_images
)
from src.pose.coordinate_frames import FRAME_GLOBAL_UTM_ENU, FRAME_CAMERA_RDF

def test_load_poses_from_csv_valid(tmp_path):
    csv_file = tmp_path / "test_pose.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp_seconds", "tx", "ty", "tz", "qx", "qy", "qz", "qw"])
        writer.writerow([0.0, 10.0, 20.0, 30.0, 0.0, 0.0, 0.0, 1.0])
        writer.writerow([1.0, 15.0, 25.0, 35.0, 0.0, 0.7071, 0.0, 0.7071])

    poses = load_poses_from_csv(csv_file)
    assert len(poses) == 2
    assert poses[0].timestamp_seconds == 0.0
    assert poses[0].position.x == 10.0
    assert poses[0].position.unit == "meter"
    assert poses[0].orientation.qw == 1.0
    assert poses[0].orientation.is_normalized()
    assert poses[1].source_frame == FRAME_GLOBAL_UTM_ENU
    assert poses[1].target_frame == FRAME_CAMERA_RDF

def test_load_poses_from_csv_invalid(tmp_path):
    # Non-existent file
    with pytest.raises(FileNotFoundError):
        load_poses_from_csv(tmp_path / "missing.csv")

    # Malformed row
    bad_csv = tmp_path / "bad.csv"
    with open(bad_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp_seconds", "tx", "ty", "tz", "qx", "qy", "qz", "qw"])
        writer.writerow(["invalid_ts", "not_a_num", 20.0, 30.0, 0.0, 0.0, 0.0, 1.0])

    with pytest.raises(ValueError):
        load_poses_from_csv(bad_csv)

def test_associate_poses_to_images():
    poses = [
        Pose(timestamp_seconds=1.0, position=Position(0, 0, 0), orientation=Quaternion(0, 0, 0, 1), source_frame="A", target_frame="B"),
        Pose(timestamp_seconds=2.0, position=Position(1, 1, 1), orientation=Quaternion(0, 0, 0, 1), source_frame="A", target_frame="B"),
    ]
    images = [
        {"image_id": 1, "filename": "img1.png", "timestamp_seconds": 1.01, "width": 1920, "height": 1080},
        {"image_id": 2, "filename": "img2.png", "timestamp_seconds": 5.00, "width": 1920, "height": 1080},
    ]

    associations = associate_poses_to_images(poses, images, max_tolerance=0.05)
    assert len(associations) == 2

    # First image matches pose at 1.0s (dt = 0.01s <= 0.05s)
    assert associations[0]["matched_pose_timestamp"] == 1.0
    assert abs(associations[0]["time_difference"] - 0.01) < 1e-4
    assert associations[0]["pose"] is not None

    # Second image is unmatched (dt = 3.0s > 0.05s)
    assert associations[1]["matched_pose_timestamp"] is None
    assert associations[1]["pose"] is None
