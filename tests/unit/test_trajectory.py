import math
import sys
import json
import pytest
from pathlib import Path

# Ensure project root is on sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.pose.models import Position, Quaternion, Pose
from src.pose.trajectory import Trajectory
from src.pose.coordinate_frames import FRAME_GLOBAL_UTM_ENU, FRAME_LOCAL_ENU

def test_trajectory_metrics_and_statistics():
    p1 = Pose(timestamp_seconds=0.0, position_xyz=Position(0, 0, 0), orientation_xyzw=Quaternion(0, 0, 0, 1), source_frame="Local", target_frame="Cam")
    p2 = Pose(timestamp_seconds=1.0, position_xyz=Position(3, 4, 0), orientation_xyzw=Quaternion(0, 0, 0, 1), source_frame="Local", target_frame="Cam")
    p3 = Pose(timestamp_seconds=2.0, position_xyz=Position(3, 4, 5), orientation_xyzw=Quaternion(0, 0, 0, 1), source_frame="Local", target_frame="Cam")

    traj = Trajectory([p1, p2, p3], frame_id="Local")
    stats = traj.compute_statistics()

    assert stats["pose_count"] == 3
    assert stats["valid_pose_count"] == 3
    assert stats["duration_seconds"] == 2.0
    # Distance: (0,0,0)->(3,4,0) = 5.0m; (3,4,0)->(3,4,5) = 5.0m; Total = 10.0m
    assert abs(stats["trajectory_length_meters"] - 10.0) < 1e-4
    assert abs(stats["mean_speed_mps"] - 5.0) < 1e-4
    assert abs(stats["median_speed_mps"] - 5.0) < 1e-4
    assert abs(stats["max_speed_mps"] - 5.0) < 1e-4

def test_trajectory_validation_rules():
    # 1. Normal valid trajectory
    p1 = Pose(0.0, Position(0, 0, 0), Quaternion(0, 0, 0, 1), "Local", "Cam")
    p2 = Pose(1.0, Position(1, 1, 1), Quaternion(0, 0, 0, 1), "Local", "Cam")
    traj = Trajectory([p1, p2])
    val = traj.validate_trajectory()
    assert val["status"] == "PASS"
    assert val["issues_detected"] == 0

    # 2. Unnormalized quaternion
    p_bad_quat = Pose(2.0, Position(2, 2, 2), Quaternion(1, 1, 1, 1), "Local", "Cam")
    traj_bad_quat = Trajectory([p1, p2, p_bad_quat])
    val_bad_quat = traj_bad_quat.validate_trajectory()
    assert val_bad_quat["status"] == "FAIL"

    # 3. Excessive position discontinuity / speed
    p_jump = Pose(2.0, Position(1000, 1000, 1000), Quaternion(0, 0, 0, 1), "Local", "Cam")
    traj_jump = Trajectory([p1, p2, p_jump])
    val_jump = traj_jump.validate_trajectory()
    assert val_jump["status"] == "FAIL"

def test_trajectory_export(tmp_path):
    p1 = Pose(0.0, Position(0, 0, 0), Quaternion(0, 0, 0, 1), "Local", "Cam")
    p2 = Pose(1.0, Position(1, 2, 3), Quaternion(0, 0, 0, 1), "Local", "Cam")
    traj = Trajectory([p1, p2])

    out_csv = tmp_path / "traj.csv"
    out_json = tmp_path / "traj.json"

    traj.export_csv(out_csv)
    traj.export_json(out_json, extra_metadata={"test_key": "test_val"})

    assert out_csv.exists()
    assert out_json.exists()

    with open(out_json, "r", encoding="utf-8") as f:
        data = json.load(f)
        assert data["test_key"] == "test_val"
        assert data["pose_count"] == 2
