import os
import sys
import json
import pytest
from pathlib import Path

# Ensure project root is on sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.pose.pose_loader import load_poses_from_csv, load_image_metadata, associate_poses_to_images
from src.pose.trajectory import Trajectory
from src.pose.coordinate_frames import FRAME_LOCAL_ENU
from src.visualization.trajectory_plot import plot_topdown_trajectory, plot_3d_trajectory

@pytest.fixture
def normalized_dir():
    p = Path("datasets/normalized/zurich_mav_sample").resolve()
    if not (p / "pose.csv").exists():
        p = Path("outputs/reports/zurich_mav").resolve()
    if not (p / "pose.csv").exists():
        pytest.skip(f"Normalized Zurich MAV dataset not found in {p}")
    return p

def test_zurich_pose_loading_and_quaternions(normalized_dir):
    pose_csv = normalized_dir / "pose.csv"
    poses = load_poses_from_csv(pose_csv)

    # 1. Pose parsing
    assert len(poses) == 2708

    # 2. Quaternion parsing & normalization
    for p in poses:
        assert p.orientation.is_normalized(tol=1e-3)
        assert p.position.unit == "meter"

def test_zurich_trajectory_conversion_and_statistics(normalized_dir, tmp_path):
    pose_csv = normalized_dir / "pose.csv"
    poses = load_poses_from_csv(pose_csv)

    traj_utm = Trajectory.from_poses(poses)
    traj_local = traj_utm.to_local_enu()

    assert traj_local.frame_id == FRAME_LOCAL_ENU
    assert traj_local.poses[0].position.x == 0.0
    assert traj_local.poses[0].position.y == 0.0
    assert traj_local.poses[0].position.z == 0.0

    stats = traj_local.compute_statistics()
    # 7. Trajectory statistics
    assert stats["pose_count"] == 2708
    assert stats["valid_pose_count"] == 2708
    assert 2690.0 <= stats["duration_seconds"] <= 2710.0
    assert 1900.0 <= stats["trajectory_length_meters"] <= 1950.0
    assert 0.5 <= stats["average_speed_mps"] <= 1.0

    # 8. Output schema verification
    out_csv = tmp_path / "traj.csv"
    out_json = tmp_path / "traj.json"
    traj_local.export_csv(out_csv)
    traj_local.export_json(out_json)

    assert out_csv.exists()
    assert out_json.exists()

    with open(out_json, "r", encoding="utf-8") as f:
        meta = json.load(f)
        assert "spatial_extent" in meta
        assert "trajectory_length_meters" in meta

def test_image_pose_synchronization(normalized_dir):
    pose_csv = normalized_dir / "pose.csv"
    images_csv = normalized_dir / "images.csv"

    if not images_csv.exists():
        pytest.skip("images.csv not found")

    poses = load_poses_from_csv(pose_csv)
    images = load_image_metadata(images_csv)

    associations = associate_poses_to_images(poses, images, max_tolerance=0.05)
    assert len(associations) == len(images)
    # 6. No silent timestamp loss
    matched = [a for a in associations if a["matched_pose_timestamp"] is not None]
    assert len(matched) > 0
    for m in matched:
        assert m["time_difference"] <= 0.05

def test_trajectory_visualization_generation(normalized_dir, tmp_path):
    pose_csv = normalized_dir / "pose.csv"
    poses = load_poses_from_csv(pose_csv)
    traj_local = Trajectory.from_poses(poses).to_local_enu()

    topdown_png = tmp_path / "topdown.png"
    three_d_png = tmp_path / "3d.png"

    plot_topdown_trajectory(traj_local, topdown_png)
    plot_3d_trajectory(traj_local, three_d_png)

    assert topdown_png.exists() and topdown_png.stat().st_size > 1000
    assert three_d_png.exists() and three_d_png.stat().st_size > 1000
