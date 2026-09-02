import pytest
import shutil
import json
from pathlib import Path
from argparse import Namespace

from pipelines.application.reconstruct_video import reconstruct_video
from tests.integration.synthetic_texture_video import generate_textured_video
from src.reconstruction.session import ReconstructionSession

@pytest.fixture
def textured_dataset(tmp_path):
    video_path = tmp_path / "textured_smoke.mp4"
    frames_dir = generate_textured_video(video_path, num_frames=10)
    
    calib_path = tmp_path / "synthetic_calib.json"
    calib_path.write_text(json.dumps({"fx": 500, "fy": 500, "cx": 320, "cy": 240}))
    
    # Fake poses matching the translation
    pose_path = tmp_path / "synthetic_poses.csv"
    with open(pose_path, "w") as f:
        f.write("imgid,filename,x_w,y_w,z_w,qx_w,qy_w,qz_w,qw_w\n")
        for i in range(10):
            f.write(f"{i},{i:04d}.jpg,{i*15},{i*5},0,0,0,0,1\n")
            
    yield video_path, frames_dir, calib_path, pose_path
    
def test_b61_case_c_supplied_all(textured_dataset, tmp_path):
    video_path, frames_dir, calib_path, pose_path = textured_dataset
    session_dir = tmp_path / "session_c"
    
    args = Namespace(
        video=str(video_path),
        output=str(session_dir),
        gps=None, imu=None,
        calibration=str(calib_path),
        poses=str(pose_path),
        rtk=None
    )
    result = reconstruct_video(args)
    
    assert result["status"] == "RELATIVE_RECONSTRUCTION_READY"
    assert not result.get("pose_diagnostics")  # Should not have attempted COLMAP
    
def test_b61_case_b_supplied_calib_auto_pose(textured_dataset, tmp_path):
    video_path, frames_dir, calib_path, pose_path = textured_dataset
    session_dir = tmp_path / "session_b"
    
    # We must explicitly copy the frames into the session so COLMAP finds them, 
    # since reconstruct_video currently skips the FFmpeg extraction phase in testing mockup.
    # To properly simulate it, let's manually place frames where the session expects them.
    sess = ReconstructionSession("session_b", str(tmp_path))
    for f in frames_dir.glob("*.jpg"):
        shutil.copy(f, sess.frames_dir / f.name)
        
    args = Namespace(
        video=str(video_path),
        output=str(session_dir),
        gps=None, imu=None,
        calibration=str(calib_path),
        poses=None,  # Pose missing, should trigger COLMAP
        rtk=None
    )
    
    result = reconstruct_video(args)
    
    # COLMAP should run
    # On a synthetic dataset, it might fail (INSUFFICIENT_FEATURES) or succeed
    # Regardless, it should NOT crash
    assert result["status"] in ["RECONSTRUCTION_BLOCKED", "RELATIVE_RECONSTRUCTION_READY"]
    assert "pose_diagnostics" in result
    assert result["pose_diagnostics"]["status"] in ["POSE_ESTIMATION_READY", "POSE_ESTIMATION_FAILED", "POSE_QUALITY_LOW"]
    
def test_b61_case_a_video_only(textured_dataset, tmp_path):
    video_path, frames_dir, calib_path, pose_path = textured_dataset
    session_dir = tmp_path / "session_a"
    
    sess = ReconstructionSession("session_a", str(tmp_path))
    for f in frames_dir.glob("*.jpg"):
        shutil.copy(f, sess.frames_dir / f.name)
        
    args = Namespace(
        video=str(video_path),
        output=str(session_dir),
        gps=None, imu=None,
        calibration=None,  # Calibration missing, should trigger auto
        poses=None,        # Pose missing, should trigger auto
        rtk=None
    )
    
    result = reconstruct_video(args)
    assert result["status"] in ["RECONSTRUCTION_BLOCKED", "RELATIVE_RECONSTRUCTION_READY"]
    assert "pose_diagnostics" in result
