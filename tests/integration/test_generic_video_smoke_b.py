import pytest
import shutil
import cv2
import numpy as np
import json
from pathlib import Path
from pipelines.application.reconstruct_video import reconstruct_video
from argparse import Namespace

@pytest.fixture
def generic_inputs_b(tmp_path):
    video_path = tmp_path / "smoke_b.mp4"
    out = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*'mp4v'), 30, (640, 480))
    for i in range(10):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.circle(frame, (320 + i*5, 240), 20, (255, 255, 255), -1)
        out.write(frame)
    out.release()
    
    calib_path = tmp_path / "calib_b.json"
    calib_path.write_text(json.dumps({"fx": 500, "fy": 500, "cx": 320, "cy": 240}))
    
    pose_path = tmp_path / "poses_b.csv"
    pose_path.write_text("x,y,z,qx,qy,qz,qw\n0,0,0,0,0,0,1")
    
    yield video_path, calib_path, pose_path

def test_generic_smoke_b(generic_inputs_b, tmp_path):
    video_path, calib_path, pose_path = generic_inputs_b
    session_dir = tmp_path / "session_smoke_b"
    
    args = Namespace(
        video=str(video_path),
        output=str(session_dir),
        gps=None,
        imu=None,
        calibration=str(calib_path),
        poses=str(pose_path),
        rtk=None
    )
    
    result = reconstruct_video(args)
    
    assert result["status"] == "RELATIVE_RECONSTRUCTION_READY"
    assert result["metric"] is False
    assert result["scale_type"] == "relative"
    
    # Verify session isolation
    assert (session_dir / "geometry" / "pointcloud.ply").exists()
    assert (session_dir / "exports" / "reconstruction_summary.json").exists()
