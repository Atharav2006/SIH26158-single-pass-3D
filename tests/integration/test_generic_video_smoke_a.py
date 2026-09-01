import pytest
import shutil
import cv2
import numpy as np
from pathlib import Path
from pipelines.application.reconstruct_video import reconstruct_video
from argparse import Namespace

@pytest.fixture
def generic_video_a(tmp_path):
    video_path = tmp_path / "smoke_a.mp4"
    out = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*'mp4v'), 30, (640, 480))
    for i in range(10):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.circle(frame, (320 + i*5, 240), 20, (255, 255, 255), -1)
        out.write(frame)
    out.release()
    yield video_path
    
def test_generic_smoke_a(generic_video_a, tmp_path):
    session_dir = tmp_path / "session_smoke_a"
    
    args = Namespace(
        video=str(generic_video_a),
        output=str(session_dir),
        gps=None,
        imu=None,
        calibration=None,
        poses=None,
        rtk=None
    )
    
    result = reconstruct_video(args)
    
    assert result["status"] == "RECONSTRUCTION_BLOCKED"
    assert "poses" in result["missing_requirements"]
    assert "calibration" in result["missing_requirements"]
    
    # Verify session was created anyway
    assert (session_dir / "inputs").exists()
    assert (session_dir / "diagnostics" / "status.json").exists()
