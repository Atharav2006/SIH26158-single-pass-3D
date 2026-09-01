import pytest
from pathlib import Path
from src.reconstruction.session import ReconstructionSession

def test_generic_multi_session_isolation(tmp_path):
    # Two completely distinct generic synthetic video sources
    video_a = tmp_path / "video_a"
    video_b = tmp_path / "video_b"
    
    session_a = ReconstructionSession("session_a", str(video_a))
    session_b = ReconstructionSession("session_b", str(video_b))
    
    # Assert bases are distinct
    assert session_a.base_dir != session_b.base_dir
    
    # Assert child folders are distinct
    for a_dir, b_dir in zip(
        [session_a.inputs_dir, session_a.geometry_dir, session_a.depth_dir],
        [session_b.inputs_dir, session_b.geometry_dir, session_b.depth_dir]
    ):
        assert a_dir != b_dir
        assert not str(a_dir).startswith(str(session_b.base_dir))
        assert not str(b_dir).startswith(str(session_a.base_dir))
