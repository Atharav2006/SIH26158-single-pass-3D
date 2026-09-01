import pytest
from pathlib import Path
from src.reconstruction.session import ReconstructionSession
import shutil

@pytest.fixture
def temp_workspace(tmp_path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    yield str(ws)
    shutil.rmtree(ws, ignore_errors=True)

def test_session_isolation(temp_workspace):
    session_a = ReconstructionSession("VideoA", temp_workspace)
    session_b = ReconstructionSession("VideoB", temp_workspace)
    
    assert session_a.session_id == "VideoA"
    assert session_b.session_id == "VideoB"
    
    # Check disjoint paths
    assert session_a.base_dir != session_b.base_dir
    assert str(session_a.base_dir) not in str(session_b.base_dir)
    
    # Check directory creation
    assert session_a.inputs_dir.exists()
    assert session_a.frames_dir.exists()
    assert session_a.geometry_dir.exists()
    
    assert session_b.inputs_dir.exists()
    assert session_b.geometry_dir.exists()

def test_get_path_resolves_correctly(temp_workspace):
    session = ReconstructionSession("TestVideo", temp_workspace)
    p = session.get_path("some_file.json")
    assert "TestVideo" in str(p)
    assert p.name == "some_file.json"
