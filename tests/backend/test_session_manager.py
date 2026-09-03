import pytest
import os
import uuid
import tempfile
from pathlib import Path
from src.backend.session_manager import BackendSessionManager, SessionManagerError

@pytest.fixture
def temp_workspace():
    with tempfile.TemporaryDirectory() as d:
        yield d

@pytest.fixture
def manager(temp_workspace):
    return BackendSessionManager(base_workspace_dir=temp_workspace)

def test_create_session(manager):
    session_id = manager.create_session({"test_key": "test_value"})
    
    # Verify ID is a valid UUID
    assert str(uuid.UUID(session_id)) == session_id
    
    workspace = manager.get_session_workspace(session_id)
    assert workspace.exists()
    
    for d in ["inputs", "temp", "outputs", "metadata", "logs"]:
        assert (workspace / d).exists()
        assert (workspace / d).is_dir()
        
    meta_file = workspace / "metadata" / "session_info.json"
    assert meta_file.exists()

def test_session_exists(manager):
    session_id = manager.create_session()
    assert manager.session_exists(session_id) is True
    assert manager.session_exists(str(uuid.uuid4())) is False

def test_get_session(manager):
    session_id = manager.create_session({"my_key": "val"})
    data = manager.get_session(session_id)
    assert data["session_id"] == session_id
    assert data["status"] == "created"
    assert data["input_metadata"] == {"my_key": "val"}
    assert "created_at" in data
    assert "updated_at" in data

def test_update_metadata(manager):
    session_id = manager.create_session()
    original_data = manager.get_session(session_id)
    original_updated = original_data["updated_at"]
    original_created = original_data["created_at"]
    
    manager.update_metadata(session_id, {"status": "processing", "error": "none"})
    
    new_data = manager.get_session(session_id)
    assert new_data["status"] == "processing"
    assert new_data["error"] == "none"
    assert new_data["created_at"] == original_created
    assert new_data["updated_at"] != original_updated

def test_invalid_session_id_format(manager):
    with pytest.raises(SessionManagerError):
        manager.get_session_workspace("not-a-uuid")
        
    with pytest.raises(SessionManagerError):
        manager.get_session_workspace("../../../escaped")

def test_cross_session_isolation(manager):
    session_a = manager.create_session({"owner": "Alice"})
    session_b = manager.create_session({"owner": "Bob"})
    
    # Test directories are different
    workspace_a = manager.get_session_workspace(session_a)
    workspace_b = manager.get_session_workspace(session_b)
    assert workspace_a != workspace_b
    assert str(workspace_a) not in str(workspace_b)
    
    # Test metadata does not bleed
    data_a = manager.get_session(session_a)
    data_b = manager.get_session(session_b)
    
    assert data_a["input_metadata"]["owner"] == "Alice"
    assert data_b["input_metadata"]["owner"] == "Bob"
