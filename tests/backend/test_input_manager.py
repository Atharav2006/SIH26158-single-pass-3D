import pytest
import tempfile
import os
from pathlib import Path
from src.backend.session_manager import BackendSessionManager
from src.backend.input_manager import BackendInputManager, InputManagerError

@pytest.fixture
def temp_workspace():
    with tempfile.TemporaryDirectory() as d:
        yield d

@pytest.fixture
def session_manager(temp_workspace):
    return BackendSessionManager(base_workspace_dir=temp_workspace)

@pytest.fixture
def input_manager(session_manager):
    return BackendInputManager(session_manager, max_file_size_bytes=1024 * 1024)  # 1MB limit for tests

@pytest.fixture
def sample_video(temp_workspace):
    path = Path(temp_workspace) / "source_video.mp4"
    path.write_bytes(b"mock video data")
    return path

@pytest.fixture
def oversized_video(temp_workspace):
    path = Path(temp_workspace) / "huge_video.mp4"
    # Create a file slightly larger than 1MB
    with open(path, "wb") as f:
        f.seek(1024 * 1024 + 10)
        f.write(b"0")
    return path

def test_save_valid_video(session_manager, input_manager, sample_video):
    session_id = session_manager.create_session()
    record = input_manager.save_input(session_id, sample_video, "my_flight.mp4", "video")
    assert "stored_filename" in record
    assert record["original_filename"] == "my_flight.mp4"
    assert record["input_type"] == "video"
    assert record["extension"] == ".mp4"

def test_saved_file_exists(session_manager, input_manager, sample_video):
    session_id = session_manager.create_session()
    record = input_manager.save_input(session_id, sample_video, "my_flight.mp4")
    path = input_manager.get_input_path(session_id, record["stored_filename"])
    assert path.exists()
    assert path.is_file()
    assert path.read_bytes() == b"mock video data"

def test_original_source_file_remains_unchanged(session_manager, input_manager, sample_video):
    session_id = session_manager.create_session()
    input_manager.save_input(session_id, sample_video, "test.mp4")
    assert sample_video.exists()
    assert sample_video.read_bytes() == b"mock video data"

def test_stored_filename_is_safe(session_manager, input_manager, sample_video):
    session_id = session_manager.create_session()
    record = input_manager.save_input(session_id, sample_video, "../../../etc/passwd.mp4")
    stored = record["stored_filename"]
    assert "../" not in stored
    assert "/" not in stored
    assert "\\" not in stored
    assert stored.endswith("_passwd.mp4")

def test_original_filename_is_preserved_in_metadata(session_manager, input_manager, sample_video):
    session_id = session_manager.create_session()
    weird_name = "some/weird\\path\0with_null.mp4"
    record = input_manager.save_input(session_id, sample_video, weird_name)
    assert record["original_filename"] == weird_name

def test_file_metadata_is_persisted(session_manager, input_manager, sample_video):
    session_id = session_manager.create_session()
    input_manager.save_input(session_id, sample_video, "test.mp4")
    # Verify by fetching session metadata directly
    session_data = session_manager.get_session(session_id)
    files = session_data["input_metadata"].get("files", [])
    assert len(files) == 1
    assert files[0]["original_filename"] == "test.mp4"

def test_list_inputs(session_manager, input_manager, sample_video):
    session_id = session_manager.create_session()
    input_manager.save_input(session_id, sample_video, "test1.mp4")
    input_manager.save_input(session_id, sample_video, "test2.mp4")
    inputs = input_manager.list_inputs(session_id)
    assert len(inputs) == 2
    assert inputs[0]["original_filename"] == "test1.mp4"

def test_get_input_path(session_manager, input_manager, sample_video):
    session_id = session_manager.create_session()
    record = input_manager.save_input(session_id, sample_video, "test.mp4")
    stored_filename = record["stored_filename"]
    path = input_manager.get_input_path(session_id, stored_filename)
    assert path.name == stored_filename

def test_delete_input(session_manager, input_manager, sample_video):
    session_id = session_manager.create_session()
    record = input_manager.save_input(session_id, sample_video, "test.mp4")
    stored = record["stored_filename"]
    
    assert len(input_manager.list_inputs(session_id)) == 1
    input_manager.delete_input(session_id, stored)
    
    assert len(input_manager.list_inputs(session_id)) == 0
    with pytest.raises(InputManagerError):
        input_manager.get_input_path(session_id, stored)

def test_unsupported_extension_rejected(session_manager, input_manager, sample_video):
    session_id = session_manager.create_session()
    with pytest.raises(InputManagerError) as exc:
        input_manager.save_input(session_id, sample_video, "script.py")
    assert "Unsupported file extension" in str(exc.value)

def test_oversized_file_rejected(session_manager, input_manager, oversized_video):
    session_id = session_manager.create_session()
    with pytest.raises(InputManagerError) as exc:
        input_manager.save_input(session_id, oversized_video, "huge.mp4")
    assert "exceeds maximum allowed" in str(exc.value)

def test_path_traversal_rejected(session_manager, input_manager):
    session_id = session_manager.create_session()
    with pytest.raises(InputManagerError):
        input_manager.get_input_path(session_id, "../temp/some_file.mp4")

def test_absolute_path_rejected(session_manager, input_manager):
    session_id = session_manager.create_session()
    with pytest.raises(InputManagerError):
        input_manager.get_input_path(session_id, "/etc/passwd")

def test_cross_session_access_rejected(session_manager, input_manager, sample_video):
    session_a = session_manager.create_session()
    session_b = session_manager.create_session()
    
    record = input_manager.save_input(session_a, sample_video, "test.mp4")
    stored = record["stored_filename"]
    
    # Try to access session_a's file through session_b
    with pytest.raises(InputManagerError):
        input_manager.get_input_path(session_b, stored)

def test_invalid_session_rejected(input_manager, sample_video):
    with pytest.raises(InputManagerError):
        input_manager.save_input("invalid-session", sample_video, "test.mp4")

def test_session_metadata_preserves_unrelated_fields(session_manager, input_manager, sample_video):
    session_id = session_manager.create_session({"custom_field": "val"})
    input_manager.save_input(session_id, sample_video, "test.mp4")
    
    session_data = session_manager.get_session(session_id)
    assert session_data["input_metadata"]["custom_field"] == "val"
    assert len(session_data["input_metadata"]["files"]) == 1

def test_failed_copy_does_not_create_false_record(session_manager, input_manager, temp_workspace):
    session_id = session_manager.create_session()
    non_existent = Path(temp_workspace) / "missing.mp4"
    
    with pytest.raises(InputManagerError):
        input_manager.save_input(session_id, non_existent, "test.mp4")
        
    assert len(input_manager.list_inputs(session_id)) == 0
