import pytest
import tempfile
import os
from pathlib import Path
from fastapi.testclient import TestClient

from src.backend.api import app, get_session_manager, get_input_manager
from src.backend.session_manager import BackendSessionManager
from src.backend.input_manager import BackendInputManager

@pytest.fixture
def temp_workspace():
    with tempfile.TemporaryDirectory() as d:
        yield d

@pytest.fixture
def client(temp_workspace):
    # Override the default dependency to use the temporary workspace
    def override_get_session_manager():
        return BackendSessionManager(base_workspace_dir=temp_workspace)
        
    app.dependency_overrides[get_session_manager] = override_get_session_manager
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()

# 1. create session
def test_create_session(client):
    response = client.post("/sessions", json={"metadata": {"test_key": "test_val"}})
    assert response.status_code == 201
    data = response.json()
    assert "session_id" in data
    assert data["status"] == "created"

# 2. get session
def test_get_session(client):
    resp1 = client.post("/sessions", json={})
    session_id = resp1.json()["session_id"]
    
    resp2 = client.get(f"/sessions/{session_id}")
    assert resp2.status_code == 200
    data = resp2.json()
    assert data["session_id"] == session_id
    assert "input_metadata" in data

# 3. upload valid input and verify internal contract and cleanup
def test_upload_valid_input_and_cleanup(client):
    resp1 = client.post("/sessions", json={})
    session_id = resp1.json()["session_id"]
    
    files = {"file": ("test_vid.mp4", b"dummy video content", "video/mp4")}
    data = {"input_type": "video"}
    
    # Check temp dir before
    temp_dir = Path(tempfile.gettempdir())
    initial_temp_files = set(temp_dir.iterdir())
    
    resp2 = client.post(f"/sessions/{session_id}/inputs", files=files, data=data)
    assert resp2.status_code == 201
    record = resp2.json()
    assert record["original_filename"] == "test_vid.mp4"
    assert record["extension"] == ".mp4"
    assert record["input_type"] == "video"
    
    # Verify temporary staging file is removed
    current_temp_files = set(temp_dir.iterdir())
    assert len(current_temp_files - initial_temp_files) == 0

# 4. reject invalid extension and verify cleanup
def test_reject_invalid_extension_and_cleanup(client):
    resp1 = client.post("/sessions", json={})
    session_id = resp1.json()["session_id"]
    
    temp_dir = Path(tempfile.gettempdir())
    initial_temp_files = set(temp_dir.iterdir())
    
    files = {"file": ("script.py", b"print('hello')", "text/x-python")}
    resp2 = client.post(f"/sessions/{session_id}/inputs", files=files)
    assert resp2.status_code == 400
    assert "Unsupported file extension" in resp2.json()["detail"]
    
    # Verify temp file cleaned up on InputManager failure
    current_temp_files = set(temp_dir.iterdir())
    assert len(current_temp_files - initial_temp_files) == 0

# 5. reject missing session
def test_reject_missing_session(client):
    files = {"file": ("test.mp4", b"content", "video/mp4")}
    resp = client.post("/sessions/123e4567-e89b-12d3-a456-426614174000/inputs", files=files)
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()
    
    resp_get = client.get("/sessions/123e4567-e89b-12d3-a456-426614174000")
    assert resp_get.status_code == 404

# 6. list inputs
def test_list_inputs(client):
    resp1 = client.post("/sessions", json={})
    session_id = resp1.json()["session_id"]
    
    client.post(f"/sessions/{session_id}/inputs", files={"file": ("f1.mp4", b"v1")})
    client.post(f"/sessions/{session_id}/inputs", files={"file": ("f2.mp4", b"v2")})
    
    resp_list = client.get(f"/sessions/{session_id}/inputs")
    assert resp_list.status_code == 200
    inputs = resp_list.json()
    assert len(inputs) == 2

# 7. retrieve/download input and verify exact bytes
def test_retrieve_input_bytes(client):
    resp1 = client.post("/sessions", json={})
    session_id = resp1.json()["session_id"]
    
    test_bytes = b"exact_binary_data_123"
    up_resp = client.post(f"/sessions/{session_id}/inputs", files={"file": ("f.mp4", test_bytes)})
    stored_name = up_resp.json()["stored_filename"]
    
    dl_resp = client.get(f"/sessions/{session_id}/inputs/{stored_name}")
    assert dl_resp.status_code == 200
    assert dl_resp.content == test_bytes

# 8. delete input
def test_delete_input(client):
    resp1 = client.post("/sessions", json={})
    session_id = resp1.json()["session_id"]
    
    up_resp = client.post(f"/sessions/{session_id}/inputs", files={"file": ("f.mp4", b"c")})
    stored_name = up_resp.json()["stored_filename"]
    
    del_resp = client.delete(f"/sessions/{session_id}/inputs/{stored_name}")
    assert del_resp.status_code == 200
    
    list_resp = client.get(f"/sessions/{session_id}/inputs")
    assert len(list_resp.json()) == 0

# 9. create job
def test_create_job(client):
    resp1 = client.post("/sessions", json={})
    session_id = resp1.json()["session_id"]
    
    resp_job = client.post(f"/sessions/{session_id}/jobs", json={"reconstruction_mode": "relative"})
    assert resp_job.status_code == 201
    job_data = resp_job.json()
    assert "job_id" in job_data
    assert job_data["status"] == "queued"

# 10. get job
def test_get_job(client):
    resp1 = client.post("/sessions", json={})
    session_id = resp1.json()["session_id"]
    job_id = client.post(f"/sessions/{session_id}/jobs", json={}).json()["job_id"]
    
    resp_get = client.get(f"/jobs/{job_id}")
    assert resp_get.status_code == 200
    assert resp_get.json()["job_id"] == job_id
    assert resp_get.json()["status"] == "queued"

# 11. valid job status transition
def test_valid_job_transition(client):
    resp1 = client.post("/sessions", json={})
    session_id = resp1.json()["session_id"]
    job_id = client.post(f"/sessions/{session_id}/jobs", json={}).json()["job_id"]
    
    resp_status = client.post(f"/jobs/{job_id}/status", json={"status": "processing"})
    assert resp_status.status_code == 200
    assert resp_status.json()["status"] == "processing"

# 12. invalid job status transition
def test_invalid_job_transition(client):
    resp1 = client.post("/sessions", json={})
    session_id = resp1.json()["session_id"]
    job_id = client.post(f"/sessions/{session_id}/jobs", json={}).json()["job_id"]
    
    resp_status = client.post(f"/jobs/{job_id}/status", json={"status": "completed"})
    assert resp_status.status_code == 409
    assert "invalid job state transition" in resp_status.json()["detail"].lower()

# 13. missing job
def test_missing_job(client):
    resp = client.get("/jobs/123e4567-e89b-12d3-a456-426614174000")
    assert resp.status_code == 404
    
    resp_status = client.post("/jobs/123e4567-e89b-12d3-a456-426614174000/status", json={"status": "processing"})
    assert resp_status.status_code == 404

# 14. session isolation between two sessions
def test_session_isolation(client):
    sess1 = client.post("/sessions", json={}).json()["session_id"]
    sess2 = client.post("/sessions", json={}).json()["session_id"]
    
    file_record = client.post(f"/sessions/{sess1}/inputs", files={"file": ("f.mp4", b"v")}).json()
    stored_name = file_record["stored_filename"]
    
    resp = client.get(f"/sessions/{sess2}/inputs/{stored_name}")
    assert resp.status_code == 404

# 15. upload filename/path traversal protection
def test_upload_path_traversal(client):
    sess1 = client.post("/sessions", json={}).json()["session_id"]
    files = {"file": ("../../../etc/passwd.mp4", b"v")}
    resp = client.post(f"/sessions/{sess1}/inputs", files=files)
    assert resp.status_code == 201
    stored = resp.json()["stored_filename"]
    assert "../" not in stored
    assert "/" not in stored

    resp2 = client.get(f"/sessions/{sess1}/inputs/../anotherfile.mp4")
    assert resp2.status_code in [400, 404]

# 16. oversized upload handling
def test_oversized_upload(client):
    sess1 = client.post("/sessions", json={}).json()["session_id"]
    
    def override_get_input_manager():
        sm = BackendSessionManager(base_workspace_dir=client.app.dependency_overrides.get(get_session_manager, get_session_manager)().base_dir)
        return BackendInputManager(sm, max_file_size_bytes=5)
        
    client.app.dependency_overrides[get_input_manager] = override_get_input_manager
    
    files = {"file": ("big.mp4", b"123456")} # 6 bytes
    resp = client.post(f"/sessions/{sess1}/inputs", files=files)
    assert resp.status_code == 413
    assert "large" in resp.json()["detail"].lower()
    
    client.app.dependency_overrides.pop(get_input_manager)

# 17. malformed request handling
def test_malformed_request(client):
    sess1 = client.post("/sessions", json={}).json()["session_id"]
    job_id = client.post(f"/sessions/{sess1}/jobs", json={}).json()["job_id"]
    
    resp = client.post(f"/jobs/{job_id}/status", json={"error": "missing status"})
    assert resp.status_code == 422 

# 18. API responses do not expose internal filesystem paths
def test_api_responses_hide_paths(client):
    sess1 = client.post("/sessions", json={}).json()["session_id"]
    file_record = client.post(f"/sessions/{sess1}/inputs", files={"file": ("f.mp4", b"v")}).json()
    
    for val in file_record.values():
        if isinstance(val, str):
            assert "\\" not in val
            if "T" not in val: # ignore datetime iso string which has punctuation, focus on path separators
                assert "/" not in val

    sess_data = client.get(f"/sessions/{sess1}").json()
    assert "\\" not in str(sess_data)
