import pytest
import tempfile
import os
from pathlib import Path
from fastapi.testclient import TestClient

from src.backend.api import app, get_session_manager, get_input_manager
from src.backend.session_manager import BackendSessionManager
from src.backend.input_manager import BackendInputManager
from src.backend.metadata_store import MetadataStore

@pytest.fixture
def temp_workspace():
    with tempfile.TemporaryDirectory() as d:
        yield d

@pytest.fixture
def store(temp_workspace):
    store = MetadataStore(db_path=Path(temp_workspace) / "test.sqlite3")
    store.initialize()
    return store

@pytest.fixture
def client(temp_workspace, store):
    # Override the default dependency to use the temporary workspace
    def override_get_session_manager():
        return BackendSessionManager(base_workspace_dir=temp_workspace, metadata_store=store)
        
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
    
    import tempfile
    import os
    from unittest.mock import patch

    real_mkstemp = tempfile.mkstemp
    created_paths = []
    def mkstemp_side_effect(*args, **kwargs):
        fd, path = real_mkstemp(*args, **kwargs)
        created_paths.append(path)
        return fd, path

    with patch('src.backend.api.tempfile.mkstemp', side_effect=mkstemp_side_effect):
        resp2 = client.post(f"/sessions/{session_id}/inputs", files=files, data=data)
        
    assert resp2.status_code == 201
    record = resp2.json()
    assert record["original_filename"] == "test_vid.mp4"
    assert record["extension"] == ".mp4"
    assert record["input_type"] == "video"

    # Verify temporary staging file is removed
    assert len(created_paths) > 0, "Expected at least one temp file to be created during upload"
    for path in created_paths:
        assert not os.path.exists(path), f"Temporary file was not cleaned up: {path}"

# 4. reject invalid extension and verify cleanup
def test_reject_invalid_extension_and_cleanup(client):
    resp1 = client.post("/sessions", json={})
    session_id = resp1.json()["session_id"]
    
    files = {"file": ("script.py", b"print('hello')", "text/x-python")}
    
    import tempfile
    import os
    from unittest.mock import patch

    real_mkstemp = tempfile.mkstemp
    created_paths = []
    def mkstemp_side_effect(*args, **kwargs):
        fd, path = real_mkstemp(*args, **kwargs)
        created_paths.append(path)
        return fd, path

    with patch('src.backend.api.tempfile.mkstemp', side_effect=mkstemp_side_effect):
        resp2 = client.post(f"/sessions/{session_id}/inputs", files=files)
        
    assert resp2.status_code == 400
    assert "Unsupported file extension" in resp2.json()["detail"]
    
    # Verify temp file cleaned up on InputManager failure
    assert len(created_paths) > 0, "Expected at least one temp file to be created during upload"
    for path in created_paths:
        assert not os.path.exists(path), f"Temporary file was not cleaned up: {path}"

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

# 11. missing job
def test_missing_job(client):
    resp = client.get("/jobs/123e4567-e89b-12d3-a456-426614174000")
    assert resp.status_code == 404

# 12. removed status endpoint returns 404/405
def test_removed_status_endpoint(client):
    resp = client.post("/jobs/123e4567-e89b-12d3-a456-426614174000/status", json={"status": "processing"})
    assert resp.status_code in (404, 405)

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
        # Get the same session manager being used by the rest of the test
        sm = client.app.dependency_overrides.get(get_session_manager, get_session_manager)()
        return BackendInputManager(sm, max_file_size_bytes=5)
        
    client.app.dependency_overrides[get_input_manager] = override_get_input_manager
    
    files = {"file": ("big.mp4", b"123456")} # 6 bytes
    resp = client.post(f"/sessions/{sess1}/inputs", files=files)
    assert resp.status_code == 413
    assert "large" in resp.json()["detail"].lower()
    
    client.app.dependency_overrides.pop(get_input_manager)

# 17. session deletion removes workspace and DB records
def test_session_cleanup(client):
    sess1 = client.post("/sessions", json={}).json()["session_id"]
    client.post(f"/sessions/{sess1}/inputs", files={"file": ("f.mp4", b"v")})
    
    # Verify it exists
    assert client.get(f"/sessions/{sess1}").status_code == 200
    
    # Delete it
    del_resp = client.delete(f"/sessions/{sess1}")
    assert del_resp.status_code == 204
    
    # Verify DB record gone
    assert client.get(f"/sessions/{sess1}").status_code == 404
    
    # Wait, workspace should be gone. We can't check directly without internal access, but we can verify inputs are 404
    assert client.get(f"/sessions/{sess1}/inputs").status_code == 404

# 18. missing-session deletion
def test_missing_session_deletion(client):
    del_resp = client.delete("/sessions/123e4567-e89b-12d3-a456-426614174000")
    assert del_resp.status_code == 404

# 19. job-list API and isolation
def test_api_list_jobs(client):
    sess1 = client.post("/sessions", json={}).json()["session_id"]
    sess2 = client.post("/sessions", json={}).json()["session_id"]
    
    job1 = client.post(f"/sessions/{sess1}/jobs", json={}).json()["job_id"]
    job2 = client.post(f"/sessions/{sess2}/jobs", json={}).json()["job_id"]
    
    resp1 = client.get(f"/sessions/{sess1}/jobs")
    assert resp1.status_code == 200
    jobs1 = resp1.json()
    assert len(jobs1) == 1
    assert jobs1[0]["job_id"] == job1
    
    resp2 = client.get(f"/sessions/{sess2}/jobs")
    assert resp2.status_code == 200
    jobs2 = resp2.json()
    assert len(jobs2) == 1
    assert jobs2[0]["job_id"] == job2

# 20. input locking
def test_input_locked_during_processing(client):
    sess1 = client.post("/sessions", json={}).json()["session_id"]
    client.post(f"/sessions/{sess1}/inputs", files={"file": ("f.mp4", b"v")})
    
    job1 = client.post(f"/sessions/{sess1}/jobs", json={}).json()["job_id"]
    
    # Job is now 'queued'. Input mutation should be locked.
    upload_resp = client.post(f"/sessions/{sess1}/inputs", files={"file": ("f2.mp4", b"v2")})
    assert upload_resp.status_code == 409
    
    # Try delete
    inputs = client.get(f"/sessions/{sess1}/inputs").json()
    del_resp = client.delete(f"/sessions/{sess1}/inputs/{inputs[0]['stored_filename']}")
    assert del_resp.status_code == 409

# 21. failed job input behavior
def test_failed_job_input_behavior(client):
    sess1 = client.post("/sessions", json={}).json()["session_id"]
    job1 = client.post(f"/sessions/{sess1}/jobs", json={}).json()["job_id"]
    
    # Use store to artificially transition to 'failed' to simulate worker failure
    store = client.app.dependency_overrides.get(get_session_manager, get_session_manager)().store
    store.update_job(job1, {"status": "failed", "error": "test error"})
    
    # Inputs should be unlocked now
    upload_resp = client.post(f"/sessions/{sess1}/inputs", files={"file": ("f.mp4", b"v")})
    assert upload_resp.status_code == 201

# 22. zombie job recovery
def test_zombie_job_cleanup(client):
    sess1 = client.post("/sessions", json={}).json()["session_id"]
    job1 = client.post(f"/sessions/{sess1}/jobs", json={}).json()["job_id"]
    
    # Direct DB mutation to bypass manager checks and simulate stuck processing job
    from src.backend.api import get_session_manager, get_job_manager, get_input_manager, get_execution_manager
    session_manager = client.app.dependency_overrides.get(get_session_manager, get_session_manager)()
    store = session_manager.store
    store.update_job(job1, {"status": "processing"})
    
    # Execute the lifespan / recovery routine for the test execution manager
    job_manager = get_job_manager(session_manager)
    input_manager = get_input_manager(session_manager)
    exec_mgr = get_execution_manager(job_manager, input_manager, session_manager)
    exec_mgr.reap_stuck_jobs()
    
    # Verify job is failed
    resp = client.get(f"/jobs/{job1}")
    assert resp.json()["status"] == "failed"
    assert "backend restart" in resp.json()["error"].lower()

# 23. E2E real API pipeline test
def test_real_api_e2e_pipeline(client):
    from unittest.mock import patch
    import time
    
    def mock_reconstruct(args):
        import shutil
        from pathlib import Path
        workspace = Path(args.video).parent.parent
        exports_dir = workspace / "exports"
        exports_dir.mkdir(exist_ok=True, parents=True)
        (exports_dir / "test_artifact.bin").write_bytes(b"artifact")
        return {"status": "SUCCESS"}

    with patch('src.backend.reconstruction_worker.reconstruct_video', side_effect=mock_reconstruct):
        # 1. Create Session
        sess1 = client.post("/sessions", json={}).json()["session_id"]
        
        # 2. Upload Input
        client.post(f"/sessions/{sess1}/inputs", files={"file": ("test.mp4", b"mock_video")})
        
        # 3. Create Job
        job1 = client.post(f"/sessions/{sess1}/jobs", json={"reconstruction_mode": "relative"}).json()["job_id"]
        
        # 4. Submit Job
        submit_resp = client.post(f"/sessions/{sess1}/jobs/{job1}/submit")
        assert submit_resp.status_code == 202
        
        # 5. Poll Status
        status = "queued"
        for _ in range(20):
            job_info = client.get(f"/jobs/{job1}").json()
            status = job_info["status"]
            if status in ("completed", "failed"):
                break
            time.sleep(0.1)
            
        assert status == "completed", f"Job failed: {job_info.get('error')}"
        
        # 6. Retrieve Results
        results_resp = client.get(f"/sessions/{sess1}/jobs/{job1}/results")
        assert results_resp.status_code == 200
        results = results_resp.json()
        assert len(results) > 0
        
        # 7. Download specific result
        result_id = results[0]["result_id"]
        dl_resp = client.get(f"/sessions/{sess1}/jobs/{job1}/results/{result_id}")
        assert dl_resp.status_code == 200
        assert dl_resp.content == b"artifact"

# 24. session deletion with queued job returns 409
def test_delete_session_with_queued_job_returns_409(client):
    sess = client.post("/sessions", json={}).json()["session_id"]
    client.post(f"/sessions/{sess}/jobs", json={})  # creates a queued job
    
    del_resp = client.delete(f"/sessions/{sess}")
    assert del_resp.status_code == 409
    assert "active job" in del_resp.json()["detail"].lower() or "cannot delete" in del_resp.json()["detail"].lower()

# 25. session deletion with processing job returns 409
def test_delete_session_with_processing_job_returns_409(client):
    sess = client.post("/sessions", json={}).json()["session_id"]
    job_id = client.post(f"/sessions/{sess}/jobs", json={}).json()["job_id"]
    
    # Transition to processing via direct store mutation (simulating worker)
    store = client.app.dependency_overrides.get(get_session_manager, get_session_manager)().store
    store.update_job(job_id, {"status": "processing"})
    
    del_resp = client.delete(f"/sessions/{sess}")
    assert del_resp.status_code == 409

# 26. nonexistent session deletion returns 404
def test_delete_nonexistent_session_returns_404(client):
    del_resp = client.delete("/sessions/123e4567-e89b-12d3-a456-426614174000")
    assert del_resp.status_code == 404

# 27. session deletion after all jobs completed/failed succeeds
def test_delete_session_after_completed_job_succeeds(client):
    sess = client.post("/sessions", json={}).json()["session_id"]
    job_id = client.post(f"/sessions/{sess}/jobs", json={}).json()["job_id"]
    
    # Transition to completed via direct store mutation
    store = client.app.dependency_overrides.get(get_session_manager, get_session_manager)().store
    store.update_job(job_id, {"status": "processing"})
    store.update_job(job_id, {"status": "completed"})
    
    del_resp = client.delete(f"/sessions/{sess}")
    assert del_resp.status_code == 204
    
    # Confirm session is gone
    get_resp = client.get(f"/sessions/{sess}")
    assert get_resp.status_code == 404

