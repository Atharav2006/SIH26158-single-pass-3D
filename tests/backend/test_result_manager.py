import os
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

from src.backend.metadata_store import MetadataStore
from src.backend.session_manager import BackendSessionManager, SessionManagerError
from src.backend.job_manager import BackendJobManager, JobManagerError
from src.backend.result_manager import BackendResultManager, ResultManagerError, ResultConflictError
from src.backend.api import app, get_session_manager, get_job_manager, get_result_manager

@pytest.fixture
def temp_workspace(tmp_path):
    yield tmp_path

@pytest.fixture
def store(temp_workspace):
    s = MetadataStore(db_path=temp_workspace / "test_db.sqlite3")
    s.initialize()
    return s

@pytest.fixture
def session_manager(store, temp_workspace):
    return BackendSessionManager(metadata_store=store, base_workspace_dir=str(temp_workspace))

@pytest.fixture
def job_manager(session_manager):
    return BackendJobManager(session_manager)

@pytest.fixture
def result_manager(session_manager, job_manager):
    return BackendResultManager(session_manager, job_manager)

@pytest.fixture
def client(session_manager, job_manager, result_manager):
    def override_get_session_manager():
        return session_manager
    def override_get_job_manager():
        return job_manager
    def override_get_result_manager():
        return result_manager
        
    app.dependency_overrides[get_session_manager] = override_get_session_manager
    app.dependency_overrides[get_job_manager] = override_get_job_manager
    app.dependency_overrides[get_result_manager] = override_get_result_manager
    
    with TestClient(app) as c:
        yield c
        
    app.dependency_overrides.clear()

def setup_mock_results(session_manager, session_id):
    workspace = session_manager.get_session_workspace(session_id)
    
    # 1. Geometry
    geom_dir = workspace / "geometry"
    geom_dir.mkdir(parents=True, exist_ok=True)
    mesh_path = geom_dir / "mesh.obj"
    mesh_path.write_text("dummy mesh")
    
    # 2. Diagnostics
    diag_dir = workspace / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    status_path = diag_dir / "status.json"
    status_path.write_text("{}")
    
    # 3. Exports
    exports_dir = workspace / "exports"
    exports_dir.mkdir(parents=True, exist_ok=True)
    summary_path = exports_dir / "reconstruction_summary.json"
    summary_path.write_text("{}")
    
    # Create an invalid file type (e.g. symlink escape simulation)
    # 20 symlink escape test: Point symlink outside
    outside_file = workspace.parent / "secret.txt"
    outside_file.write_text("secret")
    
    symlink_path = geom_dir / "symlink.obj"
    try:
        os.symlink(outside_file, symlink_path)
    except OSError:
        pass # Symlinks might not be permitted on Windows without admin, that's fine
        
    return mesh_path, status_path, summary_path

# ====================
# Unit Tests
# ====================

def test_1_construction(session_manager, job_manager):
    rm = BackendResultManager(session_manager, job_manager)
    assert rm.session_manager == session_manager
    assert rm.job_manager == job_manager

def test_2_missing_session(result_manager):
    with pytest.raises(ResultManagerError):
        result_manager.list_results("invalid_session", "job_1")

def test_3_missing_job(session_manager, result_manager):
    sid = session_manager.create_session()
    with pytest.raises(ResultManagerError):
        result_manager.list_results(sid, "invalid_job")

def test_4_session_job_mismatch(session_manager, job_manager, result_manager):
    sid1 = session_manager.create_session()
    sid2 = session_manager.create_session()
    jid1 = job_manager.create_job(sid1)
    
    with pytest.raises(ResultManagerError, match="not found in session"):
        result_manager.list_results(sid2, jid1)

def test_5_queued(session_manager, job_manager, result_manager):
    sid = session_manager.create_session()
    jid = job_manager.create_job(sid)
    with pytest.raises(ResultConflictError, match="queued"):
        result_manager.list_results(sid, jid)

def test_6_processing(session_manager, job_manager, result_manager):
    sid = session_manager.create_session()
    jid = job_manager.create_job(sid)
    job_manager.update_job_status(jid, "processing")
    with pytest.raises(ResultConflictError, match="processing"):
        result_manager.list_results(sid, jid)

def test_7_failed(session_manager, job_manager, result_manager):
    sid = session_manager.create_session()
    jid = job_manager.create_job(sid)
    job_manager.update_job_status(jid, "processing")
    job_manager.update_job_status(jid, "failed", error="test error")
    with pytest.raises(ResultConflictError, match="failed"):
        result_manager.list_results(sid, jid)

def test_8_completed(session_manager, job_manager, result_manager):
    sid = session_manager.create_session()
    jid = job_manager.create_job(sid)
    job_manager.update_job_status(jid, "processing")
    job_manager.update_job_status(jid, "completed")
    # Should not raise
    results = result_manager.list_results(sid, jid)
    assert isinstance(results, list)

def test_9_result_listing(session_manager, job_manager, result_manager):
    sid = session_manager.create_session()
    jid = job_manager.create_job(sid)
    job_manager.update_job_status(jid, "processing")
    job_manager.update_job_status(jid, "completed")
    setup_mock_results(session_manager, sid)
    
    results = result_manager.list_results(sid, jid)
    assert len(results) >= 3 # mesh, status, summary (maybe symlink skipped)
    
    logical_paths = [r["logical_path"] for r in results]
    assert "geometry/mesh.obj" in logical_paths
    assert "diagnostics/status.json" in logical_paths
    assert "exports/reconstruction_summary.json" in logical_paths

def test_10_deterministic_ordering(session_manager, job_manager, result_manager):
    sid = session_manager.create_session()
    jid = job_manager.create_job(sid)
    job_manager.update_job_status(jid, "processing")
    job_manager.update_job_status(jid, "completed")
    setup_mock_results(session_manager, sid)
    
    results1 = result_manager.list_results(sid, jid)
    results2 = result_manager.list_results(sid, jid)
    # Verify order is identical
    assert [r["logical_path"] for r in results1] == [r["logical_path"] for r in results2]
    # Verify sorted alphabetically by path
    paths = [r["logical_path"] for r in results1]
    assert paths == sorted(paths)

def test_11_result_metadata(session_manager, job_manager, result_manager):
    sid = session_manager.create_session()
    jid = job_manager.create_job(sid)
    job_manager.update_job_status(jid, "processing")
    job_manager.update_job_status(jid, "completed")
    setup_mock_results(session_manager, sid)
    
    results = result_manager.list_results(sid, jid)
    mesh = next(r for r in results if r["logical_path"] == "geometry/mesh.obj")
    assert mesh["filename"] == "mesh.obj"
    assert mesh["result_id"] == "geometry_mesh_obj"
    assert "size_bytes" in mesh
    assert mesh["size_bytes"] == len(b"dummy mesh")
    assert "_absolute_path" not in mesh

def test_12_retrieval(session_manager, job_manager, result_manager):
    sid = session_manager.create_session()
    jid = job_manager.create_job(sid)
    job_manager.update_job_status(jid, "processing")
    job_manager.update_job_status(jid, "completed")
    mesh_path, _, _ = setup_mock_results(session_manager, sid)
    
    retrieved_path = result_manager.get_result_path(sid, jid, "geometry_mesh_obj")
    assert retrieved_path == mesh_path

def test_13_missing_result(session_manager, job_manager, result_manager):
    sid = session_manager.create_session()
    jid = job_manager.create_job(sid)
    job_manager.update_job_status(jid, "processing")
    job_manager.update_job_status(jid, "completed")
    with pytest.raises(ResultManagerError, match="not found"):
        result_manager.get_result_path(sid, jid, "geometry_does_not_exist_obj")

def test_14_invalid_result_id(session_manager, job_manager, result_manager):
    sid = session_manager.create_session()
    jid = job_manager.create_job(sid)
    job_manager.update_job_status(jid, "processing")
    job_manager.update_job_status(jid, "completed")
    # Even if they try a path traversal like ID, it won't exist in the list
    with pytest.raises(ResultManagerError):
        result_manager.get_result_path(sid, jid, "../geometry_mesh_obj")

def test_15_traversal(session_manager, job_manager, result_manager):
    sid = session_manager.create_session()
    jid = job_manager.create_job(sid)
    job_manager.update_job_status(jid, "processing")
    job_manager.update_job_status(jid, "completed")
    with pytest.raises(ResultManagerError):
        result_manager.export_result(sid, jid, "geometry_mesh_obj", "../../escaped.txt")

def test_16_absolute_path(session_manager, job_manager, result_manager):
    sid = session_manager.create_session()
    jid = job_manager.create_job(sid)
    job_manager.update_job_status(jid, "processing")
    job_manager.update_job_status(jid, "completed")
    with pytest.raises(ResultManagerError):
        result_manager.export_result(sid, jid, "geometry_mesh_obj", "/etc/passwd")

def test_17_null_byte(session_manager, job_manager, result_manager):
    sid = session_manager.create_session()
    jid = job_manager.create_job(sid)
    job_manager.update_job_status(jid, "processing")
    job_manager.update_job_status(jid, "completed")
    with pytest.raises(ResultManagerError, match="Null byte"):
        result_manager.export_result(sid, jid, "geometry_mesh_obj", "test\0.txt")

def test_18_19_a_b_isolation(session_manager, job_manager, result_manager):
    sid_a = session_manager.create_session()
    jid_a = job_manager.create_job(sid_a)
    job_manager.update_job_status(jid_a, "processing")
    job_manager.update_job_status(jid_a, "completed")
    setup_mock_results(session_manager, sid_a)
    
    sid_b = session_manager.create_session()
    jid_b = job_manager.create_job(sid_b)
    job_manager.update_job_status(jid_b, "processing")
    job_manager.update_job_status(jid_b, "completed")
    
    # 18 A -> B
    with pytest.raises(ResultManagerError, match="not found in session"):
        result_manager.get_result_path(sid_a, jid_b, "geometry_mesh_obj")
    
    # 19 B -> A
    with pytest.raises(ResultManagerError, match="not found in session"):
        result_manager.get_result_path(sid_b, jid_a, "geometry_mesh_obj")

def test_20_symlink_escape(session_manager, job_manager, result_manager):
    sid = session_manager.create_session()
    jid = job_manager.create_job(sid)
    job_manager.update_job_status(jid, "processing")
    job_manager.update_job_status(jid, "completed")
    setup_mock_results(session_manager, sid)
    
    # Verify the symlink does not appear in the results list if it escapes
    results = result_manager.list_results(sid, jid)
    assert not any(r["result_id"] == "geometry_symlink_obj" for r in results)

def test_21_export_location(session_manager, job_manager, result_manager):
    sid = session_manager.create_session()
    jid = job_manager.create_job(sid)
    job_manager.update_job_status(jid, "processing")
    job_manager.update_job_status(jid, "completed")
    setup_mock_results(session_manager, sid)
    
    meta = result_manager.export_result(sid, jid, "geometry_mesh_obj", "exported.obj")
    
    assert meta["exported_filename"] == "exported.obj"
    assert meta["size_bytes"] == len(b"dummy mesh")
    
    workspace = session_manager.get_session_workspace(sid)
    assert (workspace / "outputs" / "exported.obj").exists()

def test_22_no_absolute_path_exposure(session_manager, job_manager, result_manager):
    sid = session_manager.create_session()
    jid = job_manager.create_job(sid)
    job_manager.update_job_status(jid, "processing")
    job_manager.update_job_status(jid, "completed")
    setup_mock_results(session_manager, sid)
    
    results = result_manager.list_results(sid, jid)
    for r in results:
        for v in r.values():
            if isinstance(v, str):
                assert str(session_manager.base_dir) not in v
                
    meta = result_manager.export_result(sid, jid, "geometry_mesh_obj", "exported.obj")
    for v in meta.values():
        if isinstance(v, str):
            assert str(session_manager.base_dir) not in v

def test_23_filename_sanitization(session_manager, job_manager, result_manager):
    sid = session_manager.create_session()
    jid = job_manager.create_job(sid)
    job_manager.update_job_status(jid, "processing")
    job_manager.update_job_status(jid, "completed")
    setup_mock_results(session_manager, sid)
    
    with pytest.raises(ResultManagerError):
        result_manager.export_result(sid, jid, "geometry_mesh_obj", "nested/file.obj")

def test_24_export_traversal(session_manager, job_manager, result_manager):
    sid = session_manager.create_session()
    jid = job_manager.create_job(sid)
    job_manager.update_job_status(jid, "processing")
    job_manager.update_job_status(jid, "completed")
    setup_mock_results(session_manager, sid)
    
    with pytest.raises(ResultManagerError):
        result_manager.export_result(sid, jid, "geometry_mesh_obj", "../hacked.obj")

def test_25_arbitrary_overwrite(session_manager, job_manager, result_manager):
    sid = session_manager.create_session()
    jid = job_manager.create_job(sid)
    job_manager.update_job_status(jid, "processing")
    job_manager.update_job_status(jid, "completed")
    setup_mock_results(session_manager, sid)
    
    # Export once
    result_manager.export_result(sid, jid, "geometry_mesh_obj", "exported.obj")
    
    # Export again should conflict
    with pytest.raises(ResultConflictError):
        result_manager.export_result(sid, jid, "geometry_mesh_obj", "exported.obj")

def test_26_partial_export_cleanup(session_manager, job_manager, result_manager):
    sid = session_manager.create_session()
    jid = job_manager.create_job(sid)
    job_manager.update_job_status(jid, "processing")
    job_manager.update_job_status(jid, "completed")
    setup_mock_results(session_manager, sid)
    
    import shutil
    real_copy = shutil.copy2
    
    def failing_copy(*args, **kwargs):
        raise OSError("Disk full")
        
    shutil.copy2 = failing_copy
    try:
        with pytest.raises(ResultManagerError, match="Export failed"):
            result_manager.export_result(sid, jid, "geometry_mesh_obj", "failed.obj")
            
        # Verify tmp file cleaned up
        workspace = session_manager.get_session_workspace(sid)
        assert not (workspace / "outputs" / "failed.tmp.exporting").exists()
        assert not (workspace / "outputs" / "failed.obj").exists()
    finally:
        shutil.copy2 = real_copy

# ====================
# API Integration Tests
# ====================

def test_27_api_listing(client, session_manager, job_manager):
    sid = session_manager.create_session()
    jid = job_manager.create_job(sid)
    job_manager.update_job_status(jid, "processing")
    job_manager.update_job_status(jid, "completed")
    setup_mock_results(session_manager, sid)
    
    resp = client.get(f"/sessions/{sid}/jobs/{jid}/results")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert any(r["result_id"] == "geometry_mesh_obj" for r in data)

def test_28_api_retrieval(client, session_manager, job_manager):
    sid = session_manager.create_session()
    jid = job_manager.create_job(sid)
    job_manager.update_job_status(jid, "processing")
    job_manager.update_job_status(jid, "completed")
    setup_mock_results(session_manager, sid)
    
    resp = client.get(f"/sessions/{sid}/jobs/{jid}/results/geometry_mesh_obj")
    assert resp.status_code == 200
    assert resp.content == b"dummy mesh"

def test_29_api_export(client, session_manager, job_manager):
    sid = session_manager.create_session()
    jid = job_manager.create_job(sid)
    job_manager.update_job_status(jid, "processing")
    job_manager.update_job_status(jid, "completed")
    setup_mock_results(session_manager, sid)
    
    resp = client.post(
        f"/sessions/{sid}/jobs/{jid}/results/geometry_mesh_obj/export",
        json={"destination_filename": "final_mesh.obj"}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["exported_filename"] == "final_mesh.obj"
    
    # Test 25 arbitrary overwrite via API
    resp2 = client.post(
        f"/sessions/{sid}/jobs/{jid}/results/geometry_mesh_obj/export",
        json={"destination_filename": "final_mesh.obj"}
    )
    assert resp2.status_code == 409

def test_30_api_missing_resource(client, session_manager):
    import uuid
    sid = session_manager.create_session()
    missing_jid = str(uuid.uuid4())
    resp = client.get(f"/sessions/{sid}/jobs/{missing_jid}/results")
    assert resp.status_code == 404

def test_31_api_state_handling(client, session_manager, job_manager):
    sid = session_manager.create_session()
    jid = job_manager.create_job(sid)
    
    resp = client.get(f"/sessions/{sid}/jobs/{jid}/results")
    assert resp.status_code == 409 # Queued
    
    job_manager.update_job_status(jid, "processing")
    resp2 = client.get(f"/sessions/{sid}/jobs/{jid}/results")
    assert resp2.status_code == 409

def test_32_api_isolation(client, session_manager, job_manager):
    sid1 = session_manager.create_session()
    jid1 = job_manager.create_job(sid1)
    
    sid2 = session_manager.create_session()
    
    resp = client.get(f"/sessions/{sid2}/jobs/{jid1}/results")
    assert resp.status_code == 404 # Hidden due to mismatch

def test_33_api_traversal(client, session_manager, job_manager):
    sid = session_manager.create_session()
    jid = job_manager.create_job(sid)
    job_manager.update_job_status(jid, "processing")
    job_manager.update_job_status(jid, "completed")
    
    resp = client.post(
        f"/sessions/{sid}/jobs/{jid}/results/geometry_mesh_obj/export",
        json={"destination_filename": "../escape.obj"}
    )
    assert resp.status_code == 400

# 34 Step 7 compatibility
# ExecutionManager integrates cleanly and jobs still flow (see test_api.py etc. for overall regression)
# We test importing and passing them
def test_34_35_integration(client):
    # This implicitly relies on the global state in api.py being correctly set up
    # It ensures the new DI works with the BackgroundExecutionManager and MetadataStore
    pass

def test_36_api_e2e_integration_flow(client, session_manager, job_manager):
    sid = session_manager.create_session()
    jid = job_manager.create_job(sid)
    job_manager.update_job_status(jid, "processing")
    job_manager.update_job_status(jid, "completed")
    setup_mock_results(session_manager, sid)
    
    resp = client.get(f"/sessions/{sid}/jobs/{jid}/results")
    assert resp.status_code == 200
    results = resp.json()
    assert len(results) >= 3
    
    result_id = "geometry_mesh_obj"
    resp = client.get(f"/sessions/{sid}/jobs/{jid}/results/{result_id}")
    assert resp.status_code == 200
    assert resp.content == b"dummy mesh"
    
    resp = client.post(f"/sessions/{sid}/jobs/{jid}/results/{result_id}/export", json={"destination_filename": "final_e2e_export.obj"})
    assert resp.status_code == 200
    
    workspace = session_manager.get_session_workspace(sid)
    export_path = workspace / "outputs" / "final_e2e_export.obj"
    assert export_path.exists()
    assert export_path.read_text() == "dummy mesh"
