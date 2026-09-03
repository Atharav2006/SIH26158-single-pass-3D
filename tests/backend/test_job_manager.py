import pytest
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from src.backend.session_manager import BackendSessionManager
from src.backend.job_manager import BackendJobManager, JobManagerError

@pytest.fixture
def temp_workspace():
    with tempfile.TemporaryDirectory() as d:
        yield d

@pytest.fixture
def session_manager(temp_workspace):
    return BackendSessionManager(base_workspace_dir=temp_workspace)

@pytest.fixture
def job_manager(session_manager):
    return BackendJobManager(session_manager)

def test_create_job_valid_session(session_manager, job_manager):
    session_id = session_manager.create_session()
    job_id = job_manager.create_job(session_id)
    assert str(uuid.UUID(job_id)) == job_id
    
def test_job_starts_as_queued(session_manager, job_manager):
    session_id = session_manager.create_session()
    job_id = job_manager.create_job(session_id)
    job = job_manager.get_job(job_id)
    assert job["status"] == "queued"

def test_unique_job_ids(session_manager, job_manager):
    session_id = session_manager.create_session()
    job_id1 = job_manager.create_job(session_id)
    job_id2 = job_manager.create_job(session_id)
    assert job_id1 != job_id2

def test_persisted_job_metadata(session_manager, job_manager):
    session_id = session_manager.create_session()
    job_id = job_manager.create_job(session_id, reconstruction_mode="METRIC")
    
    workspace = session_manager.get_session_workspace(session_id)
    job_file = workspace / "metadata" / "jobs" / f"{job_id}.json"
    assert job_file.exists()

def test_get_job_reads_persisted_data(session_manager, job_manager):
    session_id = session_manager.create_session()
    job_id = job_manager.create_job(session_id)
    job = job_manager.get_job(job_id)
    assert job["job_id"] == job_id
    assert job["session_id"] == session_id

def test_queued_to_processing(session_manager, job_manager):
    session_id = session_manager.create_session()
    job_id = job_manager.create_job(session_id)
    
    job_manager.update_job_status(job_id, "processing")
    job = job_manager.get_job(job_id)
    assert job["status"] == "processing"
    assert job["started_at"] is not None

def test_processing_to_completed(session_manager, job_manager):
    session_id = session_manager.create_session()
    job_id = job_manager.create_job(session_id)
    job_manager.update_job_status(job_id, "processing")
    
    job_manager.update_job_status(job_id, "completed", result_metadata={"success": True})
    job = job_manager.get_job(job_id)
    assert job["status"] == "completed"
    assert job["completed_at"] is not None
    assert job["result_metadata"] == {"success": True}

def test_processing_to_failed(session_manager, job_manager):
    session_id = session_manager.create_session()
    job_id = job_manager.create_job(session_id)
    job_manager.update_job_status(job_id, "processing")
    
    job_manager.update_job_status(job_id, "failed", error="out of memory")
    job = job_manager.get_job(job_id)
    assert job["status"] == "failed"
    assert job["error"] == "out of memory"

def test_invalid_status_rejected(session_manager, job_manager):
    session_id = session_manager.create_session()
    job_id = job_manager.create_job(session_id)
    
    with pytest.raises(JobManagerError):
        job_manager.update_job_status(job_id, "magical")

def test_invalid_lifecycle_transition(session_manager, job_manager):
    session_id = session_manager.create_session()
    job_id = job_manager.create_job(session_id)
    
    # Cannot go directly from queued to completed
    with pytest.raises(JobManagerError):
        job_manager.update_job_status(job_id, "completed")

    job_manager.update_job_status(job_id, "processing")
    job_manager.update_job_status(job_id, "completed")
    
    # Cannot fail a completed job
    with pytest.raises(JobManagerError):
        job_manager.update_job_status(job_id, "failed", error="late failure")

def test_missing_job_rejected(job_manager):
    with pytest.raises(JobManagerError):
        job_manager.get_job(str(uuid.uuid4()))

def test_invalid_job_id_rejected(job_manager):
    with pytest.raises(JobManagerError):
        job_manager.get_job("invalid-uuid")

def test_traversal_attempt_rejected(job_manager):
    with pytest.raises(JobManagerError):
        job_manager.get_job("../../../etc/passwd")

def test_session_job_isolation(session_manager, job_manager):
    session_a = session_manager.create_session()
    session_b = session_manager.create_session()
    
    job_a = job_manager.create_job(session_a)
    job_b = job_manager.create_job(session_b)
    
    # Prove job_a is only in session_a workspace
    workspace_a = session_manager.get_session_workspace(session_a)
    workspace_b = session_manager.get_session_workspace(session_b)
    
    assert (workspace_a / "metadata" / "jobs" / f"{job_a}.json").exists()
    assert not (workspace_b / "metadata" / "jobs" / f"{job_a}.json").exists()
    
def test_session_metadata_reflects_lifecycle(session_manager, job_manager):
    session_id = session_manager.create_session()
    job_id = job_manager.create_job(session_id)
    
    session = session_manager.get_session(session_id)
    assert session["status"] == "queued"
    assert session["active_job_id"] == job_id
    
    job_manager.update_job_status(job_id, "processing")
    session = session_manager.get_session(session_id)
    assert session["status"] == "processing"

def test_timestamps_behave_correctly(session_manager, job_manager):
    session_id = session_manager.create_session()
    job_id = job_manager.create_job(session_id)
    
    job = job_manager.get_job(job_id)
    created_at = job["created_at"]
    updated_at = job["updated_at"]
    assert created_at == updated_at
    
    # Update to processing
    job_manager.update_job_status(job_id, "processing")
    job = job_manager.get_job(job_id)
    assert job["created_at"] == created_at
    assert job["updated_at"] != updated_at
    assert job["started_at"] is not None
    assert job["completed_at"] is None
