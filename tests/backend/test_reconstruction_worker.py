import pytest
import tempfile
import json
import uuid
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.backend.session_manager import BackendSessionManager, SessionManagerError
from src.backend.input_manager import BackendInputManager
from src.backend.job_manager import BackendJobManager, JobManagerError
from src.backend.reconstruction_worker import BackendReconstructionWorker
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
def managers(temp_workspace, store):
    sm = BackendSessionManager(base_workspace_dir=temp_workspace, metadata_store=store)
    im = BackendInputManager(sm)
    jm = BackendJobManager(sm)
    worker = BackendReconstructionWorker(sm, im, jm)
    return sm, im, jm, worker

def create_mock_video(sm, im, session_id):
    ws = sm.get_session_workspace(session_id)
    vid_path = ws / "dummy.mp4"
    vid_path.write_text("fake video")
    return im.save_input(session_id, str(vid_path), "dummy.mp4", "video")

# 1) success lifecycle
@patch('src.backend.reconstruction_worker.reconstruct_video')
def test_1_success_lifecycle(mock_reconstruct, managers):
    sm, im, jm, worker = managers
    session_id = sm.create_session({})
    create_mock_video(sm, im, session_id)
    job_id = jm.create_job(session_id)
    
    mock_reconstruct.return_value = {"status": "SUCCESS"}
    job_data = worker.run_job(job_id)
    
    assert job_data["status"] == "completed"

# 2) failure lifecycle (missing video input producing error)
@patch('src.backend.reconstruction_worker.reconstruct_video')
def test_2_failure_lifecycle(mock_reconstruct, managers):
    sm, im, jm, worker = managers
    session_id = sm.create_session({})
    job_id = jm.create_job(session_id)
    
    # Run without saving a video input
    job_data = worker.run_job(job_id)
    assert job_data["status"] == "failed"
    assert "No video input found" in job_data["error"]
    mock_reconstruct.assert_not_called()

# 3) missing job
def test_3_missing_job(managers):
    sm, im, jm, worker = managers
    with pytest.raises(ValueError, match="Cannot run job"):
        worker.run_job("missing-id")

# 4) invalid job/session relationship
@patch('src.backend.reconstruction_worker.reconstruct_video')
def test_4_invalid_job_session_relationship(mock_reconstruct, managers):
    sm, im, jm, worker = managers
    # We will simulate a job with a session ID that doesn't exist
    session_id = sm.create_session({})
    job_id = jm.create_job(session_id)
    
    # Mutate the job in the database to have an invalid session ID
    with jm.store._get_connection() as conn:
        conn.execute("PRAGMA foreign_keys = OFF")
        conn.execute("UPDATE jobs SET session_id = 'invalid-session-uuid' WHERE job_id = ?", (job_id,))
    
        
    # Attempting to run it will fail when trying to update the job status
    with pytest.raises(JobManagerError, match="invalid-session-uuid"):
        worker.run_job(job_id)

# 5) processing duplicate
@patch('src.backend.reconstruction_worker.reconstruct_video')
def test_5_processing_duplicate(mock_reconstruct, managers):
    sm, im, jm, worker = managers
    session_id = sm.create_session({})
    job_id = jm.create_job(session_id)
    jm.update_job_status(job_id, "processing")
    
    with pytest.raises(ValueError, match="is in status 'processing'"):
        worker.run_job(job_id)

# 6) completed rerun
@patch('src.backend.reconstruction_worker.reconstruct_video')
def test_6_completed_rerun(mock_reconstruct, managers):
    sm, im, jm, worker = managers
    session_id = sm.create_session({})
    job_id = jm.create_job(session_id)
    jm.update_job_status(job_id, "processing")
    jm.update_job_status(job_id, "completed")
    
    with pytest.raises(ValueError, match="is in status 'completed'"):
        worker.run_job(job_id)

# 7) failed rerun
@patch('src.backend.reconstruction_worker.reconstruct_video')
def test_7_failed_rerun(mock_reconstruct, managers):
    sm, im, jm, worker = managers
    session_id = sm.create_session({})
    job_id = jm.create_job(session_id)
    jm.update_job_status(job_id, "processing")
    jm.update_job_status(job_id, "failed", error="mock error")
    
    with pytest.raises(ValueError, match="is in status 'failed'"):
        worker.run_job(job_id)

# 8) unexpected exception
@patch('src.backend.reconstruction_worker.reconstruct_video')
def test_8_unexpected_exception(mock_reconstruct, managers):
    sm, im, jm, worker = managers
    session_id = sm.create_session({})
    create_mock_video(sm, im, session_id)
    job_id = jm.create_job(session_id)
    
    mock_reconstruct.side_effect = RuntimeError("GPU crash")
    
    job_data = worker.run_job(job_id)
    assert job_data["status"] == "failed"
    assert "RuntimeError" in job_data["error"]
    assert "GPU crash" in job_data["error"]

# 9) RECONSTRUCTION_BLOCKED
@patch('src.backend.reconstruction_worker.reconstruct_video')
def test_9_reconstruction_blocked(mock_reconstruct, managers):
    sm, im, jm, worker = managers
    session_id = sm.create_session({})
    create_mock_video(sm, im, session_id)
    job_id = jm.create_job(session_id)
    
    mock_reconstruct.return_value = {
        "status": "RECONSTRUCTION_BLOCKED",
        "recommended_action": "Missing calibration"
    }
    
    job_data = worker.run_job(job_id)
    assert job_data["status"] == "failed"
    assert job_data["error"] == "Missing calibration"

# 10) result metadata persistence
@patch('src.backend.reconstruction_worker.reconstruct_video')
def test_10_result_metadata_persistence(mock_reconstruct, managers):
    sm, im, jm, worker = managers
    session_id = sm.create_session({})
    create_mock_video(sm, im, session_id)
    job_id = jm.create_job(session_id)
    
    mock_reconstruct.return_value = {"status": "SUCCESS", "metric": True}
    worker.run_job(job_id)
    
    # Reload from disk through manager
    reloaded_job = jm.get_job(job_id)
    assert reloaded_job["result_metadata"]["status"] == "SUCCESS"
    assert reloaded_job["result_metadata"]["metric"] is True

# 11) correct output directory
@patch('src.backend.reconstruction_worker.reconstruct_video')
def test_11_correct_output_directory(mock_reconstruct, managers):
    sm, im, jm, worker = managers
    session_id = sm.create_session({})
    create_mock_video(sm, im, session_id)
    job_id = jm.create_job(session_id)
    
    mock_reconstruct.return_value = {"status": "SUCCESS"}
    worker.run_job(job_id)
    
    args_passed = mock_reconstruct.call_args[0][0]
    expected_workspace = str(sm.get_session_workspace(session_id))
    assert args_passed.output == expected_workspace

# 12) Session A/B isolation
@patch('src.backend.reconstruction_worker.reconstruct_video')
def test_12_session_ab_isolation(mock_reconstruct, managers):
    sm, im, jm, worker = managers
    sessA = sm.create_session({})
    sessB = sm.create_session({})
    
    create_mock_video(sm, im, sessA)
    # Don't put video in B
    
    jobA = jm.create_job(sessA)
    jobB = jm.create_job(sessB)
    
    mock_reconstruct.return_value = {"status": "SUCCESS"}
    
    # Run A
    worker.run_job(jobA)
    argsA = mock_reconstruct.call_args[0][0]
    workspaceA = str(sm.get_session_workspace(sessA))
    assert argsA.output == workspaceA
    
    mock_reconstruct.reset_mock()
    # Run B
    jobB_data = worker.run_job(jobB)
    assert jobB_data["status"] == "failed"
    mock_reconstruct.assert_not_called()

# 13) no absolute path leaks
@patch('src.backend.reconstruction_worker.reconstruct_video')
def test_13_no_absolute_path_leaks(mock_reconstruct, managers):
    sm, im, jm, worker = managers
    session_id = sm.create_session({})
    create_mock_video(sm, im, session_id)
    job_id = jm.create_job(session_id)
    
    workspace = str(sm.get_session_workspace(session_id))
    # inject workspace base dir to test scrubbing
    base_dir = str(sm.base_dir)
    mock_reconstruct.side_effect = RuntimeError(f"Crash at {base_dir}\\secret\\path")
    
    job_data = worker.run_job(job_id)
    assert job_data["status"] == "failed"
    assert "<WORKSPACE>\\secret\\path" in job_data["error"] 
    assert base_dir not in job_data["error"]

# Extra check for multiple videos rejection
@patch('src.backend.reconstruction_worker.reconstruct_video')
def test_14_rejects_multiple_videos(mock_reconstruct, managers):
    sm, im, jm, worker = managers
    session_id = sm.create_session({})
    create_mock_video(sm, im, session_id)
    
    # second video
    ws = sm.get_session_workspace(session_id)
    vid_path2 = ws / "dummy2.mp4"
    vid_path2.write_text("fake video")
    im.save_input(session_id, str(vid_path2), "dummy2.mp4", "video")
    
    job_id = jm.create_job(session_id)
    job_data = worker.run_job(job_id)
    assert job_data["status"] == "failed"
    assert "Multiple video inputs found" in job_data["error"]
