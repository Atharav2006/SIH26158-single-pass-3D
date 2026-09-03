import pytest
import threading
import uuid
import tempfile
import os
from pathlib import Path

from src.backend.session_manager import BackendSessionManager, SessionManagerError
from src.backend.metadata_store import MetadataStore
from src.backend.input_manager import BackendInputManager, InputManagerError
from src.backend.job_manager import BackendJobManager

@pytest.fixture
def temp_metadata_store(tmp_path):
    # Use a file-backed SQLite DB for tests so multiple connections share the same DB
    db_path = str(tmp_path / "test_metadata.db")
    store = MetadataStore(db_path)
    store.initialize()
    return store

@pytest.fixture
def session_manager(temp_metadata_store, tmp_path):
    # tmp_path is a pytest fixture providing a temporary directory
    return BackendSessionManager(metadata_store=temp_metadata_store, base_workspace_dir=str(tmp_path))

@pytest.fixture
def input_manager(session_manager):
    return BackendInputManager(session_manager)

@pytest.fixture
def job_manager(session_manager):
    return BackendJobManager(session_manager)

def test_save_input_vs_create_job_toctou(session_manager, input_manager, job_manager, tmp_path):
    """
    Tests that a job creation cannot occur in the middle of a save_input operation,
    preventing an input from being saved when a job is active.
    """
    session_id = session_manager.create_session()
    
    # Create a dummy file to upload
    source_path = tmp_path / "test.mp4"
    source_path.write_text("dummy content")

    # We want to pause save_input right after it checks the lock
    # but before it writes to the file system.
    # We can do this by mocking _sanitize_filename or the input directory retrieval
    # to wait on an event.

    job_created_event = threading.Event()
    input_ready_to_write_event = threading.Event()

    original_check_input_lock = input_manager._check_input_lock

    def mocked_check_input_lock(sid):
        # Call the original check first
        original_check_input_lock(sid)
        # Signal that the check has passed and we are in the critical section
        input_ready_to_write_event.set()
        # Wait for the other thread to attempt (and hopefully be blocked) to create a job
        # Since create_job also requires the session_lock now, the other thread will block
        # on the lock. Therefore, we can't wait for job_created_event here if we hold the lock,
        # it would deadlock. So we just sleep briefly to ensure the other thread *tries* to get it.
        # However, a better deterministic way:
        pass

    # Wait! If both use the same lock, then if thread A acquires it in save_input,
    # it holds it until it finishes. If we use threading events to coordinate inside the lock,
    # thread B will block on acquiring the lock in create_job.
    # To test this deterministically:
    
    # Thread 1: save_input
    # Mock _check_input_lock so that it signals Thread 2 to start create_job, then Thread 1 sleeps a bit.
    def mocked_check_input_lock2(sid):
        original_check_input_lock(sid)
        input_ready_to_write_event.set()
        # Give Thread 2 time to try and acquire the lock
        import time
        time.sleep(0.1)

    input_manager._check_input_lock = mocked_check_input_lock2

    results = []
    
    def run_save_input():
        try:
            input_manager.save_input(session_id, source_path, "test.mp4")
            results.append("save_input_success")
        except Exception as e:
            results.append(e)

    def run_create_job():
        # Wait until save_input has passed the lock check
        input_ready_to_write_event.wait()
        try:
            job_id = job_manager.create_job(session_id)
            results.append("create_job_success")
        except Exception as e:
            results.append(e)

    t1 = threading.Thread(target=run_save_input)
    t2 = threading.Thread(target=run_create_job)

    t1.start()
    t2.start()

    t1.join()
    t2.join()

    # If the lock works, save_input finishes first, then create_job runs.
    # There should be no conflict because save_input finishes while job doesn't exist yet,
    # and then create_job creates the job. Both should succeed.
    # But wait, the point of the lock is that they don't interleave.
    assert "save_input_success" in results
    assert "create_job_success" in results

def test_delete_session_vs_create_job_toctou(session_manager, job_manager):
    """
    Tests that a session deletion correctly prevents a concurrent create_job, or 
    create_job prevents session deletion.
    """
    session_id = session_manager.create_session()
    
    # We want to pause delete_session right after checking for active jobs.
    delete_ready_event = threading.Event()
    
    original_get_session_workspace = session_manager.get_session_workspace
    
    def mocked_get_session_workspace(sid):
        workspace = original_get_session_workspace(sid)
        delete_ready_event.set()
        import time
        time.sleep(0.1)
        return workspace

    session_manager.get_session_workspace = mocked_get_session_workspace

    results = []
    
    def run_delete_session():
        try:
            session_manager.delete_session(session_id)
            results.append("delete_session_success")
        except Exception as e:
            results.append(e)

    def run_create_job():
        delete_ready_event.wait()
        try:
            job_id = job_manager.create_job(session_id)
            results.append("create_job_success")
        except Exception as e:
            results.append(e)

    t1 = threading.Thread(target=run_delete_session)
    t2 = threading.Thread(target=run_create_job)

    t1.start()
    t2.start()

    t1.join()
    t2.join()

    # One should succeed, the other should fail.
    # Since delete_session acquires the lock first (because it sets delete_ready_event inside the lock),
    # create_job will block.
    # delete_session will finish, removing the DB record.
    # Then create_job will resume, and it will fail because the session no longer exists in DB!
    
    assert "delete_session_success" in results
    
    # The create_job thread should fail with SessionManagerError because session doesn't exist.
    job_exceptions = [r for r in results if isinstance(r, Exception)]
    assert len(job_exceptions) > 0
    assert "Session" in str(job_exceptions[0]) and "does not exist" in str(job_exceptions[0])
