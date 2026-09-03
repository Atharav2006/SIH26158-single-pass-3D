import pytest
import time
import threading
from typing import Dict, Any
from unittest.mock import MagicMock, patch

from src.backend.session_manager import BackendSessionManager
from src.backend.input_manager import BackendInputManager
from src.backend.job_manager import BackendJobManager
from src.backend.reconstruction_worker import BackendReconstructionWorker
from src.backend.execution_manager import BackgroundExecutionManager

@pytest.fixture
def managers():
    sm = BackendSessionManager()
    im = BackendInputManager(sm)
    jm = BackendJobManager(sm)
    worker = BackendReconstructionWorker(sm, im, jm)
    return sm, im, jm, worker

def test_1_execution_manager_constructed(managers):
    sm, im, jm, worker = managers
    em = BackgroundExecutionManager(worker)
    assert em.worker == worker
    em.shutdown()

def test_2_queued_job_can_be_submitted(managers):
    sm, im, jm, worker = managers
    session_id = sm.create_session({})
    job_id = jm.create_job(session_id)
    em = BackgroundExecutionManager(worker)
    
    # Mock execute_job so it doesn't actually run anything real
    with patch.object(em, '_execute_job') as mock_exec:
        future = em.submit(job_id, session_id)
        assert future is not None
        # Let it finish
        future.result()
        mock_exec.assert_called_once_with(job_id)
        
    em.shutdown()

def test_3_submit_returns_without_waiting(managers):
    sm, im, jm, worker = managers
    session_id = sm.create_session({})
    job_id = jm.create_job(session_id)
    em = BackgroundExecutionManager(worker)
    
    event = threading.Event()
    
    def slow_exec(j_id):
        event.wait()
        
    with patch.object(em, '_execute_job', side_effect=slow_exec):
        start_time = time.time()
        future = em.submit(job_id, session_id)
        end_time = time.time()
        
        # Should return almost instantly
        assert (end_time - start_time) < 0.5 
        assert not future.done()
        
        event.set()
        future.result()
        
    em.shutdown()

def test_4_5_submitted_job_reaches_processing_completed(managers):
    sm, im, jm, worker = managers
    session_id = sm.create_session({})
    job_id = jm.create_job(session_id)
    em = BackgroundExecutionManager(worker)
    
    # We'll patch worker.run_job so we can monitor the state correctly
    # without doing actual reconstruction, but letting it update state.
    event = threading.Event()
    
    def delayed_run_job(j_id):
        jm.update_job_status(j_id, "processing")
        event.wait()
        jm.update_job_status(j_id, "completed")
        return jm.get_job(j_id)
        
    with patch.object(worker, 'run_job', side_effect=delayed_run_job):
        future = em.submit(job_id, session_id)
        
        # Wait a tiny bit to ensure thread starts
        time.sleep(0.1)
        # Should be processing
        job = jm.get_job(job_id)
        assert job["status"] == "processing"
        
        # Finish it
        event.set()
        future.result()
        
        # Should be completed
        job = jm.get_job(job_id)
        assert job["status"] == "completed"
        
    em.shutdown()

def test_6_reconstruction_failure_results_in_failed(managers):
    sm, im, jm, worker = managers
    session_id = sm.create_session({})
    job_id = jm.create_job(session_id)
    em = BackgroundExecutionManager(worker)
    
    def failing_run_job(j_id):
        jm.update_job_status(j_id, "processing")
        jm.update_job_status(j_id, "failed", error="Failed locally")
        return jm.get_job(j_id)
        
    with patch.object(worker, 'run_job', side_effect=failing_run_job):
        future = em.submit(job_id, session_id)
        future.result()
        
        job = jm.get_job(job_id)
        assert job["status"] == "failed"
        
    em.shutdown()

def test_7_unexpected_worker_exception_results_in_failed(managers):
    sm, im, jm, worker = managers
    session_id = sm.create_session({})
    job_id = jm.create_job(session_id)
    em = BackgroundExecutionManager(worker)
    
    def crash_run_job(j_id):
        jm.update_job_status(j_id, "processing")
        raise RuntimeError("Total catastrophe")
        
    with patch.object(worker, 'run_job', side_effect=crash_run_job):
        future = em.submit(job_id, session_id)
        # The future will raise the exception
        with pytest.raises(RuntimeError):
            future.result()
        
        job = jm.get_job(job_id)
        assert job["status"] == "failed"
        assert "Total catastrophe" in job["error"]
        
    em.shutdown()

def test_8_duplicate_submission_rejected(managers):
    sm, im, jm, worker = managers
    session_id = sm.create_session({})
    job_id = jm.create_job(session_id)
    em = BackgroundExecutionManager(worker)
    
    event = threading.Event()
    def slow_exec(j_id):
        event.wait()
        
    with patch.object(em, '_execute_job', side_effect=slow_exec):
        future1 = em.submit(job_id, session_id)
        
        with pytest.raises(ValueError, match="already submitted"):
            em.submit(job_id, session_id)
            
        event.set()
        future1.result()
        
    em.shutdown()

def test_8b_concurrent_duplicate_submission_race(managers):
    sm, im, jm, worker = managers
    session_id = sm.create_session({})
    job_id = jm.create_job(session_id)
    em = BackgroundExecutionManager(worker)
    
    # We will track exactly how many times the real worker logic is invoked
    execute_count = 0
    exec_lock = threading.Lock()
    
    run_started_event = threading.Event()
    finish_run_event = threading.Event()

    def controlled_run(j_id):
        jm.update_job_status(j_id, "processing")
        nonlocal execute_count
        with exec_lock:
            execute_count += 1
        run_started_event.set()
        finish_run_event.wait()
        jm.update_job_status(j_id, "completed")
        return jm.get_job(j_id)
        
    # We want multiple threads to call em.submit concurrently
    barrier = threading.Barrier(3)
    
    exceptions = []
    futures = []
    
    def thread_worker():
        barrier.wait() # Wait for all threads to be ready
        try:
            f = em.submit(job_id, session_id)
            futures.append(f)
        except Exception as e:
            exceptions.append(e)
            
    with patch.object(worker, 'run_job', side_effect=controlled_run):
        t1 = threading.Thread(target=thread_worker)
        t2 = threading.Thread(target=thread_worker)
        t3 = threading.Thread(target=thread_worker)
        
        t1.start()
        t2.start()
        t3.start()
        
        # We know one job gets submitted and starts processing, which sets run_started_event.
        # But wait, t1.join() would block forever if finish_run_event is not set!
        # So we should wait for run_started_event, then we know at least one thread submitted it and the worker started.
        run_started_event.wait()
        
        # Now all other threads will either be waiting for the lock, or have failed.
        # We can safely allow the worker to finish.
        finish_run_event.set()
        
        t1.join()
        t2.join()
        t3.join()
        
        # Exactly one submission should succeed
        assert len(futures) == 1
        # The other two should have raised ValueError
        assert len(exceptions) == 2
        for exc in exceptions:
            assert isinstance(exc, ValueError)
            
        # Ensure the underlying execution only happened once
        futures[0].result()
        assert execute_count == 1

    em.shutdown()

def test_9_10_11_invalid_states_cannot_be_resubmitted(managers):
    sm, im, jm, worker = managers
    session_id = sm.create_session({})
    em = BackgroundExecutionManager(worker)
    
    # 9. Processing
    job_id1 = jm.create_job(session_id)
    jm.update_job_status(job_id1, "processing")
    with pytest.raises(ValueError, match="not 'queued'"):
        em.submit(job_id1, session_id)
        
    # 10. Completed
    job_id2 = jm.create_job(session_id)
    jm.update_job_status(job_id2, "processing")
    jm.update_job_status(job_id2, "completed")
    with pytest.raises(ValueError, match="not 'queued'"):
        em.submit(job_id2, session_id)
        
    # 11. Failed
    job_id3 = jm.create_job(session_id)
    jm.update_job_status(job_id3, "processing")
    jm.update_job_status(job_id3, "failed", error="x")
    with pytest.raises(ValueError, match="not 'queued'"):
        em.submit(job_id3, session_id)

    em.shutdown()

def test_12_missing_job_rejected(managers):
    sm, im, jm, worker = managers
    session_id = sm.create_session({})
    em = BackgroundExecutionManager(worker)
    
    import uuid
    valid_uuid = str(uuid.uuid4())
    with pytest.raises(ValueError, match="not found"):
        em.submit(valid_uuid, session_id)
        
    em.shutdown()

def test_13_session_job_mismatch_rejected(managers):
    sm, im, jm, worker = managers
    session_a = sm.create_session({})
    session_b = sm.create_session({})
    
    job_id = jm.create_job(session_a)
    em = BackgroundExecutionManager(worker)
    
    with pytest.raises(ValueError, match="does not belong to the requested session"):
        em.submit(job_id, session_b)
        
    em.shutdown()

def test_14_15_independent_jobs_and_isolation(managers):
    sm, im, jm, worker = managers
    session_a = sm.create_session({})
    session_b = sm.create_session({})
    
    job_a = jm.create_job(session_a)
    job_b = jm.create_job(session_b)
    
    em = BackgroundExecutionManager(worker, max_workers=2)
    
    event_a = threading.Event()
    event_b = threading.Event()
    
    def controlled_run(j_id):
        if j_id == job_a:
            event_a.wait()
            jm.update_job_status(j_id, "processing")
            jm.update_job_status(j_id, "completed")
        else:
            event_b.wait()
            jm.update_job_status(j_id, "processing")
            jm.update_job_status(j_id, "completed")
            
    with patch.object(worker, 'run_job', side_effect=controlled_run):
        f_a = em.submit(job_a, session_a)
        f_b = em.submit(job_b, session_b)
        
        # Both should be running independently
        assert not f_a.done()
        assert not f_b.done()
        
        event_a.set()
        f_a.result()
        
        assert jm.get_job(job_a)["status"] == "completed"
        assert not f_b.done() # B still waiting
        
        event_b.set()
        f_b.result()
        assert jm.get_job(job_b)["status"] == "completed"
        
    em.shutdown()

def test_16_executor_shutdown(managers):
    sm, im, jm, worker = managers
    em = BackgroundExecutionManager(worker)
    
    em.shutdown(wait=True)
    
    # Should not be able to submit after shutdown (ThreadPoolExecutor raises RuntimeError)
    session_id = sm.create_session({})
    job_id = jm.create_job(session_id)
    with pytest.raises(RuntimeError):
        em.submit(job_id, session_id)

def test_17_18_19_api_integration():
    from fastapi.testclient import TestClient
    from src.backend.api import app
    
    client = TestClient(app)
    
    # Need to override dependencies to use a controlled event/mock so we can test the API
    with patch('src.backend.api._execution_manager._execute_job') as mock_exec:
        event = threading.Event()
        def block_exec(j_id):
            event.wait()
        mock_exec.side_effect = block_exec
        
        # Create session
        res = client.post("/sessions", json={})
        session_id = res.json()["session_id"]
        
        # Create job
        res = client.post(f"/sessions/{session_id}/jobs", json={})
        job_id = res.json()["job_id"]
        
        # Submit job
        res = client.post(f"/sessions/{session_id}/jobs/{job_id}/submit")
        assert res.status_code == 202
        assert res.json()["status"] == "submitted"
        
        # Check API status
        res = client.get(f"/jobs/{job_id}")
        assert res.json()["status"] == "queued"  # Hasn't started processing because of our mock logic
        
        event.set()

def test_20_no_absolute_path_leaks_in_execution_error(managers):
    sm, im, jm, worker = managers
    session_id = sm.create_session({})
    job_id = jm.create_job(session_id)
    em = BackgroundExecutionManager(worker)
    
    workspace_path = str(sm.base_dir)
    
    def crash_with_path(j_id):
        raise RuntimeError(f"Failed at {workspace_path}/some/file")
        
    with patch.object(worker, 'run_job', side_effect=crash_with_path):
        future = em.submit(job_id, session_id)
        with pytest.raises(RuntimeError):
            future.result()
            
        job = jm.get_job(job_id)
        assert workspace_path not in job["error"]
        assert "<WORKSPACE>" in job["error"]

    em.shutdown()
