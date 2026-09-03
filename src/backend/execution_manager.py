import threading
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Dict, Optional, Any

from .reconstruction_worker import BackendReconstructionWorker
from .job_manager import JobManagerError


class BackgroundExecutionManager:
    """
    Local-process background execution manager for reconstruction jobs.
    Uses a ThreadPoolExecutor to run reconstruction synchronously on a background thread.
    Not designed for distributed execution or durable queueing across process restarts.
    """

    def __init__(self, worker: BackendReconstructionWorker, max_workers: int = 2):
        self.worker = worker
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="ReconWorker")
        self.futures: Dict[str, Future] = {}
        self._lock = threading.Lock()
        
    def submit(self, job_id: str, session_id: str) -> Future:
        """
        Submits a job for background execution.
        Validates job state and session ownership before submitting.
        Prevents duplicate execution.
        """
        with self._lock:
            try:
                job_data = self.worker.job_manager.get_job(job_id)
            except JobManagerError as e:
                raise ValueError(str(e))
                
            if job_data["session_id"] != session_id:
                raise ValueError("Job does not belong to the requested session.")
                
            if job_data["status"] != "queued":
                raise ValueError(f"Job is in status '{job_data['status']}', not 'queued'.")
                
            if job_id in self.futures and not self.futures[job_id].done():
                raise ValueError("Job is already submitted and running.")
                
            future = self.executor.submit(self._execute_job, job_id)
            self.futures[job_id] = future
            
        return future

    def _execute_job(self, job_id: str) -> Dict[str, Any]:
        """
        Internal target for background thread execution.
        Any unexpected exceptions here are caught and the job is marked failed.
        """
        try:
            # The run_job method handles setting status to 'processing'
            # and transitions to 'completed' or 'failed' appropriately.
            return self.worker.run_job(job_id)
        except Exception as e:
            # Catch unexpected exceptions in the thread so we don't just swallow them.
            # Usually the worker catches it, but just in case:
            try:
                base_dir = str(self.worker.session_manager.base_dir)
                safe_err = str(e).replace(base_dir, "<WORKSPACE>")
                self.worker.job_manager.update_job_status(job_id, "failed", error=f"Unexpected background error: {safe_err}")
            except Exception:
                pass
            raise

    def get_future(self, job_id: str) -> Optional[Future]:
        with self._lock:
            return self.futures.get(job_id)

    def shutdown(self, wait: bool = True):
        """Shuts down the underlying executor."""
        self.executor.shutdown(wait=wait)

    def reap_stuck_jobs(self):
        """
        Scans for jobs left in 'queued' or 'processing' states upon backend initialization,
        which indicates they were orphaned by a server restart. Marks them as failed.
        """
        store = self.worker.job_manager.store
        try:
            with store._get_connection() as conn:
                cursor = conn.execute("SELECT job_id FROM jobs WHERE status IN ('queued', 'processing')")
                stuck_job_ids = [row[0] for row in cursor.fetchall()]
                
            for j_id in stuck_job_ids:
                # Use JobManager to properly sync the failed status back to the session
                try:
                    self.worker.job_manager.update_job_status(
                        j_id, 
                        status="failed", 
                        error="Job interrupted by backend restart"
                    )
                except Exception:
                    pass
        except Exception:
            pass
