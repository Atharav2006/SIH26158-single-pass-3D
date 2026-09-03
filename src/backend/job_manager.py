import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional

from src.backend.session_manager import BackendSessionManager, SessionManagerError

class JobManagerError(Exception):
    pass

class BackendJobManager:
    """Manages reconstruction lifecycle jobs associated with backend sessions."""

    VALID_STATUSES = {"queued", "processing", "completed", "failed"}
    VALID_TRANSITIONS = {
        "queued": {"processing", "failed"},
        "processing": {"completed", "failed"},
        "completed": set(),
        "failed": set(),
    }

    def __init__(self, session_manager: BackendSessionManager):
        self.session_manager = session_manager

    def _validate_job_id(self, job_id: str) -> None:
        if not isinstance(job_id, str):
            raise JobManagerError("Job ID must be a string.")
        try:
            val = uuid.UUID(job_id, version=4)
            if str(val) != job_id:
                raise ValueError()
        except ValueError:
            raise JobManagerError(f"Invalid job ID format: {job_id}")

    def _get_jobs_dir(self, session_id: str) -> Path:
        """Returns the isolated jobs directory for a specific session."""
        workspace = self.session_manager.get_session_workspace(session_id)
        jobs_dir = workspace / "metadata" / "jobs"
        jobs_dir.mkdir(parents=True, exist_ok=True)
        return jobs_dir

    def _find_session_for_job(self, job_id: str) -> str:
        """Locates the owning session for a given job by searching workspaces."""
        self._validate_job_id(job_id)
        # Search all session directories for this job
        for session_dir in self.session_manager.base_dir.iterdir():
            if session_dir.is_dir():
                try:
                    # Quick validation that this is actually a valid session
                    # We check UUID format to avoid path traversal logic
                    uuid.UUID(session_dir.name, version=4)
                    job_file = session_dir / "metadata" / "jobs" / f"{job_id}.json"
                    if job_file.is_file():
                        return session_dir.name
                except ValueError:
                    continue
        raise JobManagerError(f"Job {job_id} not found.")

    def create_job(self, session_id: str, reconstruction_mode: Optional[str] = None) -> str:
        """Creates a new job for a given session."""
        if not self.session_manager.session_exists(session_id):
            raise JobManagerError(f"Session {session_id} does not exist.")

        job_id = str(uuid.uuid4())
        jobs_dir = self._get_jobs_dir(session_id)
        job_file = jobs_dir / f"{job_id}.json"

        # Check if job already exists (extremely unlikely due to UUID4)
        if job_file.exists():
            raise JobManagerError(f"Job {job_id} already exists.")

        now = datetime.now(timezone.utc).isoformat()
        
        job_data = {
            "job_id": job_id,
            "session_id": session_id,
            "status": "queued",
            "reconstruction_mode": reconstruction_mode,
            "created_at": now,
            "updated_at": now,
            "started_at": None,
            "completed_at": None,
            "error": None,
            "result_metadata": {}
        }

        self._write_job_metadata(session_id, job_id, job_data)
        
        # Update session to reflect the job exists and its initial status
        self._update_session_with_job(session_id, job_id, "queued")

        return job_id

    def get_job(self, job_id: str) -> Dict[str, Any]:
        """Retrieves a job by its ID, locating its owning session first."""
        session_id = self._find_session_for_job(job_id)
        jobs_dir = self._get_jobs_dir(session_id)
        job_file = jobs_dir / f"{job_id}.json"

        try:
            with open(job_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            raise JobManagerError(f"Failed to read job {job_id}: {str(e)}")

    def update_job_status(self, job_id: str, status: str, error: Optional[str] = None, result_metadata: Optional[dict] = None) -> None:
        """Updates the status and lifecycle timestamps of a job."""
        if status not in self.VALID_STATUSES:
            raise JobManagerError(f"Invalid status: {status}")

        session_id = self._find_session_for_job(job_id)
        job_data = self.get_job(job_id)
        
        current_status = job_data["status"]

        if status not in self.VALID_TRANSITIONS.get(current_status, set()):
            raise JobManagerError(f"Invalid lifecycle transition from {current_status} to {status}.")

        now = datetime.now(timezone.utc).isoformat()
        job_data["status"] = status
        job_data["updated_at"] = now

        if status == "processing":
            job_data["started_at"] = now
        elif status == "completed":
            job_data["completed_at"] = now
        elif status == "failed":
            if error is None:
                raise JobManagerError("Error message must be provided when status is failed.")
            job_data["error"] = error
            
        if result_metadata is not None:
            job_data["result_metadata"] = result_metadata

        self._write_job_metadata(session_id, job_id, job_data)
        self._update_session_with_job(session_id, job_id, status)

    def _write_job_metadata(self, session_id: str, job_id: str, data: Dict[str, Any]) -> None:
        """Atomically writes job metadata to disk."""
        jobs_dir = self._get_jobs_dir(session_id)
        job_file = jobs_dir / f"{job_id}.json"
        temp_file = jobs_dir / f"{job_id}.json.tmp"
        
        try:
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)
            temp_file.replace(job_file)
        except Exception as e:
            if temp_file.exists():
                temp_file.unlink()
            raise JobManagerError(f"Failed to write job {job_id}: {str(e)}")

    def _update_session_with_job(self, session_id: str, job_id: str, status: str) -> None:
        """Syncs the job lifecycle back to the owning session."""
        try:
            self.session_manager.update_metadata(session_id, {
                "active_job_id": job_id,
                "status": status
            })
        except SessionManagerError as e:
            raise JobManagerError(f"Failed to update owning session {session_id}: {str(e)}")
