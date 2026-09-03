import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional

from src.backend.session_manager import BackendSessionManager, SessionManagerError
from src.backend.metadata_store import MetadataStoreError

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
        # Access the store from the session manager to avoid double injection
        self.store = session_manager.store

    def _validate_job_id(self, job_id: str) -> None:
        if not isinstance(job_id, str):
            raise JobManagerError("Job ID must be a string.")
        try:
            val = uuid.UUID(job_id, version=4)
            if str(val) != job_id:
                raise ValueError()
        except ValueError:
            raise JobManagerError(f"Invalid job ID format: {job_id}")

    def create_job(self, session_id: str, reconstruction_mode: Optional[str] = None) -> str:
        """Creates a new job for a given session."""
        with self.session_manager.session_lock(session_id):
            if not self.session_manager.session_exists(session_id):
                raise JobManagerError(f"Session {session_id} does not exist.")

            job_id = str(uuid.uuid4())
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

            try:
                self.store.create_job(job_data)
            except MetadataStoreError as e:
                raise JobManagerError(str(e))
            
            # Update session to reflect the job exists and its initial status
            self._update_session_with_job(session_id, job_id, "queued")

            return job_id

    def list_jobs(self, session_id: str) -> list[Dict[str, Any]]:
        """Retrieves all jobs for a given session."""
        if not self.session_manager.session_exists(session_id):
            raise JobManagerError(f"Session {session_id} does not exist.")
        try:
            return self.store.list_jobs(session_id)
        except MetadataStoreError as e:
            raise JobManagerError(str(e))

    def get_job(self, job_id: str) -> Dict[str, Any]:
        """Retrieves a job by its ID."""
        self._validate_job_id(job_id)
        try:
            return self.store.get_job(job_id)
        except MetadataStoreError as e:
            raise JobManagerError(str(e))

    def update_job_status(self, job_id: str, status: str, error: Optional[str] = None, result_metadata: Optional[dict] = None) -> None:
        """Updates the status and lifecycle timestamps of a job."""
        if status not in self.VALID_STATUSES:
            raise JobManagerError(f"Invalid status: {status}")

        try:
            job_data = self.store.get_job(job_id)
        except MetadataStoreError as e:
            raise JobManagerError(str(e))
            
        session_id = job_data["session_id"]
        current_status = job_data["status"]

        if status not in self.VALID_TRANSITIONS.get(current_status, set()):
            raise JobManagerError(f"Invalid lifecycle transition from {current_status} to {status}.")

        now = datetime.now(timezone.utc).isoformat()
        updates = {
            "status": status,
            "updated_at": now
        }

        if status == "processing":
            updates["started_at"] = now
        elif status == "completed":
            updates["completed_at"] = now
        elif status == "failed":
            if error is None:
                raise JobManagerError("Error message must be provided when status is failed.")
            updates["error"] = error
            
        if result_metadata is not None:
            updates["result_metadata"] = result_metadata

        try:
            self.store.update_job(job_id, updates)
        except MetadataStoreError as e:
            raise JobManagerError(str(e))
            
        self._update_session_with_job(session_id, job_id, status)

    def _update_session_with_job(self, session_id: str, job_id: str, status: str) -> None:
        """Syncs the job lifecycle back to the owning session."""
        try:
            self.session_manager.update_metadata(session_id, {
                "status": status,
                "output_metadata": {
                    "active_job_id": job_id,
                    "status": status
                }
            })
        except SessionManagerError as e:
            raise JobManagerError(f"Failed to update owning session {session_id}: {str(e)}")
