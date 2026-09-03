import os
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

from src.backend.session_manager import BackendSessionManager, SessionManagerError

class InputManagerError(Exception):
    pass

class BackendInputManager:
    """Secure file staging and validation layer for session inputs."""

    ALLOWED_EXTENSIONS = {
        ".mp4", ".mov", ".avi", ".mkv", ".webm", ".tif", ".tiff",  # Video/Images
        ".csv", ".txt", ".json", ".yaml", ".yml", ".bin"           # Data/Metadata
    }

    def __init__(self, session_manager: BackendSessionManager, max_file_size_bytes: int = 10 * 1024 * 1024 * 1024):
        # Default 10GB max file size, configurable
        self.session_manager = session_manager
        self.max_file_size_bytes = max_file_size_bytes

    def _check_input_lock(self, session_id: str) -> None:
        """
        Checks whether the session is locked from input mutations.
        Inputs cannot be modified if a job is in 'queued', 'processing', or 'completed' state.
        Modifications are allowed if there are no jobs or all jobs are 'failed'.
        """
        try:
            # Check jobs directly from the metadata store.
            jobs = self.session_manager.store.list_jobs(session_id)
            for job in jobs:
                if job.get("status") in ("queued", "processing", "completed"):
                    raise InputManagerError(f"Inputs are locked because job {job['job_id']} is in '{job.get('status')}' state.")
        except InputManagerError:
            raise
        except Exception:
            # If session doesn't exist or DB error, let downstream logic handle it
            pass

    def _sanitize_filename(self, filename: str) -> str:
        """Removes dangerous characters, path separators, and null bytes from a filename."""
        if not filename:
            raise InputManagerError("Filename cannot be empty.")
            
        # Strip null bytes
        clean_name = filename.replace("\0", "")
        # Remove any path separators
        clean_name = os.path.basename(clean_name)
        
        # Optionally, restrict to alphanumeric, dot, underscore, dash
        clean_name = re.sub(r'[^A-Za-z0-9_.-]', '_', clean_name)
        
        if not clean_name:
            raise InputManagerError("Filename is invalid after sanitization.")
            
        return clean_name

    def _validate_extension(self, filename: str) -> str:
        """Validates that the file extension is allowed."""
        ext = Path(filename).suffix.lower()
        if ext not in self.ALLOWED_EXTENSIONS:
            raise InputManagerError(f"Unsupported file extension: '{ext}'. Allowed: {', '.join(self.ALLOWED_EXTENSIONS)}")
        return ext

    def _get_inputs_dir(self, session_id: str) -> Path:
        """Returns the isolated inputs directory for a session."""
        try:
            workspace = self.session_manager.get_session_workspace(session_id)
            inputs_dir = workspace / "inputs"
            if not inputs_dir.is_dir():
                raise InputManagerError(f"Inputs directory missing for session {session_id}")
            return inputs_dir
        except SessionManagerError as e:
            raise InputManagerError(f"Session error: {str(e)}")

    def _update_session_files_metadata(self, session_id: str, new_file_record: Optional[Dict[str, Any]] = None, remove_stored_filename: Optional[str] = None):
        """Safely updates the 'files' array in the session metadata."""
        try:
            session_data = self.session_manager.get_session(session_id)
        except SessionManagerError as e:
            raise InputManagerError(f"Could not retrieve session: {str(e)}")
            
        input_meta = session_data.get("input_metadata", {})
        files: List[Dict[str, Any]] = input_meta.get("files", [])
        
        # Remove an existing entry if requested (e.g. for delete or overwrite)
        if remove_stored_filename:
            files = [f for f in files if f.get("stored_filename") != remove_stored_filename]
            
        if new_file_record:
            # Prevent exact duplicate records
            existing_names = {f.get("stored_filename") for f in files}
            if new_file_record["stored_filename"] not in existing_names:
                files.append(new_file_record)
                
        input_meta["files"] = files
        
        try:
            self.session_manager.update_metadata(session_id, {"input_metadata": input_meta})
        except SessionManagerError as e:
            raise InputManagerError(f"Failed to update session metadata: {str(e)}")

    def save_input(self, session_id: str, source_path: Union[str, Path], original_filename: str, input_type: Optional[str] = None) -> Dict[str, Any]:
        """Safely stages a file into the session's isolated inputs directory."""
        with self.session_manager.session_lock(session_id):
            self._check_input_lock(session_id)
            source_path = Path(source_path)
            
            if not source_path.is_file():
                raise InputManagerError(f"Source file does not exist or is not a file: {source_path}")
                
            file_size = source_path.stat().st_size
            if file_size > self.max_file_size_bytes:
                raise InputManagerError(f"File size {file_size} exceeds maximum allowed {self.max_file_size_bytes} bytes.")
                
            ext = self._validate_extension(original_filename)
            sanitized_original = self._sanitize_filename(original_filename)
            
            # Generate a secure stored filename: <uuid4>_<sanitized>
            stored_filename = f"{uuid.uuid4()}_{sanitized_original}"
            
            inputs_dir = self._get_inputs_dir(session_id)
            dest_path = (inputs_dir / stored_filename).resolve()
            
            # Security: ensure path resolves strictly inside inputs directory
            if not str(dest_path).startswith(str(inputs_dir.resolve())):
                raise InputManagerError("Path traversal attempt detected during save.")
                
            # Copy the file
            try:
                shutil.copy2(source_path, dest_path)
            except Exception as e:
                if dest_path.exists():
                    dest_path.unlink()
                raise InputManagerError(f"Failed to copy file safely: {str(e)}")
                
            # Register in session metadata
            now = datetime.now(timezone.utc).isoformat()
            file_record = {
                "stored_filename": stored_filename,
                "original_filename": original_filename,
                "input_type": input_type,
                "size_bytes": file_size,
                "extension": ext,
                "created_at": now
            }
            
            self._update_session_files_metadata(session_id, new_file_record=file_record)
            return file_record

    def get_input_path(self, session_id: str, stored_filename: str) -> Path:
        """Retrieves and validates the secure path to a stored input file."""
        if not stored_filename or "/" in stored_filename or "\\" in stored_filename or ".." in stored_filename:
            raise InputManagerError("Invalid stored_filename format.")
            
        inputs_dir = self._get_inputs_dir(session_id)
        file_path = (inputs_dir / stored_filename).resolve()
        
        if not str(file_path).startswith(str(inputs_dir.resolve())):
            raise InputManagerError("Path traversal attempt detected.")
            
        if not file_path.is_file():
            raise InputManagerError(f"File {stored_filename} not found in session {session_id}.")
            
        return file_path

    def list_inputs(self, session_id: str) -> List[Dict[str, Any]]:
        """Lists metadata for all files staged in the session."""
        try:
            session_data = self.session_manager.get_session(session_id)
            return session_data.get("input_metadata", {}).get("files", [])
        except SessionManagerError as e:
            raise InputManagerError(f"Invalid session: {str(e)}")

    def delete_input(self, session_id: str, stored_filename: str) -> None:
        """Deletes a staged file and removes its metadata record."""
        with self.session_manager.session_lock(session_id):
            self._check_input_lock(session_id)
            # get_input_path ensures the file exists and is strictly within inputs_dir
            file_path = self.get_input_path(session_id, stored_filename)
            
            try:
                file_path.unlink()
            except Exception as e:
                raise InputManagerError(f"Failed to delete file {stored_filename}: {str(e)}")
                
            self._update_session_files_metadata(session_id, remove_stored_filename=stored_filename)
