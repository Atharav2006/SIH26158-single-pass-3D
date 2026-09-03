import os
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.backend.session_manager import BackendSessionManager, SessionManagerError
from src.backend.job_manager import BackendJobManager, JobManagerError

class ResultManagerError(Exception):
    pass

class ResultConflictError(ResultManagerError):
    pass

class BackendResultManager:
    """Manages secure access and export of reconstruction result artifacts."""
    
    # Only these directories are scanned for results
    ALLOWED_RESULT_DIRS = ["geometry", "diagnostics", "exports"]
    
    def __init__(self, session_manager: BackendSessionManager, job_manager: BackendJobManager):
        self.session_manager = session_manager
        self.job_manager = job_manager

    def _validate_job_access(self, session_id: str, job_id: str) -> None:
        """Ensures the job belongs to the session and is completed."""
        try:
            job = self.job_manager.get_job(job_id)
        except JobManagerError as e:
            raise ResultManagerError(f"Cannot access results: {str(e)}")
            
        if job["session_id"] != session_id:
            # Mask existence of jobs belonging to other sessions for security
            raise ResultManagerError(f"Job {job_id} not found in session {session_id}")
            
        status = job["status"]
        if status != "completed":
            raise ResultConflictError(f"Results are unavailable because job is in '{status}' state.")

    def _generate_result_id(self, rel_path: str) -> str:
        """Generates a safe, logical result ID from a relative path (e.g. 'geometry_mesh_obj')."""
        return rel_path.replace("/", "_").replace("\\", "_").replace(".", "_")

    def _get_result_path_internal(self, workspace: Path, result_id: str) -> Optional[Path]:
        """Resolves a result_id back to an actual file by checking known result items."""
        # Find the actual path by scanning allowed directories and matching the generated ID
        all_results = self._scan_workspace_results(workspace)
        for item in all_results:
            if item["result_id"] == result_id:
                return Path(item["_absolute_path"])
        return None

    def _scan_workspace_results(self, workspace: Path) -> List[Dict[str, Any]]:
        """Scans the allowed directories for regular files and builds metadata."""
        results = []
        for d in self.ALLOWED_RESULT_DIRS:
            target_dir = workspace / d
            if not target_dir.is_dir():
                continue
                
            for root, dirs, files in os.walk(target_dir):
                root_path = Path(root)
                # Sort files to ensure deterministic order
                for file_name in sorted(files):
                    file_path = root_path / file_name
                    
                    # Security: Enforce that it's a regular file and resolve symlinks to prevent escapes
                    if not file_path.is_file():
                        continue
                    
                    resolved_file_path = file_path.resolve()
                    
                    # Ensure the resolved file is still within the target_dir
                    if not str(resolved_file_path).startswith(str(target_dir.resolve())):
                        continue
                        
                    rel_path = str(file_path.relative_to(workspace)).replace("\\", "/")
                    file_size = resolved_file_path.stat().st_size
                    
                    results.append({
                        "result_id": self._generate_result_id(rel_path),
                        "filename": file_name,
                        "logical_path": rel_path,
                        "size_bytes": file_size,
                        "_absolute_path": str(resolved_file_path)
                    })
                    
        # Deterministic sorting of final list based on logical_path
        results.sort(key=lambda x: x["logical_path"])
        return results

    def list_results(self, session_id: str, job_id: str) -> List[Dict[str, Any]]:
        """Lists available result artifacts for a completed job."""
        self._validate_job_access(session_id, job_id)
        
        try:
            workspace = self.session_manager.get_session_workspace(session_id)
        except SessionManagerError as e:
            raise ResultManagerError(f"Session error: {str(e)}")
            
        results = self._scan_workspace_results(workspace)
        
        # Remove internal absolute path from returned metadata
        for r in results:
            r.pop("_absolute_path", None)
            
        return results

    def get_result_path(self, session_id: str, job_id: str, result_id: str) -> Path:
        """Retrieves the safe, absolute Path to a specific result artifact."""
        self._validate_job_access(session_id, job_id)
        
        try:
            workspace = self.session_manager.get_session_workspace(session_id)
        except SessionManagerError as e:
            raise ResultManagerError(f"Session error: {str(e)}")
            
        # Validates and resolves the result_id
        # Null byte, absolute path, traversal etc. are inherently mitigated because
        # result_id must EXACTLY match a pre-scanned valid file ID.
        path = self._get_result_path_internal(workspace, result_id)
        if not path:
            raise ResultManagerError(f"Result '{result_id}' not found.")
            
        return path

    def export_result(self, session_id: str, job_id: str, result_id: str, destination_filename: str) -> Dict[str, Any]:
        """
        Exports a result artifact to the session's outputs directory.
        Rejects if destination_filename already exists.
        """
        # Strict validation on destination_filename to ensure it's just a basename
        if "\0" in destination_filename:
            raise ResultManagerError("Null bytes not allowed in destination filename.")
            
        if "/" in destination_filename or "\\" in destination_filename:
            raise ResultManagerError("Destination filename must not contain path separators.")
            
        if destination_filename in [".", ".."]:
            raise ResultManagerError("Invalid destination filename.")
            
        # Ensure we can read the source result
        source_path = self.get_result_path(session_id, job_id, result_id)
        
        # Get the outputs directory
        try:
            workspace = self.session_manager.get_session_workspace(session_id)
        except SessionManagerError as e:
            raise ResultManagerError(f"Session error: {str(e)}")
            
        outputs_dir = (workspace / "outputs").resolve()
        
        # In rare cases, outputs might not exist if creation failed, ensure it exists
        outputs_dir.mkdir(parents=True, exist_ok=True)
        
        dest_path = (outputs_dir / destination_filename).resolve()
        
        # Extra safeguard: ensure dest_path didn't traverse outside outputs_dir
        if not str(dest_path).startswith(str(outputs_dir)):
            raise ResultManagerError("Path traversal attempt detected in destination filename.")
            
        # Collision Policy: HTTP 409 Conflict if file already exists
        if dest_path.exists():
            raise ResultConflictError(f"Export destination '{destination_filename}' already exists.")
            
        # Safely copy the file
        temp_dest = dest_path.with_suffix(".tmp.exporting")
        try:
            # We copy to a temporary file in outputs_dir first, then rename, to avoid partial files on failure
            shutil.copy2(source_path, temp_dest)
            os.rename(temp_dest, dest_path)
        except Exception as e:
            # Clean up temp on failure
            if temp_dest.exists():
                try:
                    temp_dest.unlink()
                except Exception:
                    pass
            raise ResultManagerError(f"Export failed: {str(e)}")
            
        # Return stable metadata
        return {
            "result_id": result_id,
            "exported_filename": destination_filename,
            "size_bytes": dest_path.stat().st_size
        }
