import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any

class SessionManagerError(Exception):
    pass

class BackendSessionManager:
    """Manages secure, isolated backend reconstruction sessions."""
    
    REQUIRED_DIRS = ["inputs", "temp", "outputs", "metadata", "logs"]

    def __init__(self, base_workspace_dir: str = "data/backend_workspaces/"):
        self.base_dir = Path(base_workspace_dir).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _validate_session_id(self, session_id: str) -> None:
        """Validates that a session_id is a properly formatted UUID4 string."""
        if not isinstance(session_id, str):
            raise SessionManagerError("Session ID must be a string.")
        try:
            val = uuid.UUID(session_id, version=4)
            if str(val) != session_id:
                raise ValueError()
        except ValueError:
            raise SessionManagerError(f"Invalid session ID format: {session_id}")

    def get_session_workspace(self, session_id: str) -> Path:
        """
        Returns the resolved, secure workspace Path for a session.
        Prevents path traversal attacks.
        """
        self._validate_session_id(session_id)
        
        # Resolve the session directory
        session_dir = (self.base_dir / session_id).resolve()
        
        # Ensure the resolved path remains strictly within the base_dir
        if not str(session_dir).startswith(str(self.base_dir)):
            raise SessionManagerError("Path traversal attempt detected.")
            
        return session_dir

    def session_exists(self, session_id: str) -> bool:
        """Checks if a session workspace and its metadata file exist."""
        try:
            workspace = self.get_session_workspace(session_id)
            meta_file = workspace / "metadata" / "session_info.json"
            return workspace.is_dir() and meta_file.is_file()
        except SessionManagerError:
            return False

    def create_session(self, metadata: Dict[str, Any] = None) -> str:
        """Creates a new session, its directory structure, and initializes metadata."""
        session_id = str(uuid.uuid4())
        workspace = self.get_session_workspace(session_id)
        
        # In the very rare case of UUID collision
        if workspace.exists():
            raise SessionManagerError(f"Workspace already exists for {session_id}")
            
        # Create directories
        for d in self.REQUIRED_DIRS:
            (workspace / d).mkdir(parents=True, exist_ok=True)
            
        now = datetime.now(timezone.utc).isoformat()
        
        session_data = {
            "session_id": session_id,
            "created_at": now,
            "updated_at": now,
            "status": "created",
            "reconstruction_mode": None,
            "input_metadata": metadata or {},
            "output_metadata": {},
            "error": None
        }
        
        self._write_metadata(session_id, session_data)
        return session_id

    def get_session(self, session_id: str) -> Dict[str, Any]:
        """Loads and returns the persisted metadata for a session."""
        if not self.session_exists(session_id):
            raise SessionManagerError(f"Session {session_id} does not exist.")
            
        workspace = self.get_session_workspace(session_id)
        meta_file = workspace / "metadata" / "session_info.json"
        
        with open(meta_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def update_metadata(self, session_id: str, new_metadata: Dict[str, Any]) -> None:
        """Updates the session metadata and bumps the updated_at timestamp."""
        current_data = self.get_session(session_id)
        
        # Prevent overriding fixed core fields accidentally unless done carefully
        current_data.update(new_metadata)
        
        # Enforce stable ID and created_at (just in case they were in new_metadata)
        current_data["session_id"] = session_id
        # We assume created_at remains stable; we just overwrite updated_at
        current_data["updated_at"] = datetime.now(timezone.utc).isoformat()
        
        self._write_metadata(session_id, current_data)

    def _write_metadata(self, session_id: str, data: Dict[str, Any]) -> None:
        """Safely writes metadata via atomic replacement."""
        workspace = self.get_session_workspace(session_id)
        meta_file = workspace / "metadata" / "session_info.json"
        temp_file = workspace / "metadata" / "session_info.json.tmp"
        
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
            
        # Atomic replace
        temp_file.replace(meta_file)
