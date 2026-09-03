import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any

from src.backend.metadata_store import MetadataStore, MetadataStoreError

class SessionManagerError(Exception):
    pass

class BackendSessionManager:
    """Manages secure, isolated backend reconstruction sessions."""
    
    REQUIRED_DIRS = ["inputs", "temp", "outputs", "logs"]

    def __init__(self, metadata_store: MetadataStore, base_workspace_dir: str = "data/backend_workspaces/"):
        self.store = metadata_store
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
        """Checks if a session exists in the database and workspace."""
        try:
            # If get_session succeeds, it exists in DB
            self.store.get_session(session_id)
            # We also check the workspace
            workspace = self.get_session_workspace(session_id)
            return workspace.is_dir()
        except (MetadataStoreError, SessionManagerError):
            return False

    def create_session(self, metadata: Dict[str, Any] = None) -> str:
        """Creates a new session, its directory structure, and initializes metadata in DB."""
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
        
        try:
            self.store.create_session(session_data)
        except MetadataStoreError as e:
            raise SessionManagerError(str(e))
            
        return session_id

    def get_session(self, session_id: str) -> Dict[str, Any]:
        """Loads and returns the persisted metadata for a session."""
        try:
            return self.store.get_session(session_id)
        except MetadataStoreError as e:
            raise SessionManagerError(str(e))

    def update_metadata(self, session_id: str, new_metadata: Dict[str, Any]) -> None:
        """Updates the session metadata and bumps the updated_at timestamp."""
        # Ensure session exists first
        self.get_session(session_id)
        
        updates = new_metadata.copy()
        updates["updated_at"] = datetime.now(timezone.utc).isoformat()
        
        try:
            self.store.update_session(session_id, updates)
        except MetadataStoreError as e:
            raise SessionManagerError(str(e))
