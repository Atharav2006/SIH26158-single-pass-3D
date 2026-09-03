import sqlite3
import json
import threading
from contextlib import contextmanager
from typing import Dict, Any, List, Optional
from pathlib import Path


class MetadataStoreError(Exception):
    pass


class MetadataStore:
    """
    A lightweight, thread-safe SQLite persistence layer for backend sessions and jobs.
    Stores metadata only. File artifacts are retained in the filesystem workspace.
    """

    def __init__(self, db_path: str = "data/backend_metadata/backend.sqlite3"):
        self.db_path = Path(db_path)
        # Ensure the directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # We use a global lock to prevent concurrent write issues in sqlite, 
        # though sqlite handles concurrency, this ensures python-level safety 
        # for our transaction sequences.
        self._lock = threading.Lock()

    @contextmanager
    def _get_connection(self):
        """Context manager for acquiring a safe SQLite connection."""
        # check_same_thread=False allows us to use connections across threads
        # but we guard mutations with our own lock if needed.
        conn = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,
            isolation_level=None  # We manage transactions explicitly
        )
        # SQLite uses PRAGMA foreign_keys = ON on a per-connection basis
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA journal_mode = WAL;")  # Better concurrency
        
        try:
            yield conn
        finally:
            conn.close()

    def initialize(self):
        """Initializes the database schema."""
        try:
            with self._lock, self._get_connection() as conn:
                with conn: # Starts a transaction
                    conn.execute("""
                        CREATE TABLE IF NOT EXISTS sessions (
                            session_id TEXT PRIMARY KEY,
                            status TEXT,
                            reconstruction_mode TEXT,
                            input_metadata TEXT,
                            output_metadata TEXT,
                            error TEXT,
                            created_at TEXT,
                            updated_at TEXT
                        )
                    """)
    
                    conn.execute("""
                        CREATE TABLE IF NOT EXISTS jobs (
                            job_id TEXT PRIMARY KEY,
                            session_id TEXT NOT NULL,
                            status TEXT,
                            reconstruction_mode TEXT,
                            error TEXT,
                            result_metadata TEXT,
                            created_at TEXT,
                            updated_at TEXT,
                            started_at TEXT,
                            completed_at TEXT,
                            FOREIGN KEY(session_id) REFERENCES sessions(session_id)
                        )
                    """)
        except Exception as e:
            raise MetadataStoreError(f"Failed to initialize database: {e}")

    def _row_to_dict(self, cursor: sqlite3.Cursor, row: tuple) -> Dict[str, Any]:
        """Converts a sqlite row into a dictionary based on column names."""
        if not row:
            return None
        return {col[0]: row[i] for i, col in enumerate(cursor.description)}

    # ------------------------------------------------------------------------
    # SESSIONS
    # ------------------------------------------------------------------------

    def create_session(self, session_data: Dict[str, Any]) -> None:
        """Inserts a new session into the database."""
        try:
            with self._lock, self._get_connection() as conn:
                with conn:
                    conn.execute("""
                        INSERT INTO sessions (
                            session_id, status, reconstruction_mode,
                            input_metadata, output_metadata, error,
                            created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        session_data["session_id"],
                        session_data.get("status"),
                        session_data.get("reconstruction_mode"),
                        json.dumps(session_data.get("input_metadata", {})),
                        json.dumps(session_data.get("output_metadata", {})),
                        session_data.get("error"),
                        session_data.get("created_at"),
                        session_data.get("updated_at")
                    ))
        except sqlite3.IntegrityError as e:
            raise MetadataStoreError(f"Failed to create session: {e}")
        except Exception as e:
            raise MetadataStoreError(f"Database error: {e}")

    def get_session(self, session_id: str) -> Dict[str, Any]:
        """Retrieves a session from the database."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
            )
            row = cursor.fetchone()
            if not row:
                raise MetadataStoreError(f"Session {session_id} not found.")
            
            data = self._row_to_dict(cursor, row)
            # Deserialize JSON fields
            data["input_metadata"] = json.loads(data["input_metadata"]) if data["input_metadata"] else {}
            data["output_metadata"] = json.loads(data["output_metadata"]) if data["output_metadata"] else {}
            return data

    def update_session(self, session_id: str, updates: Dict[str, Any]) -> None:
        """
        Updates specific fields of a session.
        Keys in `updates` must match column names.
        """
        # Exclude session_id from updates if present
        updates = {k: v for k, v in updates.items() if k != "session_id"}
        if not updates:
            return

        # Serialize JSON fields if they are in updates
        for json_field in ["input_metadata", "output_metadata"]:
            if json_field in updates:
                updates[json_field] = json.dumps(updates[json_field])

        set_clause = ", ".join([f"{k} = ?" for k in updates.keys()])
        values = list(updates.values())
        values.append(session_id)

        try:
            with self._lock, self._get_connection() as conn:
                with conn:
                    cursor = conn.execute(
                        f"UPDATE sessions SET {set_clause} WHERE session_id = ?",
                        values
                    )
                    if cursor.rowcount == 0:
                        raise MetadataStoreError(f"Session {session_id} not found.")
        except Exception as e:
            raise MetadataStoreError(f"Failed to update session: {e}")

    def delete_session(self, session_id: str) -> None:
        """Deletes a session and its associated jobs."""
        try:
            with self._lock, self._get_connection() as conn:
                with conn:
                    # Foreign keys with PRAGMA foreign_keys = ON will prevent deletion
                    # if there are associated jobs, UNLESS we delete them first or cascade.
                    # We will explicitly delete jobs first to be safe.
                    conn.execute("DELETE FROM jobs WHERE session_id = ?", (session_id,))
                    conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        except Exception as e:
            raise MetadataStoreError(f"Failed to delete session: {e}")

    # ------------------------------------------------------------------------
    # JOBS
    # ------------------------------------------------------------------------

    def create_job(self, job_data: Dict[str, Any]) -> None:
        """Inserts a new job into the database."""
        try:
            with self._lock, self._get_connection() as conn:
                with conn:
                    conn.execute("""
                        INSERT INTO jobs (
                            job_id, session_id, status, reconstruction_mode,
                            error, result_metadata, created_at, updated_at,
                            started_at, completed_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        job_data["job_id"],
                        job_data["session_id"],
                        job_data.get("status"),
                        job_data.get("reconstruction_mode"),
                        job_data.get("error"),
                        json.dumps(job_data.get("result_metadata", {})),
                        job_data.get("created_at"),
                        job_data.get("updated_at"),
                        job_data.get("started_at"),
                        job_data.get("completed_at")
                    ))
        except sqlite3.IntegrityError as e:
            raise MetadataStoreError(f"Failed to create job: {e}")
        except Exception as e:
            raise MetadataStoreError(f"Database error: {e}")

    def get_job(self, job_id: str) -> Dict[str, Any]:
        """Retrieves a job from the database."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?", (job_id,)
            )
            row = cursor.fetchone()
            if not row:
                raise MetadataStoreError(f"Job {job_id} not found.")
            
            data = self._row_to_dict(cursor, row)
            # Deserialize JSON
            data["result_metadata"] = json.loads(data["result_metadata"]) if data["result_metadata"] else {}
            return data

    def update_job(self, job_id: str, updates: Dict[str, Any]) -> None:
        """
        Updates specific fields of a job.
        Keys in `updates` must match column names.
        """
        # Exclude job_id from updates if present
        updates = {k: v for k, v in updates.items() if k != "job_id" and k != "session_id"}
        if not updates:
            return

        if "result_metadata" in updates:
            updates["result_metadata"] = json.dumps(updates["result_metadata"])

        set_clause = ", ".join([f"{k} = ?" for k in updates.keys()])
        values = list(updates.values())
        values.append(job_id)

        try:
            with self._lock, self._get_connection() as conn:
                with conn:
                    cursor = conn.execute(
                        f"UPDATE jobs SET {set_clause} WHERE job_id = ?",
                        values
                    )
                    if cursor.rowcount == 0:
                        raise MetadataStoreError(f"Job {job_id} not found.")
        except Exception as e:
            raise MetadataStoreError(f"Failed to update job: {e}")

    def list_jobs(self, session_id: str) -> List[Dict[str, Any]]:
        """Lists all jobs for a specific session."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM jobs WHERE session_id = ? ORDER BY created_at ASC", (session_id,)
            )
            rows = cursor.fetchall()
            jobs = []
            for row in rows:
                data = self._row_to_dict(cursor, row)
                data["result_metadata"] = json.loads(data["result_metadata"]) if data["result_metadata"] else {}
                jobs.append(data)
            return jobs
