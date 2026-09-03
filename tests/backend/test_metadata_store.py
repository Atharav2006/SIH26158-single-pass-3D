import os
import sqlite3
import threading
from datetime import datetime, timezone
import pytest

from src.backend.metadata_store import MetadataStore, MetadataStoreError

@pytest.fixture
def store(tmp_path):
    """Fixture providing a fresh in-memory MetadataStore for each test."""
    test_db_path = tmp_path / "test.sqlite3"
    # Using an actual file instead of :memory: to test concurrent access safely.
    store = MetadataStore(db_path=test_db_path)
    store.initialize()
    return store

def test_initialization(store, tmp_path):
    """1. Test table creation and PRAGMA settings."""
    assert store.db_path == tmp_path / "test.sqlite3"
    
    with store._get_connection() as conn:
        cursor = conn.cursor()
        
        # Check PRAGMAs
        cursor.execute("PRAGMA foreign_keys;")
        assert cursor.fetchone()[0] == 1
        
        cursor.execute("PRAGMA journal_mode;")
        assert cursor.fetchone()[0].lower() == "wal"
        
        # Check tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = {row[0] for row in cursor.fetchall()}
        assert "sessions" in tables
        assert "jobs" in tables

def test_session_create(store):
    """2. Test inserting a session and retrieving it."""
    session_data = {
        "session_id": "sess_123",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "input_metadata": {"test_key": "test_value"},
        "output_metadata": {}
    }
    
    store.create_session(session_data)
    
    retrieved = store.get_session("sess_123")
    assert retrieved["session_id"] == "sess_123"
    assert retrieved["input_metadata"] == {"test_key": "test_value"}
    assert retrieved["output_metadata"] == {}

def test_session_create_duplicate(store):
    """3. Test duplicate session ID rejection."""
    session_data = {
        "session_id": "sess_123",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "input_metadata": {},
        "output_metadata": {}
    }
    store.create_session(session_data)
    
    with pytest.raises(MetadataStoreError, match="UNIQUE constraint failed"):
        store.create_session(session_data)

def test_session_get_missing(store):
    """4. Test retrieving a non-existent session."""
    with pytest.raises(MetadataStoreError, match="Session missing_sess not found"):
        store.get_session("missing_sess")

def test_session_update(store):
    """5. Test updating session metadata."""
    session_data = {
        "session_id": "sess_123",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "input_metadata": {"initial": "value"},
        "output_metadata": {}
    }
    store.create_session(session_data)
    
    updates = {
        "input_metadata": {"updated": "value"},
        "output_metadata": {"job_id": "job_123"}
    }
    store.update_session("sess_123", updates)
    
    retrieved = store.get_session("sess_123")
    assert retrieved["input_metadata"] == {"updated": "value"}
    assert retrieved["output_metadata"] == {"job_id": "job_123"}

def test_session_update_missing(store):
    """6. Test updating a non-existent session."""
    with pytest.raises(MetadataStoreError, match="Session missing_sess not found"):
        store.update_session("missing_sess", {"input_metadata": {}})

def test_job_create(store):
    """7. Test inserting a job and retrieving it."""
    # First, create a session
    store.create_session({
        "session_id": "sess_123",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "input_metadata": {},
        "output_metadata": {}
    })
    
    job_data = {
        "job_id": "job_123",
        "session_id": "sess_123",
        "status": "queued",
        "reconstruction_mode": "quality",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "started_at": None,
        "completed_at": None,
        "error": None,
        "result_metadata": {}
    }
    
    store.create_job(job_data)
    
    retrieved = store.get_job("job_123")
    assert retrieved["job_id"] == "job_123"
    assert retrieved["session_id"] == "sess_123"
    assert retrieved["status"] == "queued"
    assert retrieved["reconstruction_mode"] == "quality"
    assert retrieved["result_metadata"] == {}

def test_job_create_duplicate(store):
    """8. Test duplicate job ID rejection."""
    store.create_session({
        "session_id": "sess_123",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "input_metadata": {},
        "output_metadata": {}
    })
    
    job_data = {
        "job_id": "job_123",
        "session_id": "sess_123",
        "status": "queued",
        "reconstruction_mode": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "started_at": None,
        "completed_at": None,
        "error": None,
        "result_metadata": {}
    }
    store.create_job(job_data)
    
    with pytest.raises(MetadataStoreError, match="UNIQUE constraint failed"):
        store.create_job(job_data)

def test_job_create_foreign_key(store):
    """9. Test inserting a job for a non-existent session (foreign key enforcement)."""
    job_data = {
        "job_id": "job_123",
        "session_id": "missing_sess",
        "status": "queued",
        "reconstruction_mode": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "started_at": None,
        "completed_at": None,
        "error": None,
        "result_metadata": {}
    }
    
    with pytest.raises(MetadataStoreError, match="FOREIGN KEY constraint failed"):
        store.create_job(job_data)

def test_job_get_missing(store):
    """10. Test retrieving a non-existent job."""
    with pytest.raises(MetadataStoreError, match="Job missing_job not found"):
        store.get_job("missing_job")

def test_job_update(store):
    """11. Test updating job status and metadata."""
    store.create_session({
        "session_id": "sess_123",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "input_metadata": {},
        "output_metadata": {}
    })
    
    store.create_job({
        "job_id": "job_123",
        "session_id": "sess_123",
        "status": "queued",
        "reconstruction_mode": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "started_at": None,
        "completed_at": None,
        "error": None,
        "result_metadata": {}
    })
    
    updates = {
        "status": "processing",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "result_metadata": {"progress": 50}
    }
    store.update_job("job_123", updates)
    
    retrieved = store.get_job("job_123")
    assert retrieved["status"] == "processing"
    assert retrieved["started_at"] is not None
    assert retrieved["result_metadata"] == {"progress": 50}

def test_job_update_missing(store):
    """12. Test updating a non-existent job."""
    with pytest.raises(MetadataStoreError, match="Job missing_job not found"):
        store.update_job("missing_job", {"status": "completed"})

def test_list_jobs(store):
    """13. Test retrieving all jobs for a specific session."""
    store.create_session({
        "session_id": "sess_123",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "input_metadata": {},
        "output_metadata": {}
    })
    
    store.create_job({
        "job_id": "job_1",
        "session_id": "sess_123",
        "status": "completed",
        "reconstruction_mode": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "started_at": None,
        "completed_at": None,
        "error": None,
        "result_metadata": {}
    })
    
    store.create_job({
        "job_id": "job_2",
        "session_id": "sess_123",
        "status": "failed",
        "reconstruction_mode": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "started_at": None,
        "completed_at": None,
        "error": None,
        "result_metadata": {}
    })
    
    jobs = store.list_jobs("sess_123")
    assert len(jobs) == 2
    job_ids = [j["job_id"] for j in jobs]
    assert "job_1" in job_ids
    assert "job_2" in job_ids

def test_list_jobs_for_missing_session(store):
    """14. Test listing jobs for a non-existent session (should return empty)."""
    jobs = store.list_jobs("missing_sess")
    assert jobs == []

def test_thread_safety(store):
    """15. Test thread safety of connection lock."""
    # This verifies that the lock prevents simultaneous writes from causing issues.
    store.create_session({
        "session_id": "sess_123",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "input_metadata": {},
        "output_metadata": {}
    })

    errors = []
    
    def worker(job_id):
        try:
            store.create_job({
                "job_id": job_id,
                "session_id": "sess_123",
                "status": "queued",
                "reconstruction_mode": None,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "started_at": None,
                "completed_at": None,
                "error": None,
                "result_metadata": {}
            })
        except Exception as e:
            errors.append(e)

    threads = []
    for i in range(10):
        t = threading.Thread(target=worker, args=(f"job_{i}",))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    assert not errors
    jobs = store.list_jobs("sess_123")
    assert len(jobs) == 10

def test_json_serialization_edge_cases(store):
    """16. Test JSON serialization with special characters."""
    store.create_session({
        "session_id": "sess_123",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "input_metadata": {"key!@#$%^&*()": "value\n\t\r\"'"},
        "output_metadata": {}
    })
    retrieved = store.get_session("sess_123")
    assert retrieved["input_metadata"]["key!@#$%^&*()"] == "value\n\t\r\"'"

def test_update_session_partial_json(store):
    """17. Test that partial JSON updates overwrite completely, not merge."""
    store.create_session({
        "session_id": "sess_123",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "input_metadata": {"a": 1, "b": 2},
        "output_metadata": {}
    })
    
    # We provide a completely new dict for input_metadata
    store.update_session("sess_123", {"input_metadata": {"c": 3}})
    retrieved = store.get_session("sess_123")
    assert retrieved["input_metadata"] == {"c": 3}
    assert "a" not in retrieved["input_metadata"]

def test_invalid_db_path(tmp_path):
    """18. Test initialization with invalid DB path."""
    with pytest.raises(MetadataStoreError):
        # Provide a path to a directory instead of a file
        store = MetadataStore(db_path=tmp_path)
        store.initialize()

def test_sql_injection_prevention(store):
    """19. Test safety against malicious IDs (verifying parameterized queries)."""
    malicious_id = "sess_123'; DROP TABLE sessions; --"
    store.create_session({
        "session_id": malicious_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "input_metadata": {},
        "output_metadata": {}
    })
    
    # Table should still exist
    with sqlite3.connect(store.db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = {row[0] for row in cursor.fetchall()}
        assert "sessions" in tables

    # And we should be able to retrieve it using the literal ID
    retrieved = store.get_session(malicious_id)
    assert retrieved["session_id"] == malicious_id

def test_job_create_missing_fields(store):
    """20. Test job creation with missing mandatory fields."""
    store.create_session({
        "session_id": "sess_123",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "input_metadata": {},
        "output_metadata": {}
    })
    
    job_data = {
        # missing job_id
        "session_id": "sess_123",
        "status": "queued"
    }
    with pytest.raises((KeyError, MetadataStoreError)):
        store.create_job(job_data)

def test_job_update_no_updates(store):
    """21. Test updating a job with empty updates dictionary."""
    store.create_session({
        "session_id": "sess_123",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "input_metadata": {},
        "output_metadata": {}
    })
    
    store.create_job({
        "job_id": "job_123",
        "session_id": "sess_123",
        "status": "queued",
        "reconstruction_mode": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "started_at": None,
        "completed_at": None,
        "error": None,
        "result_metadata": {}
    })
    
    store.update_job("job_123", {})
    retrieved = store.get_job("job_123")
    assert retrieved["status"] == "queued"

def test_session_create_missing_fields(store):
    """22. Test session creation with missing mandatory fields."""
    session_data = {
        # missing session_id
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with pytest.raises((KeyError, MetadataStoreError)):
        store.create_session(session_data)

def test_get_session_type(store):
    """23. Ensure retrieved JSON strings are parsed to dicts/lists."""
    store.create_session({
        "session_id": "sess_123",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "input_metadata": {"test": 123},
        "output_metadata": {"job": "test"}
    })
    
    retrieved = store.get_session("sess_123")
    assert isinstance(retrieved["input_metadata"], dict)
    assert isinstance(retrieved["output_metadata"], dict)

def test_get_job_type(store):
    """24. Ensure retrieved job JSON strings are parsed to dicts."""
    store.create_session({
        "session_id": "sess_123",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "input_metadata": {},
        "output_metadata": {}
    })
    
    store.create_job({
        "job_id": "job_123",
        "session_id": "sess_123",
        "status": "queued",
        "reconstruction_mode": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "started_at": None,
        "completed_at": None,
        "error": None,
        "result_metadata": {"progress": 100}
    })
    
    retrieved = store.get_job("job_123")
    assert isinstance(retrieved["result_metadata"], dict)
