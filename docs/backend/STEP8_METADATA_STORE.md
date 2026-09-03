# Step 8: Persistent Metadata / Database Layer

## Objective
Introduce a clean persistent metadata layer for backend sessions and jobs using SQLite, allowing the higher-level backend managers (SessionManager, JobManager) to decouple from file-based JSON persistence.

## Architecture

The MetadataStore acts as a centralized SQLite database (`data/backend_metadata/backend.sqlite3`).
It is initialized on backend startup via the FastAPI app lifespan in `api.py`.
The schema enforces foreign key constraints linking `jobs` to their parent `sessions`.
It operates in Write-Ahead Logging (WAL) mode to improve concurrency, with a global thread lock wrapping database access to ensure atomic application-level transactions and avoid locking exceptions from SQLite.

### Key Components

- **`src/backend/metadata_store.py`**: Contains the `MetadataStore` class, managing direct SQLite operations for both sessions and jobs.
- **`src/backend/session_manager.py`**: Refactored to delegate persistence (creation, retrieval, and updating of session metadata) to the `MetadataStore` instead of individual `metadata/session_info.json` files.
- **`src/backend/job_manager.py`**: Refactored to delegate job metadata operations to the `MetadataStore` instead of individual JSON files within a session's `jobs` folder.

## Migration Note

**Existing JSON metadata is NOT migrated into SQLite.** 
Existing workspace JSON files are considered orphaned for this development phase. The SQLite database serves as the authoritative metadata source for all newly created sessions and jobs moving forward. File artifacts (uploads, temporary files, reconstruction binaries, etc.) remain in the filesystem workspace.

## Error Handling

- **Operational Errors**: `MetadataStore.initialize` handles missing directories and initializes properly.
- **Integrity Errors**: Foreign key constraints enforce relational integrity. Creating a job with a non-existent session ID raises an integrity constraint violation, triggering a `MetadataStoreError`.
- **Database Threading Safety**: A threading lock guards operations, allowing safe sharing across FastAPI background threads and reconstruction worker threads.

## Testing

A 24-item test matrix for `MetadataStore` is implemented in `tests/backend/test_metadata_store.py` to ensure rigorous validation of SQLite lifecycle behavior, constraints, cascading, error isolation, and concurrency. 110 total backend tests currently pass.
