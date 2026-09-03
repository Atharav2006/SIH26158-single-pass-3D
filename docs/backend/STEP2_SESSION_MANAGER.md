# STEP 2: Backend Session Manager

## Purpose
The `BackendSessionManager` provides the API layer with a secure, isolated workspace for staging data before invoking the core reconstruction engine. It serves as the foundation for multi-tenant job handling, allowing the backend to track, persist, and isolate jobs.

## Architecture
The `BackendSessionManager` sits conceptually "above" the core engine's `ReconstructionSession`. While `ReconstructionSession` manages the internal state of a reconstruction run, `BackendSessionManager` manages the broader lifecycle of an API request, including uploads, downloads, and persistent tracking. 

## Session Lifecycle
1. **Creation**: `create_session(metadata)` is called. A UUID4 is generated, and a secure workspace is provisioned. Initial metadata is written to disk.
2. **Staging**: The API writes uploaded files (e.g., video, GPS) to the session's `inputs/` directory.
3. **Execution (Future)**: The core engine is invoked, pointing to the session's `inputs/` and `outputs/` directories.
4. **Update**: `update_metadata(session_id, new_metadata)` is called to track status, errors, and progress.
5. **Retrieval**: `get_session(session_id)` loads the persisted metadata from disk to serve API polling endpoints.

## Metadata
Metadata is persisted as JSON (`metadata/session_info.json`). It guarantees the presence of:
- `session_id` (UUID4)
- `created_at` (ISO 8601 UTC timestamp, stable)
- `updated_at` (ISO 8601 UTC timestamp, updates on modification)
- `status` (Initial: "created")
- `reconstruction_mode` (Optional, assigned later)
- `input_metadata` (Dict of client-provided inputs)
- `output_metadata` (Dict of generated outputs)
- `error` (String, if applicable)

## Workspace Structure
Each session is assigned a unique UUID4 directory within the configured base workspace (default: `data/backend_workspaces/`).
```
<session_id>/
├── inputs/    # Uploaded user files
├── temp/      # Ephemeral staging files
├── outputs/   # Final artifacts for download
├── metadata/  # Persistent JSON tracking
└── logs/      # Execution logs
```

## Isolation Rules
1. **Directory Isolation**: Every session is guaranteed a physically distinct folder in the file system, keyed by its UUID4.
2. **State Isolation**: Metadata reads/writes strictly target the session's specific `metadata/session_info.json`. A session cannot access another session's metadata.

## Path Security
- **Strict UUID4 Validation**: Session IDs are explicitly parsed using Python's `uuid.UUID` to reject malformed strings, traversal characters (`../`), or absolute paths.
- **Path Confinement**: The manager resolves the session directory (`Path.resolve()`) and asserts that it strictly descends from the `base_workspace_dir`.

## Persistence
Metadata is persisted exclusively to the filesystem using JSON. Writes are performed using an atomic replace (`temp.json` -> `session_info.json`) to prevent corruption if the API crashes mid-write. Memory is never trusted as the source of truth for `get_session`.

## Error Handling
- Invalid session IDs raise `SessionManagerError`.
- Path traversal attempts raise `SessionManagerError`.
- Missing sessions or corrupted UUID states raise `SessionManagerError`.

## Tests
Isolated unit tests are located in `tests/backend/test_session_manager.py`. They verify:
- Complete directory initialization.
- Strict UUID format enforcement.
- Cross-session directory and metadata isolation.
- Atomic timestamp updates.
- Defensive path traversal rejections.

## Future API Integration
The REST API (to be implemented in future steps) will instantiate a singleton `BackendSessionManager`. Routes like `POST /jobs` will map directly to `create_session`, while `GET /jobs/{id}` will map to `get_session`.

## Known Limitations
1. **JSON Scaling**: Concurrent writes to the JSON file are not protected by lockfiles, making it unsuitable for highly parallel sub-processes editing the *same* session metadata simultaneously. A database (SQLite/PostgreSQL) will eventually replace JSON for metadata storage.
2. **Garbage Collection**: There is currently no mechanism to clean up or delete expired sessions from the file system.
