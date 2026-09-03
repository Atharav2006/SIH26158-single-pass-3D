# Step 5: Backend REST API Layer

## Overview

The REST API layer exposes the underlying Backend Managers (SessionManager, InputManager, JobManager) through a stable HTTP interface using FastAPI. This bridges the frontend to the backend execution lifecycle.

**Status:** IMPLEMENTED and VERIFIED

## Design Principles

1. **Manager Independence**: The API delegates all complex file, lifecycle, and validation logic to the underlying managers. It does not perform directory operations or state transitions itself.
2. **Safe Exception Mapping**: Internal exceptions from managers (like path traversal attempts, missing sessions, state conflicts) are explicitly mapped to HTTP status codes via `map_exception_to_http`. Stack traces and absolute paths are never leaked to the client.
3. **Dependency Injection**: Managers are provided via FastAPI `Depends()`, enabling easy mocking during tests and singletons at runtime.

## Endpoints

### Sessions
- **POST `/sessions`**: Creates a new isolated backend session workspace.
  - Returns: `SessionResponse` containing `session_id`, `created_at`, `status`.
- **GET `/sessions/{session_id}`**: Retrieves metadata for a given session.
  - Returns: 200 OK with session data, or 404 Not Found.

### Inputs
- **POST `/sessions/{session_id}/inputs`**: Uploads a file (multipart) to the session's `inputs/` directory.
  - Includes validation of content size, file extension, and path security.
  - Returns: `stored_filename` and file metadata.
- **GET `/sessions/{session_id}/inputs`**: Lists all inputs for a session.
  - Returns: List of input dictionaries.
- **GET `/sessions/{session_id}/inputs/{input_id}`**: Retrieves/downloads a specific file.
  - Returns: `FileResponse` with correct content disposition.
- **DELETE `/sessions/{session_id}/inputs/{input_id}`**: Deletes an input file securely.

### Jobs
- **POST `/sessions/{session_id}/jobs`**: Creates a new reconstruction job for the session.
  - Request body optional: `reconstruction_mode`.
  - Returns: 201 Created with job data including `job_id`.
- **GET `/jobs/{job_id}`**: Retrieves job metadata (without needing the session ID).
- **POST `/jobs/{job_id}/status`**: Internal/Development endpoint to manually transition job states. 
  - Validates valid transitions (e.g., queued -> processing -> completed).
  - Will be replaced or restricted when real workers are integrated.

## Testing

18 comprehensive tests in `tests/backend/test_api.py` verify:
- End-to-end endpoint success paths (creating sessions, uploading files, creating jobs).
- Status code mappings (400 for bad paths, 404 for missing resources, 409 for conflict transitions, 413 for oversized payloads).
- Cross-session isolation.
- Dependency overrides via `TestClient`.

All tests pass successfully.
