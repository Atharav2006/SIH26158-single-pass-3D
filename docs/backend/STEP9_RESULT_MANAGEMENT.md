# Step 9: Result Management & Export API

## Overview
The `BackendResultManager` provides a secure, session-isolated layer for querying and retrieving reconstruction artifacts. It enforces strict separation of concerns, ensuring clients can only access artifacts produced by completed jobs belonging to their own session.

## Result ID Design
Result artifacts are discovered dynamically from allowed output directories (`geometry`, `diagnostics`, `exports`).
The `result_id` is a deterministically generated URL-safe logical string (e.g., `geometry_mesh_obj`). This design inherently mitigates path traversal since clients cannot supply arbitrary filesystem paths; they can only request a pre-generated ID that maps directly to a known file.

## Artifact Discovery Policy
Only artifacts residing in explicit directories are exposed:
- `geometry/`: Meshes, point clouds, etc.
- `diagnostics/`: Status and error metadata (e.g., `status.json`). Explicitly exposed to help clients understand reconstruction failure details if needed.
- `exports/`: The internal reconstruction engine's final JSON summaries.
Only **regular files** are returned. Symlinks are resolved internally and verified to not escape the workspace. Directories and missing files are excluded.

## Export Collision Policy
When exporting an artifact to the session's user-facing `outputs/` folder, the API accepts a `destination_filename`. 
If the file already exists, the API rejects the request with an **HTTP 409 Conflict**. This prevents silent overwriting and ensures deterministic behavior.

## Endpoints
- `GET /sessions/{session_id}/jobs/{job_id}/results`: Lists metadata for all available artifacts.
- `GET /sessions/{session_id}/jobs/{job_id}/results/{result_id}`: Downloads the specific artifact.
- `POST /sessions/{session_id}/jobs/{job_id}/results/{result_id}/export`: Safely copies the artifact to `outputs/` using the provided `destination_filename`.
