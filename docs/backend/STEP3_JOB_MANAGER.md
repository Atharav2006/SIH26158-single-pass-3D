# STEP 3: Backend Job Manager

## Purpose
The `BackendJobManager` implements the backend job/lifecycle layer. It tracks reconstruction jobs associated with sessions provisioned by the `BackendSessionManager`, maintaining metadata, strict states, and progression metrics for API-driven reconstruction runs.

## Architecture
```
BackendSessionManager (Manages workspaces & session auth bounds)
        ↓
BackendJobManager (Manages async job state machine within a session)
        ↓
future REST/API layer (Triggers jobs)
        ↓
reconstruction engine (Executes jobs)
```
The Job Manager does **not** duplicate the reconstruction engine's internal state. It solely manages the outer boundary of the job lifecycle.

## Job Lifecycle and State Transition Rules
The `BackendJobManager` enforces a strict lifecycle state machine:
- `queued`: The initial state when `create_job()` is called. Valid transitions: `processing`, `failed`.
- `processing`: The job is actively being processed by the future worker. Valid transitions: `completed`, `failed`.
- `completed`: Terminal state representing successful completion. No outgoing transitions.
- `failed`: Terminal state representing failure. Requires an `error` message string. No outgoing transitions.

## Job Metadata
Stored per job in its owning session workspace:
```json
{
    "job_id": "uuid4",
    "session_id": "uuid4",
    "status": "queued",
    "reconstruction_mode": null,
    "created_at": "ISO-8601 UTC timestamp",
    "updated_at": "ISO-8601 UTC timestamp",
    "started_at": null,
    "completed_at": null,
    "error": null,
    "result_metadata": {}
}
```

## Persistence
For Step 3, persistence uses JSON serialized synchronously to the file system.
- Jobs are persisted at `<session_workspace>/metadata/jobs/<job_id>.json`.
- There is NO global shared job state file, ensuring complete session isolation.
- Writes use atomic replacement (`.tmp` file swapping) to prevent corruption.

## Session/Job Relationship
Every job belongs strictly to one session. The relationship is resolved by searching the configured backend workspace directories for the `<job_id>.json`. This eliminates the need for a global directory index while preserving absolute isolation.

## Isolation Rules
- A job can never read or write to a session directory other than its owning session.
- Arbitrary paths cannot be supplied by callers; `job_id` is strictly validated as a `UUID4`.
- A caller providing a valid `job_id` implicitly resolves their own session, preventing traversal or leaking.

## Error Handling
The module defines `JobManagerError`, used for:
- Invalid or malformed job IDs.
- Missing jobs or orphaned sessions.
- Invalid status string values.
- Invalid lifecycle state transitions (e.g., `completed` -> `processing`).
- File I/O or JSON persistence failures.

## Tests
Lightweight, isolated testing is available via `tests/backend/test_job_manager.py`. It tests:
- Job UUID generation and validation.
- Initial state creation and timestamps.
- Exact lifecycle transitions.
- Invalid transition rejections.
- Complete separation of jobs between independent sessions.

## Future Worker Integration
When a distributed task queue (e.g., Celery) is introduced, the worker will use the `update_job_status` API to report progress, preventing the need to access the database directly from the ML execution nodes.

## Future REST API Integration
The API will use `create_job` upon receiving a `POST /jobs` request and use `get_job` for `GET /jobs/{id}` polling.

## Known Limitations
1. **Polling Overhead**: Resolving the owning session requires scanning the `metadata/jobs` directories of all existing sessions. While fine for local deployments or modest scales, highly scaled deployments will eventually require a relational database (PostgreSQL) index map for `O(1)` job resolution.
2. **Reconstruction Execution Not Included**: The Job Manager currently tracks states but does *not* actually spawn the GPU reconstruction pipeline. That integration requires subsequent implementation steps.
