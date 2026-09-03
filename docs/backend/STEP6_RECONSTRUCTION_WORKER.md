# STEP 6: Reconstruction Worker Integration

## Purpose
The `BackendReconstructionWorker` abstracts away the actual invocation of the reconstruction pipeline (`pipelines.application.reconstruct_video`). It connects the `BackendJobManager` lifecycle with the reconstruction algorithms securely and efficiently without adding complex background queues immediately.

## Architecture
- The Worker does not expose REST APIs directly and is not tightly coupled to FastAPI's request lifecycle.
- It is invoked synchronously via `worker.run_job(job_id)`.
- **Existing Reconstruction Entry Point:** Uses the `reconstruct_video(args)` function located in `pipelines/application/reconstruct_video.py`. This avoids subprocess overhead and relies strictly on Python-level isolation since the function reads/writes only to the provided session output directory.

## Job Lifecycle
- **queued -> processing**: Triggered immediately upon execution.
- **processing -> completed**: Handled if the pipeline returns a non-error status.
- **processing -> failed**: Handled gracefully if the pipeline returns `"RECONSTRUCTION_BLOCKED"` or raises any uncaught exception.

## Input Discovery & Mapping
The worker extracts existing inputs via `BackendInputManager.list_inputs`. It automatically maps them to pipeline arguments (e.g., `video`, `gps`, `imu`, `calibration`, `poses`, `rtk`) based on the tracked metadata (`input_type` or file extensions/names). No custom formats are enforced.

## Output Isolation
Outputs are bound precisely to `workspace_dir` derived from `BackendSessionManager.get_session_workspace(session_id)`. The reconstruction pipeline natively takes `--output` which dictates where its session directories, logs, and `exports/` go, completely preventing interference across sessions. 

## Duplicate Execution & Timeout
- **Duplicate Prevention**: `run_job` explicitly rejects jobs that are not in `"queued"` status. Completed, processing, or failed jobs cannot be inadvertently rerun.
- **Timeouts**: Timeouts are deferred to a potential background execution layer (e.g. Celery or ProcessPool) in future steps. Direct synchronous execution blocks until completion.

## Failure Handling
Any unexpected exception or explicit blocked status produces a `"failed"` job. 
- Absolute filesystem paths are automatically scrubbed from exception strings using a reliable string-replacement mechanism (`<WORKSPACE>`) to ensure they don't leak through APIs later.

## Test Strategy
- The reconstruction pipeline execution boundary (`reconstruct_video`) is mocked.
- Verifies state transitions (queued -> processing -> completed/failed).
- Validates exception handling and safe error scrubbing.
- Confirms input/output isolation (Job A does not impact Session B).
- **Current test count:** 6 tests specifically for the worker, bringing total backend tests to 66.

## Explicit Limitations
- The worker executes synchronously. It is NOT yet a background queue, distributed worker system, or multiprocess dispatcher. This step establishes the safe execution contract *first* so that subsequent integration with tools like Celery or simply `BackgroundTasks` is trivial and robust.
