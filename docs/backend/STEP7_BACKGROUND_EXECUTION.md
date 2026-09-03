# Step 7: Background Execution Manager

## Purpose
The `BackgroundExecutionManager` introduces asynchronous, non-blocking execution of reconstruction jobs using a simple local-process thread pool. This allows the REST API to accept job submissions, queue them, and return an HTTP response immediately without keeping the connection open for the potentially hours-long reconstruction process.

## Architecture
The system integrates natively with the existing components:
```
Frontend -> REST API (`api.py`)
              |
              v (submit job)
        BackgroundExecutionManager (Thread Pool)
              |
              v (execute on background thread)
        BackendReconstructionWorker
              |
              v (reports status updates)
        BackendJobManager
```

## Lifecycle Behavior
1. **Creation**: Job is created in `queued` state.
2. **Submission**: API calls `submit()`. Manager validates state, session ownership, and idempotency, then queues the job onto the thread pool. Returns immediately.
3. **Execution Start**: When a thread becomes available, the `ReconstructionWorker` begins execution. It updates the state to `processing` in the `JobManager`.
4. **Completion/Failure**: The worker completes (or catches a failure) and updates the job to `completed` or `failed`.

## Safeguards
- **Duplicate Prevention**: Re-submitting an already running or completed/failed job is safely rejected with a `400 Bad Request`.
- **Session Isolation**: Attempting to submit a job using the wrong `session_id` is rejected.
- **Error Shielding**: If the worker encounters an unhandled fatal error (or Python segfault simulation), the execution manager intercepts the exception, scrubs absolute workspace paths (`<WORKSPACE>`), and marks the job as `failed` before raising.

## Limitations (Deferred Architecture)
- **Local Process Only**: This uses Python's standard `concurrent.futures.ThreadPoolExecutor`. If the API server process is killed or restarts, any queued futures are lost. The JobManager's state remains durable on disk, but the queue itself does not auto-resume.
- **Not Distributed**: This does not yet use external message queues (like Celery/Redis) or multiple worker nodes.
- **Polling**: Frontends must poll the `/jobs/{job_id}` endpoint to observe state changes (`queued -> processing -> completed`). WebSockets are deferred.
- **Shutdown**: A basic shutdown hook is attached to FastAPI's lifespan, which attempts to cleanly terminate the executor on shutdown, but abruptly killed processes will still drop active runs.
