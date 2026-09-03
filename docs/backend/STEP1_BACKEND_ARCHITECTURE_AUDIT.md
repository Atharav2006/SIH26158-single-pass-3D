# Backend Architecture Audit

## 1. Repository Structure
The repository is structured as a modular monolith focused on ML and geometric pipelines:
- `pipelines/application`: Contains application entry points (e.g., `reconstruct_video.py`).
- `pipelines/production`: Contains production configuration stubs.
- `src/reconstruction/`: Core engine logic defining interfaces for backend processors, mode selection, session handling, and structured results.
- `src/ingestion/`: Handles input video parsing and sensor detection.
- `scripts/`: Holds demo and verification scripts (e.g., `run_sample_demo.py`).
- `tests/`: Separated into `unit` and `integration` tests.

## 2. Existing Reconstruction Entry Point
The current entry point for triggering a reconstruction is the CLI script `pipelines/application/reconstruct_video.py`. It accepts a video file and optional sensor data as command-line arguments.

## 3. Input Flow
Input handling is formalized via the `VideoInputSpec` dataclass in `src/reconstruction/input_spec.py`.
It aggregates the following:
- Video input (`video_path`)
- Optional sensors: `gps_path`, `imu_path`, `rtk_path`
- Existing references: `poses_path`, `calibration_path`

## 4. Reconstruction Execution Flow
The `reconstruct_video.py` script orchestrates the following phases:
1. **Session Creation**: `ReconstructionSession` is instantiated with a session ID and workspace directory.
2. **Validation**: The `VideoInputSpec` ensures file existence, and `VideoValidator` performs deeper format checks.
3. **Sensor Detection**: `SensorDetector` inspects optional sensor paths.
4. **Auto-Pose/Calibration**: If poses are missing, `ColmapPoseProvider` estimates them. If calibration is missing, `ColmapCalibrationProvider` runs.
5. **Mode Selection**: `ModeSelector.evaluate()` determines if the pipeline can run metric reconstruction (if RTK is present) or falls back to relative reconstruction.
6. **Backend Execution**: Depending on the mode, either `MetricDepthBackend` or `RelativeDepthBackend` (from `src/reconstruction/reconstruction_backend.py`) is invoked to produce geometry.

## 5. Output / Result Flow
Outputs are written exclusively to the isolated session directory provided by `ReconstructionSession`:
- If blocked, structured JSON is written to `diagnostics/status.json`.
- Geometry point clouds are written to `geometry/pointcloud.ply` or `geometry/pointcloud_metric.ply` by the backend processors.
- A final success block is written to `exports/reconstruction_summary.json`.
- Internal state is modeled by `ReconstructionResult` (`src/reconstruction/reconstruction_result.py`).

## 6. Existing Session / Workspace Handling
Workspace isolation is handled by `ReconstructionSession` in `src/reconstruction/session.py`. 
- It isolates every execution into a `session_id` subfolder inside the root workspace.
- It proactively creates standard subdirectories: `inputs`, `frames`, `poses`, `depth`, `geometry`, `diagnostics`, `calibration`, `metadata`, `exports`.
- Files should be referenced via `session.get_path(relative_path)` to ensure they stay confined to the session boundary.

## 7. Existing API / Server Components
There are **no** existing REST API, gRPC, WebSocket, or server components. The system operates entirely as a local CLI process.

## 8. Existing Persistence
There is **no** database (e.g., PostgreSQL, SQLite) or persistent job state tracking. Persistence is strictly file-based within the session directory structure.

## 9. Existing Validation / Security
- File existence checking during `VideoInputSpec` initialization.
- Strict contract enforcement for metric modes (e.g., `scale_type` matching `anchor_source`) defined in `ReconstructionResult.__post_init__`.
- File path isolation within the `ReconstructionSession` base directory.

## 10. Tests Relevant to Backend Integration
The `tests/` directory contains `unit` and `integration` test suites. When running `python -m pytest tests/`, 17 errors were raised during test collection. This was caused by missing ML dependencies (`torch`) and a `ValueError` related to mutable defaults (`numpy.ndarray`) in `src/sensor_fusion/sensor_factors.py`. 
For backend integration, we will need to write isolated API tests that mock these ML dependencies to avoid heavy environment requirements.

## 11. Recommended Backend Integration Boundary
The safest integration boundary is to build an asynchronous job worker (e.g., Celery) that wraps `pipelines/application/reconstruct_video.py` as a subprocess or directly calls its Python phases. The future API service will map HTTP multipart forms into the `ReconstructionSession` directory structure, construct a `VideoInputSpec`, execute the pipeline, and poll `diagnostics/status.json` or `exports/reconstruction_summary.json` for job status.

## 12. Files Member 4 Will Need to Add
- API server entry point (e.g., `src/api/server.py` or `backend/app.py`).
- Routes/Controllers for job submission, status polling, and result fetching.
- Job management/queue configuration (e.g., Redis, Celery).
- Database ORM / schema files for Jobs and Users.
- Docker configuration for the API layer.

## 13. Files Member 4 Must NOT Modify Unless Necessary
- Core Engine Logic: `src/reconstruction/*`
- Heavy ML components: `src/depth_fusion/`, `src/neural_reconstruction/`
- Metric and geodesy math: `src/metrics/`, `src/geodesy/`

## 14. Risks / Unknowns
- **Concurrency**: While file paths are isolated by `ReconstructionSession`, we don't know if the ML backends (or Colmap) lock global resources like GPUs. Running parallel jobs may crash the host.
- **Cancellation**: There are no hooks to gracefully abort a running reconstruction.
- **Cleanup**: There is no garbage collection for old sessions, which will quickly fill the disk.

## Step 1 Conclusion
- **Current backend status**: Non-existent. The system is currently a standalone CLI application.
- **Current reconstruction entry point**: `pipelines/application/reconstruct_video.py`.
- **Recommended integration boundary**: An async task queue orchestrating the `VideoInputSpec` contract and monitoring isolated `ReconstructionSession` file outputs.
- **Whether the repository is ready for Step 2**: Yes, the session isolation and explicit input/output contracts make it highly suitable for an API wrapper.
- **Any blockers**: None for implementing the API surface, but testing end-to-end integration will require fixing local dependency issues (like `torch` installation).
