# B6 Generalized Video-to-3D Reconstruction Engine

This plan implements the architectural generalization of the B5 relative reconstruction system into a generalized, session-isolated Video-to-3D Engine.

## User Review Required
The following plan explicitly follows the user's provided phases to abstract the pipeline. No reconstruction algorithms (like 3DGS or Mesh) will be added. 

## Open Questions
- Is there any specific mock behavior expected for `PoseProvider` and `CalibrationProvider` when dealing with new, uncalibrated arbitrary videos during the smoke test, aside from gracefully failing or reporting "NOT_AVAILABLE"? (Assuming they report unavailable and mode selection handles it).

## Proposed Changes

### Configuration
#### [NEW] [reconstruction_default.yaml](file:///d:/SIH26158-single-pass-3D/configs/reconstruction_default.yaml)
Configuration defining hardware limits, frame extraction rates, depth model config, and confidence thresholds.

### Core Contracts & Session
#### [NEW] [input_spec.py](file:///d:/SIH26158-single-pass-3D/src/reconstruction/input_spec.py)
Defines the `VideoInputSpec` requiring a video path, and optionally accepting GPS, IMU, RTK, and calibration paths.

#### [NEW] [video_session.py](file:///d:/SIH26158-single-pass-3D/src/ingestion/video_session.py)
Implements video validation, frame extraction configuration, and metadata extraction (FPS, duration, resolution).

#### [MODIFY] [session.py](file:///d:/SIH26158-single-pass-3D/src/reconstruction/session.py)
Extends `ReconstructionSession` to include `exports` and `metadata` directories.

#### [NEW] [optional_sensors.py](file:///d:/SIH26158-single-pass-3D/src/ingestion/optional_sensors.py)
Adapters for GPS, IMU, and RTK files to detect presence and classify as `AVAILABLE`, `NOT_AVAILABLE`, `INVALID`.

### Providers & Selectors
#### [NEW] [mode_selector.py](file:///d:/SIH26158-single-pass-3D/src/reconstruction/mode_selector.py)
Evaluates the session inputs to decide between `RELATIVE_RECONSTRUCTION` and `METRIC_RECONSTRUCTION`.

#### [NEW] [pose_provider.py](file:///d:/SIH26158-single-pass-3D/src/reconstruction/pose_provider.py)
Abstracts pose sources (`PRECOMPUTED_B2`, `COLMAP`, etc.).

#### [NEW] [calibration_provider.py](file:///d:/SIH26158-single-pass-3D/src/reconstruction/calibration_provider.py)
Abstracts camera calibration sources (`FULL_OPENCV`, `SUPPLIED_INTRINSICS`).

### Backends
#### [NEW] [reconstruction_backend.py](file:///d:/SIH26158-single-pass-3D/src/reconstruction/reconstruction_backend.py)
Defines the `ReconstructionBackend` interface (`prepare`, `estimate_pose`, `estimate_depth`, `reconstruct_geometry`, `evaluate`, `export`).
Implements `RelativeDepthBackend` wrapping the B5 fusion logic.

### Applications & Tests
#### [NEW] [reconstruct_video.py](file:///d:/SIH26158-single-pass-3D/pipelines/application/reconstruct_video.py)
The primary CLI entry point processing the video -> session -> mode -> backend pipeline.

#### [NEW] Unit & Integration Tests
Adds test files under `tests/unit/` and `tests/integration/` mapping exactly to Phase 17 requirements.

#### [NEW] [step22_b6_general_video_to_3d_engine.md](file:///d:/SIH26158-single-pass-3D/docs/step22_b6_general_video_to_3d_engine.md)
Documentation of the new production engine vs benchmark pipeline.

## Verification Plan

### Automated Tests
- Run `pytest -q` on all new contracts, isolation layers, and mock providers.

### Manual Verification
1. **New-Video Smoke Test**: Run the CLI on a generated synthetic video to prove session creation and mode selection work for arbitrary video.
2. **Zurich Regression**: Run the CLI on the Zurich dataset using the `PRECOMPUTED_B2` pose provider and verify the pipeline outputs a valid `RELATIVE_RECONSTRUCTION` matching B5 baselines without modifying historical outputs.
