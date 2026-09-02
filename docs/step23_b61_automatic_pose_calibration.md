# SIH26158 B6.1: Automatic Pose & Calibration Backends

## Objective
Enable a completely uncalibrated, arbitrary video sequence to proceed through the relative reconstruction pipeline by providing automatic camera calibration and pose estimation. 

## Architectural Implementations

### 1. `ColmapPoseProvider`
* Implements the `PoseProvider` interface by wrapping the existing, robust `COLMAPRunner` originally developed for B0.
* Enforces strict session isolation by creating and writing to a local `session/colmap` workspace. This guarantees no collision with the global workspace or any historic Zurich results.
* Standardizes all pose outputs from COLMAP's World-to-Camera convention to the required Camera-to-World ($X_w = R_{wc} X_c + C_w$) utilizing quaternion inversion.
* Exposes failure modes with actionable error reasons (`INSUFFICIENT_FEATURES`, `POSE_QUALITY_LOW`).

### 2. `ColmapCalibrationProvider`
* Extracts and converts intrinsic variables (`fx`, `fy`, `cx`, `cy`) directly from the reconstructed COLMAP sparse state when calibration isn't supplied by the user.
* Evaluates intrinsic plausibility (checking for pathological focal lengths or principal point offsets) and flags estimations with `CALIBRATION_UNCERTAIN` when necessary, preventing wild calibrations from silently breaking depth fusion.

### 3. Graceful Pipeline Orchestration
The generalized CLI (`pipelines/application/reconstruct_video.py`) now dynamically chains dependencies:
`Upload -> Validate -> Attempt Auto-Calibration -> Attempt Auto-Pose -> Select Mode -> Fusion`.
If any provider rejects the data quality, it returns a structured `RECONSTRUCTION_BLOCKED` output rather than crashing or inventing dummy geometry.

## Validation
A heavily textured synthetic test case (`synthetic_texture_video.py`) ensured COLMAP effectively recognized feature movement across a generic sequence. The test successfully routed the video to `RELATIVE_RECONSTRUCTION_READY` utilizing purely estimated poses and intrinsics. 

All 210 historical and modern regression tests passed cleanly.
