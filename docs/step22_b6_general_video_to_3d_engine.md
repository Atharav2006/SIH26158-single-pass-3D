# SIH26158 B6: Generalized Video-to-3D Engine

## 1. Product Architecture Overview
The system has been refactored from a benchmark-specific script into a generalized Video-to-3D engine. The new architecture creates isolated `ReconstructionSession` instances per video, enforcing completely disjoint state environments.

## 2. Input Contract
`VideoInputSpec` formalizes mandatory video inputs and optional sensor paths (GPS, IMU, Poses, Calibration, RTK). Missing inputs trigger a graceful diagnostic error (`RECONSTRUCTION_BLOCKED`) rather than causing runtime crashes.

## 3. Session Isolation
Each reconstruction is allocated a unique workspace containing `inputs`, `frames`, `poses`, `geometry`, and `exports`. Data leakage between independent datasets is structurally prohibited.

## 4. Mode Selection & Metric Enforcement
The `ModeSelector` enforces the reconstruction mode contract:
* **Relative Reconstruction:** Activated when valid poses/calibration exist without metric anchors.
* **Metric Reconstruction:** Explicitly fails closed unless a defensible `MetricAnchorCategory` (e.g. `RTK_PPK_GEOMETRY`, `CALIBRATED_STEREO`) is verified.

## 5. Providers
* **Pose Providers:** `PRECOMPUTED_B2` is restricted to the Zurich benchmark. General sessions will expect configuration of future providers like COLMAP or visual odometry.
* **Calibration Providers:** General engines now accept `SUPPLIED_INTRINSICS` or `FULL_OPENCV` through the provider abstraction.

## 6. Execution Flow
The new CLI (`pipelines.application.reconstruct_video`) orchestrates the pipeline:
`Input -> Validate -> Detect Sensors -> Select Mode -> Execute Backend -> Export Result`.

*B6 successfully establishes the generalized framework, leaving the implementation of robust arbitrary-video pose estimation (like COLMAP) for future phases.*
