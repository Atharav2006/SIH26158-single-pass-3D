# B5 Monocular Depth Fusion Engine — Phases 1 and 2

This document records the foundational architecture of the B5 Monocular Depth Fusion Engine, successfully completing Phase 1 and Phase 2. The primary goal of these phases was to mathematically codify the strict boundaries between relative depth inference and metric 3D unprojection.

## 1. Depth Prior Interface (Phase 1)
* **Implementation**: `src/depth_fusion/depth_prior.py`
* **Semantics**: The `MiDaSDepthPrior` wraps the `MiDaS_small` model.
* **Strict Law**: MiDaS output is explicitly tagged as **relative inverse depth**. The interface strictly forbids arbitrary metric conversion or scaling. The output tensor must never be labeled or treated as meters.

## 2. Camera Rectification (Phase 2)
* **Implementation**: `src/depth_fusion/camera_preprocessing.py`
* **Process**: We consume the `FULL_OPENCV` intrinsic matrix and distortion coefficients validated in B0.
* **Pipeline**:
    1. Distorted 1920x1080 RGB
    2. `cv2.initUndistortRectifyMap`
    3. `cv2.remap` -> Rectified image
    4. Computes `K_rect` for pinhole equivalence.

## 3. Ray Generation & Poses (Phase 2B & 2F)
* **Implementation**: `src/depth_fusion/rays.py`
* **Conventions**:
    * Camera: OpenCV (+X Right, +Y Down, +Z Forward).
    * B2 Pose: Camera-to-World ($X_w = R_{wc} X_c + C_{w}$).
* **Transformation**: World ray directions are computed strictly as $R_{wc} \cdot d_{camera}$, originating exactly at $C_w$ (Local ENU).

## 4. Depth Semantics & Unprojection (Phase 2C & 2D)
* **Implementation**: `src/depth_fusion/depth_semantics.py`, `src/depth_fusion/unprojection.py`
* **Inversion**: Inverting the MiDaS output ($1 / D_{midas}$) yields **relative depth**, not metric depth.
* **Unprojection Safety Check**: The `unproject_to_3d` function includes a strict flag (`is_metric=True`). If the depth tensor is not metrically calibrated, the function raises a `ValueError` and halts execution, guaranteeing that relative disparity is never accidentally fused into the metric B2 trajectory.

## 5. Round-Trip Validation (Phase 2E)
A fully comprehensive round-trip projection unit test (`test_b5_unprojection.py`) mathematically verified the entire chain:
`Pixel -> Unproject to World (Metric) -> Project Back to Pixel`
The reprojection error evaluates to `< 1e-3` pixels.

## 6. Current Scientific Status
**Metric Depth Status**: **NOT ESTABLISHED**
**MiDaS Depth**: **RELATIVE ONLY**
**Phase 3**: Intentionally Deferred

### Success Condition
**B5 PHASE 1–2 STATUS: PASS**
All interfaces, unprojection mathematics, and strict semantic barriers are complete and passing the unit test suite.
