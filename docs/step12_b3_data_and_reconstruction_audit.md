# Phase 0: B3 Data and Reconstruction Audit

This document summarizes the state of the workspace prior to implementing Baseline B3 (Dense Georeferenced 3D Reconstruction).

## 1. Input Data
- **Images:** 350 `.jpg` images at 1920x1080 resolution, located in `D:/SIH26158/datasets/zurich_mav/AGZ_subset/MAV Images/`.
- **Camera Intrinsics:** Pre-calibrated `FULL_OPENCV` model. 
  - $f_x = 893.3901, f_y = 898.3265$
  - $c_x = 951.1310, c_y = 555.1335$
  - Distortion ($k_1, k_2, p_1, p_2, k_3$): `[-0.2805, 0.1158, -0.00098, 0.000158, -0.0270]`
- **Trajectory:** The B2 output (`b2_fused_trajectory.csv`) provides high-quality metric Local ENU camera centers and `Camera-to-World` ($\mathbf{q}_{wc}$) orientations.
  - Quaternion format: Hamilton `[qx, qy, qz, qw]`

## 2. COLMAP Reconstruction Status
- **Sparse Workspace:** B0 generated 50,788 sparse 3D points. 
- **Dense Workspace:** **MISSING**. No depth maps, normal maps, fused point clouds, or meshes currently exist.

## 3. Computational Environment
- **GPU:** NVIDIA GeForce RTX 3050 Laptop GPU
- **VRAM:** 4.29 GB (This is a strict constraint for dense reconstruction; out-of-core or batched processing may be necessary for large point clouds or memory-heavy architectures).
- **Frameworks:** PyTorch `2.12.0+cu130`, OpenCV `5.0.0`, Open3D `0.19.0`.

## 4. Existing Components Reusability
- **Reusable:** The 350 images, the intrinsic camera parameters (from B0), and the optimized metric poses (from B2).
- **Missing:** Any dense depth estimation, point cloud fusion, or 3D rendering pipeline modules in `src/`.

## 5. Recommended B3 Architecture
Given the missing dense workspace and the strict 4GB VRAM hardware constraint, the classical B3 baseline should avoid heavy unbatched neural rendering initially.
Instead, we should implement a robust classical multi-view stereo (MVS) pipeline:
1. **Workspace Preparation:** Export the B2 metric trajectory into a COLMAP-compatible dense workspace (`images.txt`, `cameras.txt` with B2 metric poses). Note that COLMAP requires World-to-Camera poses (`q_cw`, `t_cw`), so the B2 `Camera-to-World` poses must be inverted during export.
2. **Dense Stereo:** Use COLMAP's PatchMatch Stereo (via CLI wrappers) to compute dense depth and normal maps. PatchMatch Stereo handles memory out-of-core and is highly VRAM efficient.
3. **Fusion:** Fuse the depth maps into a dense 3D point cloud (`fused.ply`).
4. **Metric Georeferencing Verification:** Since the input cameras to the MVS pipeline will be the B2 *metric* poses, the resulting dense point cloud will *natively* be reconstructed in the B2 Metric Local ENU coordinate frame. We will explicitly test and verify this scale/orientation using Open3D.

## 6. Required Files to Create/Modify
- `src/reconstruction/dense_mvs.py`: Python wrappers to orchestrate COLMAP dense reconstruction CLI and handle workspace generation.
- `pipelines/baseline/b3_dense_reconstruction.py`: End-to-end B3 execution pipeline.
- `src/visualization/b3_cloud_visualizer.py`: Open3D rendering utilities for point cloud preview and trajectory overlay.
- `tests/integration/test_b3_reconstruction.py`: Validation tests for metric scale preservation and pose convention immutability.
