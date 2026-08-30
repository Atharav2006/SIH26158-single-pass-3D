# Step 7B: Classical COLMAP Structure-from-Motion Baseline (B0)

This document establishes the official photogrammetry baseline (**B0**) for the **SIH26158** single-pass 3D reconstruction system, evaluated on the complete 350-image Zurich Urban MAV development dataset.

---

## 1. Objective & Scope

Baseline **B0** represents the conventional image-based photogrammetric benchmark against which all subsequent learned models (VGGT, DUSt3R, MASt3R, Depth/Pose Fusion, 3DGS) will be evaluated.

B0 is strictly defined as:
$$\text{B0} = \text{Real 350 Aerial Images} + \text{Known Zurich Camera Calibration} + \text{SIFT Features} + \text{GPU Exhaustive Matching} + \text{Incremental COLMAP Mapper}$$

**Strict Baseline Isolation**: B0 operates **purely on image data** and does **NOT** incorporate GPS, IMU, onboard telemetry, ground truth poses, learned depth, NeRF, 3DGS, dynamic object filtering, or adaptive frame selection.

---

## 2. Input Dataset Specification

* **Dataset Source**: Zurich Urban MAV Dataset (AGZ Sample Sequence)
* **Image Directory**: [D:\SIH26158\colmap_workspace\zurich_mav_b0\images](file:///D:/SIH26158/colmap_workspace/zurich_mav_b0/images)
* **Total Image Count**: **`350`** sequential aerial frames (`00001.jpg` to `00350.jpg`)
* **Native ID Range**: `imgid ∈ [1, 350]` (1:1 correspondence with numeric filename suffix)
* **Sensor Resolution**: $1920 \times 1080$ pixels (Full HD, 100% uniform)
* **Ground-Truth Availability**: 12 exact keyframes at 1 Hz ($\Delta \text{imgid} = 30$: `1, 31, 61, ..., 331`) from bundle-adjusted photogrammetry (`GroundTruthAGL.csv`).

---

## 3. Camera Calibration & Intrinsics Mapping

* **COLMAP Model**: `FULL_OPENCV` (Model ID 8)
* **Image Size**: $1920 \times 1080$
* **Focal Length**: $f_x = 893.3901081378665\text{ px}$, $f_y = 898.3264861625313\text{ px}$ (Prior: $895.86\text{ px}$)
* **Principal Point**: $c_x = 951.1310042974931\text{ px}$, $c_y = 555.1335007742958\text{ px}$
* **Lens Distortion Vector**:
  $$\mathbf{D} = [k_1, k_2, p_1, p_2, k_3, k_4, k_5, k_6]$$
  * $k_1 = -0.2805251302544365$
  * $k_2 = +0.1158064134556822$
  * $p_1 = -0.0009843367849156311$
  * $p_2 = +0.0001584792476978901$
  * $k_3 = -0.027021503433937236$
  * $k_4 = k_5 = k_6 = 0.0$

---

## 4. Hardware & Execution Telemetry

* **GPU**: NVIDIA GeForce RTX 3050 Laptop GPU (4 GB GDDR6 VRAM, Driver 581.95)
* **CPU**: Intel Core i5-12500H (12 Cores / 16 Threads)
* **RAM**: 16 GB DDR4
* **Operating System**: Windows 11 Home 64-bit

### Runtime Breakdown
| Pipeline Stage | Algorithm / Tool | Device | Runtime (s) | Runtime (min) |
| :--- | :--- | :---: | :---: | :---: |
| **1. Feature Extraction** | SIFT GPU (`max_features=8192`) | GPU | 19.50 s | 0.33 min |
| **2. Feature Matching** | Exhaustive Pairwise Matching ($\binom{350}{2} = 61,075$ pairs) | GPU | 877.87 s | 14.63 min |
| **3. Incremental Mapping** | Incremental SfM + Ceres Levenberg-Marquardt Global BA | CPU + GPU | 9,828.00 s | 163.80 min |
| **Total Pipeline** | **End-to-End B0 Photogrammetry** | - | **`10,725.37 s`** | **`178.76 min`** |

---

## 5. Reconstruction Results & Photogrammetric Metrics

### 5.1 Image Registration Performance
* **Total Input Images**: **`350`**
* **Registered Images**: **`350`** (**`100.0%`**)
* **Unregistered Images**: **`0`** (**`0.0%`**)
* **Reconstruction Components**: **`1`** (Single unified model under `sparse/0/`)
* **Ground-Truth Keyframes Registered**: **`12 / 12`** (**`100.0%`**)

### 5.2 Sparse 3D Point Cloud Geometry
* **Total Sparse 3D Points**: **`50,788`** points
* **Total Feature Observations**: **`1,912,099`**
* **Mean Track Length**: **`37.65`** views / point
* **Median Track Length**: **`14.0`** views / point
* **Mean Observations per Image**: **`5,463.14`** points / image

### 5.3 Reprojection Error Distribution
* **Mean Reprojection Error**: **`0.9868 px`** ($< 1.0\text{ px}$)
* **Median Reprojection Error**: **`0.8801 px`**
* **Maximum Reprojection Error**: **`3.9554 px`**
* **Standard Deviation**: **`0.5709 px`**

---

## 6. Pose Representation & Coordinate Semantics

In COLMAP standard output (`images.txt` / `images.bin`), the transformation represents the **World-to-Camera** mapping ($T_{CW}$):
$$X_C = R_{CW} X_W + t_{CW}$$
Where $q_{CW} = [q_w, q_x, q_y, q_z]$ is stored as scalar-first.

In our exported dataset deliverables ([camera_poses_colmap.csv](file:///d:/SIH26158-single-pass-3D/outputs/reports/zurich_mav/b0/camera_poses_colmap.csv)), poses are provided in both representations:
1. **Camera Optical Center in World Coordinates ($C_W$)**:
   $$C_W = - R_{CW}^T t_{CW}$$
2. **Camera Attitude in World Coordinates ($q_{WC}$)**:
   $$q_{WC} = [-q_{CW,x}, -q_{CW,y}, -q_{CW,z}, q_{CW,w}]$$
   (Stored in standard Hamilton scalar-last $[q_x, q_y, q_z, q_w]$ format).

---

## 7. Known Failure Cases & Photogrammetric Limitations

1. **High Computational Complexity**: Exhaustive matching scales quadratically ($\mathcal{O}(N^2)$), requiring $14.6\text{ minutes}$ for 350 frames, while global bundle adjustments dominate total runtime ($163.8\text{ minutes}$).
2. **Scale Ambiguity**: Pure monocular SfM reconstructs geometry up to an arbitrary global scale factor. Metric scale cannot be resolved without external scale constraints (e.g. GPS / IMU baseline fusion).
3. **Dynamic Object Sensitivity**: Moving pedestrians and vehicles create ghost tracks and spurious feature matches that must be filtered in downstream intelligent pipelines.
4. **Point Sparsity in Low-Texture Regions**: Asphalt roads and uniform rooftop surfaces yield significantly fewer 3D points than textured building facades.

---

## 8. Reproducibility Command

To reproduce the exact baseline B0 run from scratch:
```powershell
# 1. Feature Extraction
& "D:\SIH26158\tools\colmap\colmap.exe" feature_extractor `
    --database_path "D:\SIH26158\colmap_workspace\zurich_mav_b0\database.db" `
    --image_path "D:\SIH26158\colmap_workspace\zurich_mav_b0\images" `
    --ImageReader.camera_model FULL_OPENCV `
    --ImageReader.single_camera 1 `
    --ImageReader.camera_params "893.3901081378665,898.3264861625313,951.1310042974931,555.1335007742958,-0.2805251302544365,0.1158064134556822,-0.0009843367849156311,0.0001584792476978901,-0.027021503433937236,0,0,0" `
    --FeatureExtraction.use_gpu 1 `
    --SiftExtraction.max_num_features 8192

# 2. Exhaustive Matching
& "D:\SIH26158\tools\colmap\colmap.exe" exhaustive_matcher `
    --database_path "D:\SIH26158\colmap_workspace\zurich_mav_b0\database.db" `
    --FeatureMatching.use_gpu 1

# 3. Incremental Mapper
& "D:\SIH26158\tools\colmap\colmap.exe" mapper `
    --database_path "D:\SIH26158\colmap_workspace\zurich_mav_b0\database.db" `
    --image_path "D:\SIH26158\colmap_workspace\zurich_mav_b0\images" `
    --output_path "D:\SIH26158\colmap_workspace\zurich_mav_b0\sparse"
```

---

## 9. Deliverables Inventory

| Deliverable File | Type | Description |
| :--- | :---: | :--- |
| [outputs/reports/zurich_mav/b0/reconstruction_summary.json](file:///d:/SIH26158-single-pass-3D/outputs/reports/zurich_mav/b0/reconstruction_summary.json) | JSON | Complete summary metrics, camera model, image counts, error stats |
| [outputs/reports/zurich_mav/b0/camera_poses_colmap.csv](file:///d:/SIH26158-single-pass-3D/outputs/reports/zurich_mav/b0/camera_poses_colmap.csv) | CSV | 350 camera centers ($C_W$), attitudes ($q_{WC}$), COLMAP $T_{CW}$, GT flags |
| [outputs/reports/zurich_mav/b0/registered_images.csv](file:///d:/SIH26158-single-pass-3D/outputs/reports/zurich_mav/b0/registered_images.csv) | CSV | Per-image registration flags, 2D observations, 3D points |
| [outputs/reports/zurich_mav/b0/sparse_points_summary.json](file:///d:/SIH26158-single-pass-3D/outputs/reports/zurich_mav/b0/sparse_points_summary.json) | JSON | Point count, track lengths, observations, reprojection errors |
| [outputs/reports/zurich_mav/b0/matching_summary.json](file:///d:/SIH26158-single-pass-3D/outputs/reports/zurich_mav/b0/matching_summary.json) | JSON | Pairwise match counts, geometric inlier stats across 61,075 pairs |
| [outputs/reports/zurich_mav/b0/performance.json](file:///d:/SIH26158-single-pass-3D/outputs/reports/zurich_mav/b0/performance.json) | JSON | Hardware specs, detailed runtime breakdown across all stages |
| [outputs/reports/zurich_mav/b0_camera_trajectory.png](file:///d:/SIH26158-single-pass-3D/outputs/reports/zurich_mav/b0_camera_trajectory.png) | Image | 2D flight path with start/end markers and 12 GT keyframe stations |
| [outputs/reports/zurich_mav/b0_sparse_reconstruction.png](file:///d:/SIH26158-single-pass-3D/outputs/reports/zurich_mav/b0_sparse_reconstruction.png) | Image | 3D isometric projection of 50,788 sparse points and camera positions |
| [outputs/reports/zurich_mav/b0_registration_map.png](file:///d:/SIH26158-single-pass-3D/outputs/reports/zurich_mav/b0_registration_map.png) | Image | Sequential timeline mapping 100% registration across all 350 frames |
