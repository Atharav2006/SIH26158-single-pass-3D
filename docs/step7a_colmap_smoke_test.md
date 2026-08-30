# Step 7A: COLMAP Feature Extraction & Matching Smoke Test

This document records the empirical results, camera model formulation, command line invocations, database verification, and hardware telemetry for the **COLMAP 4.1.1 GPU Smoke Test** on the Zurich Urban MAV dataset.

---

## 1. Objective & Scope

Before executing full incremental bundle adjustment mapping over the entire 350-image dataset, Step 7A performs a lightweight smoke test to verify:
1. COLMAP executable compatibility and CUDA GPU acceleration on the host NVIDIA RTX 3050 Laptop GPU (4 GB VRAM).
2. Exact camera model mapping from Zurich Urban MAV `camera.json` to COLMAP's `OPENCV` radial-tangential model.
3. GPU SIFT feature extraction on real high-resolution ($1920 \times 1080$) aerial frames.
4. GPU-accelerated exhaustive pairwise matching and two-view geometric inlier verification.
5. SQLite database creation, indexing, and persistent readability.

---

## 2. Image Selection

To test spatial feature matching across the entire sample rather than consecutive redundant frames, 10 representative frames were sampled across the 350-image sequence:

| Index | Filename | Original Sample Index | Selection Rationale |
| :--- | :--- | :--- | :--- |
| **1** | `00001.jpg` | Frame 1 | Takeoff / Initial baseline |
| **2** | `00035.jpg` | Frame 35 | Initial forward translation ($\Delta \approx 1\text{ s}$) |
| **3** | `00070.jpg` | Frame 70 | Forward flight corridor |
| **4** | `00105.jpg` | Frame 105 | Street canyon midsection |
| **5** | `00140.jpg` | Frame 140 | Corridor transition |
| **6** | `00175.jpg` | Frame 175 | Mid-flight checkpoint |
| **7** | `00210.jpg` | Frame 210 | Forward street segment |
| **8** | `00245.jpg` | Frame 245 | Urban intersection |
| **9** | `00280.jpg` | Frame 280 | Urban corridor segment |
| **10** | `00350.jpg` | Frame 350 | Sample sequence terminus |

Images were copied to:
`D:\SIH26158\colmap_workspace\smoke_test\images\`

---

## 3. Camera Model & Calibration Parameters

From verified dataset calibration in [outputs/reports/zurich_mav/camera.json](file:///d:/SIH26158-single-pass-3D/outputs/reports/zurich_mav/camera.json):
* **Dataset Model**: `pinhole_radial_tangential`
* **Resolution**: $1920 \times 1080$
* **Focal Length**: $f_x = 893.3901081378665$, $f_y = 898.3264861625313$
* **Principal Point**: $c_x = 951.1310042974931$, $c_y = 555.1335007742958$
* **Distortion Coefficients**:
  * $k_1 = -0.2805251302544365$
  * $k_2 = 0.1158064134556822$
  * $p_1 = -0.0009843367849156311$
  * $p_2 = 0.0001584792476978901$
  * $k_3 = -0.027021503433937236$

### COLMAP Camera Model Mapping: `OPENCV`
In COLMAP, the `OPENCV` camera model expects 8 parameters: `fx, fy, cx, cy, k1, k2, p1, p2`.
Exact parameter string passed to COLMAP:
```text
893.3901081378665,898.3264861625313,951.1310042974931,555.1335007742958,-0.2805251302544365,0.1158064134556822,-0.0009843367849156311,0.0001584792476978901
```

---

## 4. Exact Command Line Invocations

### 4.1 SIFT Feature Extraction
```powershell
& "D:\SIH26158\tools\colmap\colmap.exe" feature_extractor `
    --database_path "D:\SIH26158\colmap_workspace\smoke_test\database.db" `
    --image_path "D:\SIH26158\colmap_workspace\smoke_test\images" `
    --ImageReader.camera_model OPENCV `
    --ImageReader.single_camera 1 `
    --ImageReader.camera_params "893.3901081378665,898.3264861625313,951.1310042974931,555.1335007742958,-0.2805251302544365,0.1158064134556822,-0.0009843367849156311,0.0001584792476978901" `
    --SiftExtraction.max_num_features 8192
```

### 4.2 Exhaustive Feature Matching & Geometric Verification
```powershell
& "D:\SIH26158\tools\colmap\colmap.exe" exhaustive_matcher `
    --database_path "D:\SIH26158\colmap_workspace\smoke_test\database.db"
```

---

## 5. Execution Results & Database Verification

### 5.1 SIFT Feature Extraction Results
* **GPU Extractor**: SIFT GPU (NVIDIA RTX 3050 Laptop GPU, CUDA Device 0)
* **Total Images Processed**: 10
* **Total Features Extracted**: **`97,266`**
* **Extraction Duration**: **`0.90 seconds`** ($0.015\text{ min}$)
* **Per-Image Feature Counts**:
  * `00001.jpg`: 10,912 features
  * `00035.jpg`: 9,726 features
  * `00070.jpg`: 9,303 features
  * `00105.jpg`: 8,322 features
  * `00140.jpg`: 9,826 features
  * `00175.jpg`: 9,451 features
  * `00210.jpg`: 10,347 features
  * `00245.jpg`: 10,333 features
  * `00280.jpg`: 9,348 features
  * `00350.jpg`: 9,698 features
* **Mean Features / Image**: **`9,726.6`** ($\min = 8,322, \max = 10,912$)

### 5.2 Exhaustive Feature Matching & Geometric Inliers
* **GPU Matcher**: SIFT GPU Brute-Force Matcher (CUDA Device 0)
* **Matching Duration**: **`0.68 seconds`** ($0.012\text{ min}$)
* **Total Possible Image Pairs**: $\binom{10}{2} = 45$ pairs
* **Geometrically Verified Image Pairs**: **`45 / 45`** (**`100% success rate`**)
* **Inlier Match Statistics**:
  * Minimum Inliers: 276 matches (Pair 1 $\leftrightarrow$ 10, span across entire 350-frame flight)
  * Maximum Inliers: 1,581 matches (Pair 2 $\leftrightarrow$ 3)
  * Mean Inliers per Pair: **`667.4`** inlier matches

### 5.3 Database Integrity
* Database Path: [D:\SIH26158\colmap_workspace\smoke_test\database.db](file:///D:/SIH26158/colmap_workspace/smoke_test/database.db)
* SQLite Reopen Verification: **`PASS`**
* Schema Tables Verified: `cameras` (1 record), `images` (10 records), `keypoints` (10 records), `descriptors` (10 records), `matches` (45 records), `two_view_geometries` (45 records).

---

## 6. Machine-Readable Artifact

Created at [outputs/reports/zurich_mav/colmap_smoke_test.json](file:///d:/SIH26158-single-pass-3D/outputs/reports/zurich_mav/colmap_smoke_test.json).
