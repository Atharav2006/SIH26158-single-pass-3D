# Step 7A.3: Calibrated COLMAP Smoke Test (`FULL_OPENCV`)

This document records the execution results, parameter validation, database inspection, and graph connectivity metrics for the **Calibrated COLMAP 4.1.1 Smoke Test** using the exact 12-parameter `FULL_OPENCV` model on the Zurich Urban MAV dataset.

---

## 1. Executive Summary

| Verification Metric | Required Specification | Measured Database Value | Compliance Status |
| :--- | :--- | :--- | :---: |
| **COLMAP Executable** | COLMAP 4.1.1 (CUDA) | COLMAP 4.1.1 (Commit a0d785f with CUDA) | **PASS** |
| **Workspace Database** | `smoke_test_calibrated\database.db` | Clean SQLite database instantiated | **PASS** |
| **Camera Model** | `FULL_OPENCV` (Model ID 8 / 6) | `FULL_OPENCV` | **PASS** |
| **Focal Length $f_x, f_y$** | $893.390108, 898.326486$ | $893.390108, 898.326486$ (Prior: $895.86\text{ px}$) | **PASS** (Not 2304 px) |
| **Principal Point $c_x, c_y$** | $951.131004, 555.133501$ | $951.131004, 555.133501$ ($0.0\text{ px}$ shift) | **PASS** |
| **Distortion $k_1, k_2, p_1, p_2, k_3$** | $[-0.280525, 0.115806, -0.000984, 0.000158, -0.027022]$ | Exact 5-parameter plumb_bob vector | **PASS** |
| **Image Count** | Exactly 10 representative frames | 10 image records present | **PASS** |
| **Keypoints & Descriptors** | 10/10 coverage, 128-dim SIFT | 97,266 SIFT features & descriptors | **PASS** |
| **Pairwise Matching** | $\binom{10}{2} = 45$ pairs | 45 / 45 pairs matched | **PASS** |
| **Geometric Inliers $\ge 15$** | All viable baseline pairs | **45 / 45** (100%) | **PASS** |
| **Geometric Inliers $\ge 30$** | Strict threshold | **45 / 45** (100%) | **PASS** |
| **Graph Topology** | Single connected component | 1 Component ($K_{10}$ complete graph) | **PASS** |
| **Calibrated Smoke Status** | Useful verified matches | **`CALIBRATED COLMAP SMOKE TEST: PASS`** | **PASS** |

---

## 2. Verified Camera Model & Database Intrinsics

```text
Camera ID: 1
Model: FULL_OPENCV (Dimensions: 1920 x 1080)
Prior Focal Length: 1 (User-Supplied Fixed Prior)
Parameters (12 floats):
  fx = 893.3901081378665 px
  fy = 898.3264861625313 px
  cx = 951.1310042974931 px
  cy = 555.1335007742958 px
  k1 = -0.2805251302544365
  k2 =  0.1158064134556822
  p1 = -0.0009843367849156311
  p2 =  0.0001584792476978901
  k3 = -0.027021503433937236
  k4 =  0.0
  k5 =  0.0
  k6 =  0.0
```

---

## 3. Exact Commands Executed

### 3.1 Feature Extraction (GPU)
```powershell
& "D:\SIH26158\tools\colmap\colmap.exe" feature_extractor `
    --database_path "D:\SIH26158\colmap_workspace\smoke_test_calibrated\database.db" `
    --image_path "D:\SIH26158\colmap_workspace\smoke_test_calibrated\images" `
    --ImageReader.camera_model FULL_OPENCV `
    --ImageReader.single_camera 1 `
    --ImageReader.camera_params "893.3901081378665,898.3264861625313,951.1310042974931,555.1335007742958,-0.2805251302544365,0.1158064134556822,-0.0009843367849156311,0.0001584792476978901,-0.027021503433937236,0,0,0" `
    --FeatureExtraction.use_gpu 1 `
    --SiftExtraction.max_num_features 8192
```

### 3.2 Feature Matching (GPU)
```powershell
& "D:\SIH26158\tools\colmap\colmap.exe" exhaustive_matcher `
    --database_path "D:\SIH26158\colmap_workspace\smoke_test_calibrated\database.db" `
    --FeatureMatching.use_gpu 1
```

### 3.3 Geometric Verification
```powershell
& "D:\SIH26158\tools\colmap\colmap.exe" geometric_verifier `
    --database_path "D:\SIH26158\colmap_workspace\smoke_test_calibrated\database.db"
```

---

## 4. Feature Extraction & Database Validation

| Image ID | Filename | Dimensions | Keypoints | Descriptors | Consistency |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **1** | `00001.jpg` | $1920 \times 1080$ | 10,912 | 10,912 (128-dim) | Exact (100%) |
| **2** | `00035.jpg` | $1920 \times 1080$ | 9,726 | 9,726 (128-dim) | Exact (100%) |
| **3** | `00070.jpg` | $1920 \times 1080$ | 9,303 | 9,303 (128-dim) | Exact (100%) |
| **4** | `00105.jpg` | $1920 \times 1080$ | 8,322 | 8,322 (128-dim) | Exact (100%) |
| **5** | `00140.jpg` | $1920 \times 1080$ | 9,826 | 9,826 (128-dim) | Exact (100%) |
| **6** | `00175.jpg` | $1920 \times 1080$ | 9,451 | 9,451 (128-dim) | Exact (100%) |
| **7** | `00210.jpg` | $1920 \times 1080$ | 10,347 | 10,347 (128-dim) | Exact (100%) |
| **8** | `00245.jpg` | $1920 \times 1080$ | 10,333 | 10,333 (128-dim) | Exact (100%) |
| **9** | `00280.jpg` | $1920 \times 1080$ | 9,348 | 9,348 (128-dim) | Exact (100%) |
| **10** | `00350.jpg` | $1920 \times 1080$ | 9,698 | 9,698 (128-dim) | Exact (100%) |

* **Total Keypoints**: **`97,266`** (Mean: **`9,726.6`** features/image)
* **Extraction Duration**: **`0.75 seconds`**

---

## 5. Matching & Two-View Inlier Statistics

* **Total Attempted Pairs**: $\binom{10}{2} = 45$
* **Raw Matched Pairs**: **`45 / 45`** ($100\%$)
* **Geometrically Verified Pairs**: **`45 / 45`** ($100\%$)
* **Pairs with Inliers $\ge 15$**: **`45 / 45`** ($100\%$)
* **Pairs with Inliers $\ge 30$**: **`45 / 45`** ($100\%$)

### Inlier Distribution
* **Minimum Inliers**: **`289`** (Pair 1 $\leftrightarrow$ 10, spanning the entire flight sequence)
* **Maximum Inliers**: **`2,092`** (Pair 3 $\leftrightarrow$ 4)
* **Median Inliers**: **`922.0`**
* **Mean Inliers**: **`976.29`** (Mean Inlier Ratio: **`92.8%`**)
* **Matching Runtime**: **`0.69 seconds`**

---

## 6. Image Graph Topology & Connectivity

* **Graph Type**: Complete Connected Graph ($K_{10}$, Density $= 1.0$)
* **Number of Connected Components**: **`1`**
* **Per-Node Degree**: Every node has degree $k = 9$ and $> 5,100$ total incident verified inliers.

---

## 7. Machine-Readable Artifacts
* **Graph Report**: [outputs/reports/zurich_mav/colmap_calibrated_smoke_graph.json](file:///d:/SIH26158-single-pass-3D/outputs/reports/zurich_mav/colmap_calibrated_smoke_graph.json)
* **Full Report**: [outputs/reports/zurich_mav/colmap_calibrated_smoke_report.json](file:///d:/SIH26158-single-pass-3D/outputs/reports/zurich_mav/colmap_calibrated_smoke_report.json)
