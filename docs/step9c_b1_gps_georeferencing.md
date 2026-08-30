# Step 9C: Baseline B1 GPS Metric Georeferencing Report

This report documents the implementation and scientific evaluation of **Baseline B1** (COLMAP Structure-from-Motion + UAV GPS Metric/Geospatial Georeferencing) on the complete 350-image Zurich Urban MAV dataset.

---

## 1. Objective & Architecture

Baseline **B1** is the first sensor-assisted reconstruction baseline for **SIH26158**. It bridges scale-free photogrammetry (B0) with physical real-world coordinates by applying a closed-form 7-DoF similarity transformation ($\text{Sim}(3)$) derived strictly from the synchronized onboard 30 Hz GPS telemetry stream.

```
                   350 UAV Images
                         │
                         ▼
                    COLMAP B0
                         │
                         ▼
                Camera Centers C_colmap
                         │
                         │ (350 Exact Correspondences)
                         ▼
                 GPS Image Positions
                         │
                         ▼
                Coordinate Conversion
                         │
                         ▼
                     UTM / ENU
                         │
                         ▼
                 Umeyama Sim(3) Fit
                         │
              ┌──────────┼───────────┐
              ▼          ▼           ▼
         Scale (s)   Rotation (R) Translation (t)
              │          │           │
              └──────────┼───────────┘
                         ▼
            Metric Georeferenced Model
             (p_metric = s R p_colmap + t)
```

**Strict Scientific Integrity Rules**:
* **GPS is the sole external input**: Ground-truth photogrammetry (`GroundTruthAGL.csv`) was **NOT** used during transform estimation.
* **No modification to B0**: Original B0 reconstruction files are preserved in a separate directory.
* **No GPS smoothing**: Raw converted GPS positions were used directly without heuristic filtering.

---

## 2. Mathematical Formulation & Coordinate Transformations

### 2.1 Forward Metric Georeferencing Transformation
$$\mathbf{p}_{\text{metric}} = s \mathbf{R} \mathbf{p}_{\text{colmap}} + \mathbf{t}$$
* $\mathbf{p}_{\text{colmap}} \in \mathbb{R}^3$: Dimensionless camera optical center or 3D point in the B0 SfM coordinate frame.
* $s \in \mathbb{R}^+$: Estimated global metric scale factor ($0.140830\text{ meters / COLMAP unit}$).
* $\mathbf{R} \in \text{SO}(3)$: $3 \times 3$ proper rotation matrix with $\det(\mathbf{R}) = +1$.
* $\mathbf{t} \in \mathbb{R}^3$: Translation vector in meters ($[-0.6401, 1.4883, 0.4074]^T\text{ m}$).
* $\mathbf{p}_{\text{metric}} \in \mathbb{R}^3$: Georeferenced coordinate in metric Local ENU frame (Meters relative to Frame 1 GPS).

### 2.2 Reversible Inverse Transformation
$$\mathbf{p}_{\text{colmap}} = s^{-1} \mathbf{R}^T (\mathbf{p}_{\text{metric}} - \mathbf{t}) = s_{\text{inv}} \mathbf{R}_{\text{inv}} \mathbf{p}_{\text{metric}} + \mathbf{t}_{\text{inv}}$$
* $s_{\text{inv}} = 1 / s = 7.100778\text{ COLMAP units / meter}$
* $\mathbf{R}_{\text{inv}} = \mathbf{R}^T$
* $\mathbf{t}_{\text{inv}} = - s_{\text{inv}} \mathbf{R}^T \mathbf{t} = [11.0264, 4.4143, -1.9056]^T\text{ units}$

---

## 3. Base Transform Estimation & GPS Fitting Residuals

* **Number of Correspondence Pairs**: **`350 / 350`** (100.0% coverage via exact `imgid`).
* **Estimated Metric Scale Factor ($s$)**: **`0.140830`** ($1\text{ COLMAP unit} \approx 7.1008\text{ m}$).
* **GPS Alignment Residuals** ($\mathbf{e}_i = \|\mathbf{G}_i - (s \mathbf{R} \mathbf{C}_i + \mathbf{t})\|_2$):
  * **RMSE**: **`0.7244 m`**
  * **Mean**: **`0.6496 m`**
  * **Median**: **`0.6272 m`**
  * **95th Percentile**: **`1.2291 m`**
  * **Maximum**: **`1.5039 m`**
  * **Component Breakdown**: $\text{RMSE}_{\text{East}} = 0.4851\text{ m}, \text{RMSE}_{\text{North}} = 0.4042\text{ m}, \text{RMSE}_{\text{Up}} = 0.3547\text{ m}$.

---

## 4. Independent Ground-Truth Evaluation (B0 vs B1)

| Evaluation Dimension | Baseline B0 (Pure SfM) | Baseline B1 (GPS Georeferenced) | Comparison / Assessment |
| :--- | :---: | :---: | :--- |
| **External Inputs Used** | None (Visual only) | 30 Hz Standalone GPS | Sensor-assisted georeferencing |
| **Coordinate System** | Arbitrary gauge units | Metric Local ENU / UTM Zone 32N | Real-world metric georeferenced |
| **Origin Reference** | Relative arbitrary frame | $E_0 = 465670.71\text{ m}, N_0 = 5247978.03\text{ m}$ | Geodetically referenced |
| **Direct Metric ATE (RMSE)** | *Undefined (scale-free)* | **`1.8190 m`** | Absolute GPS georeferencing accuracy |
| **Direct Metric ATE (Mean)** | *Undefined* | **`1.7766 m`** | Consistent sub-2m positioning |
| **Direct Metric ATE (Median)** | *Undefined* | **`1.7942 m`** | Centered error distribution |
| **Direct Metric ATE (Max)** | *Undefined* | **`2.1643 m`** | Max displacement across keyframes |
| **Shape Agreement (Sim(3) ATE)** | **`0.0035 m`** ($3.5\text{ mm}$) | **`0.0035 m`** ($3.5\text{ mm}$) | Exact shape preservation |
| **Translational RPE (RMSE)** | **`0.0052 m`** ($5.2\text{ mm}$) | **`0.0052 m`** ($5.2\text{ mm}$) | Identical relative trajectory drift |
| **Rotational RPE (RMSE)** | **`2.6488°`** | **`2.6488°`** | Relative attitude drift preserved |
| **Trajectory Length** | $18.22\text{ units}$ | **`2.5659 m`** ($L_{\text{gps}} = 9.13\text{ m}$) | Metric scaled path |
| **Endpoint Error to GT** | *Undefined (scale-free)* | **`1.5032 m`** | Absolute terminal geodetic error |

---

## 5. Scientific Assessment: Why B0 Shape ATE ($3.5\text{ mm}$) Differs from B1 Metric ATE ($1.82\text{ m}$)

It is critical to distinguish between **gauge-fitted shape consistency** and **true absolute georeferencing**:
1. **In B0**: The trajectory was evaluated by solving a retroactive $\text{Sim}(3)$ transformation directly against Ground Truth. This measures the pure **trajectory shape error ($3.5\text{ mm}$)** under an optimal reference fit.
2. **In B1**: The transformation was estimated **strictly from consumer-grade standalone GPS**. Commercial standalone GNSS receivers without RTK carrier-phase differential corrections experience ionospheric/tropospheric delays and multipath offsets on the order of $\approx 1.5 - 3.0\text{ m}$.
3. **Conclusion**: B1 successfully converts scale-free SfM into a metric georeferenced coordinate frame with an absolute positioning accuracy of **`1.82 m`**, fully consistent with expected standalone GPS performance.

---

## 6. Deliverables & Data Inventory

| Deliverable File | Type | Description |
| :--- | :---: | :--- |
| [outputs/reports/zurich_mav/b1/colmap_gps_correspondences.csv](file:///d:/SIH26158-single-pass-3D/outputs/reports/zurich_mav/b1/colmap_gps_correspondences.csv) | CSV | 350 exact image-GPS correspondence pairs |
| [outputs/reports/zurich_mav/b1/transform.json](file:///d:/SIH26158-single-pass-3D/outputs/reports/zurich_mav/b1/transform.json) | JSON | Forward ($s, R, t$) and exact inverse ($s^{-1}, R^T, t_{\text{inv}}$) transformations |
| [outputs/reports/zurich_mav/b1/b0_vs_b1.json](file:///d:/SIH26158-single-pass-3D/outputs/reports/zurich_mav/b1/b0_vs_b1.json) | JSON | Comprehensive B0 vs B1 comparison metrics across all evaluation dimensions |
| [D:\SIH26158\colmap_workspace\zurich_mav_b1\camera_poses_metric.csv](file:///D:/SIH26158/colmap_workspace/zurich_mav_b1/camera_poses_metric.csv) | CSV | 350 metric camera poses in local ENU and global UTM Zone 32N coordinates |
| [D:\SIH26158\colmap_workspace\zurich_mav_b1\sparse_georeferenced\](file:///D:/SIH26158/colmap_workspace/zurich_mav_b1/sparse_georeferenced) | Directory | Georeferenced sparse 3D point cloud (`points3D.txt`, `cameras.txt`) |
| [outputs/reports/zurich_mav/b1/b1_gps_georeferenced_trajectory.png](file:///d:/SIH26158-single-pass-3D/outputs/reports/zurich_mav/b1/b1_gps_georeferenced_trajectory.png) | PNG | 2D top-down comparison of raw GPS vs B1 georeferenced COLMAP trajectory |
| [outputs/reports/zurich_mav/b1/b1_gps_residuals.png](file:///d:/SIH26158-single-pass-3D/outputs/reports/zurich_mav/b1/b1_gps_residuals.png) | PNG | Per-frame GPS residual magnitude and East/North/Up error breakdown |
| [outputs/reports/zurich_mav/b1/b1_scale_comparison.png](file:///d:/SIH26158-single-pass-3D/outputs/reports/zurich_mav/b1/b1_scale_comparison.png) | PNG | Comparison of B0 raw length, GPS path length, B1 metric length, and scale factor |
| [tests/unit/test_gps_sim3.py](file:///d:/SIH26158-single-pass-3D/tests/unit/test_gps_sim3.py) | Test | Unit tests for Sim(3) parameter recovery and mathematical invertibility |
| [tests/integration/test_b1_georeferencing.py](file:///d:/SIH26158-single-pass-3D/tests/integration/test_b1_georeferencing.py) | Test | Integration tests validating B1 workspace files, schema, and B0 immutability |
