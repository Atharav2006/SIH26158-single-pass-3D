# Step 8: Classical COLMAP B0 Trajectory Evaluation Report (Hardened for Research Comparisons)

This document presents the hardened scientific evaluation of the **Classical COLMAP Structure-from-Motion Baseline (B0)** camera trajectory against photogrammetric ground truth from the **Zurich Urban MAV Dataset**.

---

## 1. Executive Summary

| Evaluation Dimension | Metric Category | Value | Scientific Interpretation |
| :--- | :--- | :---: | :--- |
| **Internal Reconstruction** | Mean Reprojection Error | **`0.9868 px`** | Sub-pixel ray triangulation accuracy across 1.91 million observations |
| **Trajectory Shape Agreement** | $\text{Sim}(3)$ ATE RMSE | **`0.0035 m`** ($3.5\text{ mm}$) | High positional shape agreement along the flight path |
| | Translational RPE RMSE | **`0.0052 m`** ($5.2\text{ mm}$) | Sub-centimeter step-to-step relative translation drift |
| | Rotational RPE RMSE | **`2.6488°`** | Relative attitude drift per 1 Hz keyframe step |
| | Endpoint Position Error | **`0.0032 m`** ($3.2\text{ mm}$) | $0.10\%$ normalized trajectory endpoint drift |
| **Metric-Scale Analysis** | Estimated Scale ($s$) | **`0.191025`** | $1\text{ COLMAP unit} \approx 5.2349\text{ m}$ (Gauge scale freedom) |
| | Scale Discrepancy | **`80.90%`** | Quantifies unscaled monocular scale offset ($|s - 1| \times 100$) |
| | Path Length Ratio | **`5.242072`** | Ratio of raw COLMAP length ($15.9359\text{ units}$) to GT ($3.0400\text{ m}$) |
| **Evaluation Robustness** | Leave-One-Out (LOO) RMSE | **`0.0043 m`** ($4.3\text{ mm}$) | Generalization error on held-out keyframes (no single-point overfitting) |
| | Fixed-Transform Test | **`0.0035 m`** | 100% deterministic reproducibility across independent evaluations |

---

## 2. How to Correctly Interpret the B0 Results

To maintain research integrity and avoid conflating distinct photogrammetric phenomena, B0 results are classified into **four strictly segregated evaluation dimensions**:

```
                               ┌────────────────────────────────────────────────────────┐
                               │       COLMAP B0 EVALUATION DIMENSIONS                  │
                               └────────────────────────────────────────────────────────┘
                                      │                        │
             ┌────────────────────────┴─────────┐    ┌─────────┴────────────────────────┐
             ▼                                  ▼    ▼                                  ▼
┌─────────────────────────┐  ┌────────────────────┐┌─────────────────────┐  ┌─────────────────────┐
│ 1. Internal Quality     │  │ 2. Shape Agreement ││ 3. Metric-Scale     │  │ 4. Absolute Metric  │
│ (Image-Space Pixels)    │  │ (Sim(3) ATE / RPE) ││ (Scale Discrepancy) │  │ Accuracy            │
│                         │  │                    ││                     │  │                     │
│ • Reproj. Err: 0.9868px │  │ • ATE RMSE: 3.5mm  ││ • Scale s: 0.191025 │  │ • NOT achieved by   │
│ • Track Length: 37.65   │  │ • RPE RMSE: 5.2mm  ││ • Scale Err: 80.90% │  │   pure monocular    │
│ • Observations: 1.91M   │  │ • Rot. RPE: 2.65°  ││ • Length Ratio: 5.24│  │   vision alone      │
└─────────────────────────┘  └────────────────────┘└─────────────────────┘  └─────────────────────┘
```

1. **Internal Reconstruction Quality (Image-Space)**:
   * Quantified by **reprojection error ($0.9868\text{ px}$)** and mean track length ($37.65$).
   * Measures internal multi-view geometric consistency inside the image sensor grid.
2. **Trajectory Shape Agreement (Scale-Invariant Spatial Consistency)**:
   * Quantified by **$\text{Sim}(3)$ ATE ($3.5\text{ mm}$)** and **RPE ($5.2\text{ mm}$)**.
   * Measures how accurately the reconstructed trajectory shape matches the physical ground-truth flight path after resolving global gauge freedoms (7 DoF: scale, rotation, translation).
3. **Metric-Scale Accuracy (Scale Observability)**:
   * Quantified by the **scale ratio ($s = 0.191025$)** and **scale discrepancy ($80.90\%$)**.
   * Monocular vision has a fundamental scale gauge ambiguity ($X_C \sim \lambda X_C$); absolute scale is unobservable without metric sensors.
4. **Absolute Metric Accuracy**:
   * Pure monocular SfM does **not** provide absolute metric accuracy independently. Metric scale requires downstream sensor fusion (GPS, IMU, or metric depth priors).

---

## 3. Scale Reporting & Mathematical Equations

$$\text{Estimated Similarity Scale: } s = \frac{\text{Tr}(\mathbf{D} \mathbf{S})}{\sigma_{\text{colmap}}^2} = 0.191025\text{ meters / COLMAP unit}$$
$$\text{Inverse Scale Factor: } s^{-1} = \frac{1}{s} = 5.234914\text{ COLMAP units / meter}$$
$$\text{Scale Discrepancy Percentage: } \delta_s = |s - 1.0| \times 100\% = 80.90\%$$
$$\text{Raw-to-Metric Path Length Ratio: } \frac{L_{\text{colmap}}}{L_{\text{GT}}} = \frac{15.9359\text{ units}}{3.0400\text{ meters}} = 5.242072$$

---

## 4. Trajectory Shape Metrics (Segregated from ATE)

| Trajectory Shape Metric | Formula / Definition | Measured Value |
| :--- | :--- | :---: |
| **Segment Length Error (Mean)** | $\frac{1}{N-1} \sum \|s \Delta \mathbf{p}_{\text{colmap}, i}\| - \|\Delta \mathbf{p}_{\text{GT}, i}\||$ | **`0.0035 m`** ($3.5\text{ mm}$) |
| **Segment Length Error (Max)** | $\max \|s \Delta \mathbf{p}_{\text{colmap}, i}\| - \|\Delta \mathbf{p}_{\text{GT}, i}\||$ | **`0.0076 m`** ($7.6\text{ mm}$) |
| **Maximum Lateral Deviation** | Cross-track deviation perpendicular to flight corridor | **`0.0055 m`** ($5.5\text{ mm}$) |
| **Mean Lateral Deviation** | Average cross-track deviation | **`0.0030 m`** ($3.0\text{ mm}$) |
| **Maximum Vertical Deviation** | Peak altitude deviation along Z-axis: $\max |z_{\text{aligned}, i} - z_{\text{GT}, i}|$ | **`0.0038 m`** ($3.8\text{ mm}$) |
| **Mean Vertical Deviation** | Average altitude deviation | **`0.0016 m`** ($1.6\text{ mm}$) |
| **Cumulative Path-Length Ratio** | $L_{\text{aligned}} / L_{\text{GT}} = 3.0442\text{ m} / 3.0400\text{ m}$ | **`1.00137`** ($0.14\%$ error) |
| **Endpoint Position Error** | $\|\mathbf{p}_{\text{aligned}, N} - \mathbf{p}_{\text{GT}, N}\|_2$ | **`0.0032 m`** ($3.2\text{ mm}$) |

---

## 5. Rotation Error Breakdown

Step-to-step relative attitude error and global rotation deviations are exported to [b0_rotation_error.csv](file:///d:/SIH26158-single-pass-3D/outputs/reports/zurich_mav/b0/b0_rotation_error.csv):
* **Step-to-Step Relative Rotational RPE (RMSE)**: **`2.6488°`**
* **Step-to-Step Relative Rotational RPE (Mean)**: **`2.1384°`**
* **Step-to-Step Relative Rotational RPE (Median)**: **`2.2212°`**
* **Step-to-Step Relative Rotational RPE (Max)**: **`4.1166°`**
* **Global Absolute Rotation Error (Mean)**: **`1.8540°`** (Max: **`3.6705°`**)

---

## 6. Leave-One-Out (LOO) Cross-Validation Robustness Analysis

To verify that the $\text{Sim}(3)$ alignment parameters are not overfitted to specific keyframes in the 12-pose evaluation subset, leave-one-out cross-validation was performed:

$$\forall i \in \{1 \dots 12\}: \quad (s_{-i}, R_{-i}, t_{-i}) = \text{Umeyama}(\mathcal{P}_{\text{colmap} \setminus \{i\}}, \mathcal{P}_{\text{GT} \setminus \{i\}})$$
$$e_{\text{held-out}, i} = \|s_{-i} R_{-i} C_{W, i} + t_{-i} - \mathbf{p}_{\text{GT}, i}\|_2$$

* **Held-Out ATE RMSE**: **`0.0043 m`** ($4.3\text{ mm}$)
* **Held-Out ATE Mean**: **`0.0040 m`** ($4.0\text{ mm}$)
* **Held-Out ATE Median**: **`0.0039 m`** ($3.9\text{ mm}$)
* **Held-Out ATE Max**: **`0.0069 m`** ($6.9\text{ mm}$)
* **Degeneracy Check**: Zero degenerate folds encountered.
* **Full Artifact**: [outputs/reports/zurich_mav/b0/b0_leave_one_out.json](file:///d:/SIH26158-single-pass-3D/outputs/reports/zurich_mav/b0/b0_leave_one_out.json)

---

## 7. Fixed-Transform Reproducibility Verification

To ensure reproducibility across independent evaluation scripts without secondary optimization:
* **Frozen Scale ($s$)**: `0.191025`
* **Frozen Rotation Matrix ($R$)**:
  $$\begin{bmatrix} -0.007629 & -0.999818 & 0.017500 \\ 0.052674 & -0.017873 & -0.998452 \\ 0.998582 & -0.006696 & 0.052799 \end{bmatrix}$$
* **Frozen Translation ($t$)**: `[-0.015091, 0.016393, 2.378939]^T\text{ m}`
* **Recomputed ATE RMSE**: **`0.0035 m`** (Identical to reported metric).

---

## 8. Documented Scientific Limitations

1. **Discrete Keyframe Ground Truth**: Photogrammetric ground truth (`GroundTruthAGL.csv`) is surveyed at 1 Hz ($\Delta = 30$ frames). Intermediate 30 FPS frames cannot be evaluated directly without interpolation assumptions.
2. **Scale Ambiguity**: Pure monocular SfM reconstructs geometry up to an unknown scale factor ($s = 0.191025$).
3. **Subsequence Sample**: The current 350-image sequence covers an initial takeoff and flight corridor; future full-flight evaluations will assess long-term drift across the entire 81k image sequence.

---

## 9. Deliverables Inventory

| Deliverable File | Type | Description |
| :--- | :---: | :--- |
| [outputs/reports/zurich_mav/b0/b0_evaluation.json](file:///d:/SIH26158-single-pass-3D/outputs/reports/zurich_mav/b0/b0_evaluation.json) | JSON | Hardened evaluation report with all 4 segregated dimensions |
| [outputs/reports/zurich_mav/b0/b0_leave_one_out.json](file:///d:/SIH26158-single-pass-3D/outputs/reports/zurich_mav/b0/b0_leave_one_out.json) | JSON | Complete 12-fold leave-one-out cross-validation results |
| [outputs/reports/zurich_mav/b0/b0_rotation_error.csv](file:///d:/SIH26158-single-pass-3D/outputs/reports/zurich_mav/b0/b0_rotation_error.csv) | CSV | Per-keyframe global and relative step rotation errors |
| [outputs/reports/zurich_mav/b0/b0_gt_evaluation_pairs.csv](file:///d:/SIH26158-single-pass-3D/outputs/reports/zurich_mav/b0/b0_gt_evaluation_pairs.csv) | CSV | 12 exact evaluation keyframe pairs with raw GT and COLMAP poses |
| [outputs/reports/zurich_mav/b0/b0_gt_vs_colmap_topdown.png](file:///d:/SIH26158-single-pass-3D/outputs/reports/zurich_mav/b0/b0_gt_vs_colmap_topdown.png) | PNG | 2D top-down comparison with metric axes and error vectors |
| [outputs/reports/zurich_mav/b0/b0_position_error.png](file:///d:/SIH26158-single-pass-3D/outputs/reports/zurich_mav/b0/b0_position_error.png) | PNG | Per-keyframe positional ATE bar chart |
| [outputs/reports/zurich_mav/b0/b0_trajectory_comparison_3d.png](file:///d:/SIH26158-single-pass-3D/outputs/reports/zurich_mav/b0/b0_trajectory_comparison_3d.png) | PNG | 3D isometric comparison of GT vs raw COLMAP vs aligned COLMAP |
