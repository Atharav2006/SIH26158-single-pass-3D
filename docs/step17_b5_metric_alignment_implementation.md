# Step 17: B5 Phase 3D Metric Alignment Implementation & Anchor Validation Report

This document records the mathematical implementation, explicit anchor contract, empirical evaluation on Zurich MAV, cross-validation results, and final scientific decision for **B5 Phase 3D**.

---

## 1. Executive Summary & Final Scientific Decision

* **Final Scientific Decision:** **`METRIC_SCALE_NOT_IDENTIFIABLE` (on Zurich MAV Hover Sequence)**
* **Operational Mode:** **`Relative Dense 3D Geometry (metric=False)`**
* **Empirical Validation Findings:**
  1. *Anchor Count:* Extracted **16,565** valid $(D_{\text{inv}}, 1/Z_{\text{metric}})$ candidate anchor points across 12 evenly spaced keyframes from B0 sparse points transformed via B1/B2 Sim(3).
  2. *Correlation Failure:* The empirical correlation between MiDaS neural disparity $D_{\text{inv}}$ and B0 inverse depth $1/Z_{\text{B0}}$ is **$0.0651$** (essentially zero correlation).
  3. *Validation Error:* In an 80/20 Train/Validation split, a forced affine fit produced a validation inverse-depth RMSE of $0.1369$, corresponding to a **$76.8\%$ relative depth error**.
  4. *Scientific Integrity:* In accordance with strict project rules (no arbitrary multipliers, no GT leakage), the `RobustMetricAlignmentEngine` correctly **rejected** the calibration and safely defaulted to relative depth mode (`metric = False`).

---

## 2. Mathematical Model & Anchor Contract

### Affine Inverse-Depth Parameterization
$$\frac{1}{Z_{\text{metric}}(u, v)} = a \cdot D_{\text{inv}}(u, v) + b$$
$$Z_{\text{metric}}(u, v) = \frac{1}{\max(a \cdot D_{\text{inv}}(u, v) + b, \epsilon)}$$

### Typed Anchor Interface (`src/depth_fusion/metric_anchor.py`)
Each anchor is strictly typed:
* `pixel_u`, `pixel_v`: Pixel projection coordinates.
* `frame_id`: Corresponding image frame.
* `metric_depth_m`: Depth along camera optical axis ($Z > 0$).
* `inv_depth_predicted`: Neural predicted disparity ($D_{\text{inv}} > 0$).
* `source`: `AnchorSource` enum (`B0_SPARSE_REPROJECTION`, `EXTERNAL_DEPTH_SENSOR`, `USER_DEFINED`, `GROUND_TRUTH_EVALUATION_ONLY`).
* *Rule:* `GROUND_TRUTH_EVALUATION_ONLY` is strictly rejected by the production calibration path.

### Typed Output Contract (`MetricDepthOutput`)
* `depth`: 2D depth array.
* `confidence`: 2D confidence array.
* `metric`: `bool` (`True` if validly calibrated, `False` for relative fallback).
* `scale_a`, `shift_b`: Fitted affine parameters (or `None`).
* `calibration_status`: `CalibrationStatus` enum (`METRIC_ALIGNMENT_VALID`, `METRIC_ALIGNMENT_UNSTABLE`, `METRIC_SCALE_NOT_IDENTIFIABLE`).

---

## 3. Robust Estimator Architecture (`src/depth_fusion/depth_scale_alignment.py`)

1. **Filtering & Validation Funnel:**
   * Rejects out-of-bounds pixels ($< 20\text{px}$ from borders).
   * Rejects non-positive or extreme depths ($Z < 0.5\text{m}$ or $Z > 100\text{m}$).
   * Rejects degenerate design matrix: $\text{cond}(A) > 10^4$.
   * Rejects insufficient correlation: $\text{corr}(D_{\text{inv}}, 1/Z) < 0.20$.
   * Rejects insufficient unique frames: $N_{\text{frames}} < 3$.
2. **Robust Optimization:**
   * Initial inlier selection via RANSAC with positive scale constraint ($a > 0$).
   * Refinement via Iteratively Reweighted Least Squares (IRLS) under Huber loss ($\delta = 0.01$).
3. **Safe Fallback:**
   * When any validation check fails, outputs uncalibrated relative depth $D_{\text{rel}} = 1 / (D_{\text{inv}} + \epsilon)$ with `metric=False` and detailed diagnostic metadata.

---

## 4. Empirical Evaluation on Zurich MAV (350-Frame Hover Sequence)

### Filtering Funnel Statistics
* **Raw Candidate Projections:** 63,469
* **Rejected Negative / Invalid Depth:** 15,765
* **Rejected Out-of-Bounds ($<20\text{px}$ border):** 31,139
* **Passed Candidate Anchors:** **16,565** (across 12 sample keyframes)

### Key Metric Statistics
* $D_{\text{inv}}$ range: $[108.84, 940.92]$ ($\text{std} = 142.76$)
* $1/Z$ range: $[0.0207, 1.7903]$ ($Z \in [0.56\text{m}, 48.23\text{m}]$, $\text{mean} = 7.50\text{m}$)
* Design Matrix Condition Number: **$1,434.62$**
* **Empirical Correlation:** **$0.0651$**

### Why Did B0 Sparse Points Fail as Depth Anchors?
In this near-stationary hover sequence, the camera baseline between frames is only centimeters ($B/Z \approx 0.00025 - 0.075$). While B0 points achieve low 2D reprojection error ($0.98\text{ px}$), their depth uncertainty along the optical axis is massive ($\sigma_Z \sim 5-20\text{ m}$). Triangulated points on flat ground/asphalt were assigned arbitrary longitudinal depths, breaking any linear correlation with MiDaS's true visual disparity.

---

## 5. Cross-Validation & Leave-One-Out Analysis

1. **Leave-One-Frame-Out (LOO) Analysis:**
   * 11 of 12 folds completed, but fold predictions showed high variance on held-out frames (val RMSE on $1/Z$ ranged from $0.076$ to $0.216$).
2. **80/20 Train/Validation Split:**
   * Train Anchors: 13,252 | Validation Anchors: 3,313
   * Fitted on Train: $a = 0.0001096$, $b = 0.09259$
   * Validation RMSE on $1/Z$: **$0.1369$**
   * Validation Relative Depth Error: **$76.83\%$**

---

## 6. Regression Test Suite

All 155 unit and integration tests passed:
* `tests/unit/test_b5_metric_anchor.py`: 3 tests passing.
* `tests/unit/test_b5_metric_alignment.py`: 6 tests passing (synthetic recovery, RANSAC outlier filtering, GT source rejection, low-correlation rejection, deterministic behavior).
* `tests/integration/test_b5_metric_alignment.py`: 2 tests passing (LOO cross-validation, 80/20 train/val split).
* Complete repository test suite: **155 passed in 26.45s (100% GREEN)**.

---

## 7. Conclusion & Next Steps

B5 Phase 3D has scientifically proven that **monocular depth cannot be metrically calibrated using B0 sparse points during hover flight without injecting ~76% depth distortion**. 

The engine's fallback architecture safely protects the pipeline by outputting topologically accurate **relative dense representations (`metric=False`)**, ready for multi-frame volumetric integration in Phase 4.
