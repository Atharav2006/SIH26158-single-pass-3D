# Step 9A: GPS Quality & Metric-Anchor Analysis (B1 Baseline Preparation)

This document presents the GPS quality and geospatial characterization for **Baseline B1** (COLMAP Structure-from-Motion + UAV GPS Metric/Geospatial Anchoring) on the **Zurich Urban MAV Dataset**.

---

## 1. Objective & Architectural Scope

```
                   350 UAV images
                         │
                         ▼
                    COLMAP B0
                         │
                         ▼
                Camera centers C_colmap
                         │
                         │
                         │
                         ▼
                 GPS image positions
                         │
                         ▼
                Coordinate conversion
                         │
                         ▼
                     UTM / ENU
                         │
                         ▼
                 Correspondence pairs
                         │
                         ▼
                    Sim(3) estimation
                         │
              ┌──────────┼───────────┐
              ▼          ▼           ▼
            Scale     Rotation   Translation
              │          │           │
              └──────────┼───────────┘
                         ▼
                Metric camera trajectory
                         │
                         ▼
                  Georeferenced model
```

* **Purpose**: Characterize the continuous 30 Hz GPS telemetry stream and verify its suitability as a metric scale and geospatial orientation anchor for B1.
* **Strict Evaluation Isolation**:
  * No modification to the COLMAP B0 reconstruction.
  * No ground truth pose was used to derive or optimize the GPS conversions or relationships.
  * No final B1 transform or scale is applied in this stage (characterization only).

---

## 2. Image-GPS Association

* **Association Strategy**: Exact dataset-native `imgid` correspondence (1:1 with the 30 Hz synchronized telemetry log).
* **Total Image Frames Evaluated**: **`350 / 350`** (**`100.0%`**)
* **Timestamp Alignment**: Mean sampling interval $\Delta t = 0.033333\text{ s}$ ($30.0\text{ Hz}$). Maximum timestamp difference between camera and GPS log: **`0.000000 s`** (exact synchronized triggering).

---

## 3. Coordinate System & Geodetic Projections

### 3.1 Projection Formulation: WGS 84 $\to$ UTM Zone 32N (EPSG:32632)
Geodetic coordinates $(\phi, \lambda, h)$ are projected to Universal Transverse Mercator (UTM) Zone 32N using the standard Karney / Transverse Mercator formulation:
* **Ellipsoid**: WGS 84 ($a = 6378137.0\text{ m}, f = 1/298.257223563$)
* **Central Meridian ($\lambda_0$)**: $9.0^\circ\text{ E}$ (Zone 32)
* **Scale Factor ($k_0$)**: $0.9996$
* **False Easting**: $500,000.0\text{ m}$, **False Northing**: $0.0\text{ m}$

### 3.2 Local Metric Frame: Centered East-North-Up (ENU)
To ensure numerical stability and local metric interpretability:
$$\mathbf{p}_{\text{local}} = [E - E_0, N - N_0, U - U_0]^T$$
* **Reference Origin (Frame 1 GPS)**:
  * Latitude: $47.3843571^\circ\text{ N}$, Longitude: $8.5451784^\circ\text{ E}$, Altitude: $464.91\text{ m}$
  * Projected UTM Origin: $E_0 = 465,670.7068\text{ m}, N_0 = 5,247,978.0338\text{ m}, U_0 = 464.91\text{ m}$

---

## 4. GPS Stream Statistics & Quality Analysis

| Metric Category | Parameter | Measured Value | Unit |
| :--- | :--- | :---: | :---: |
| **Stream Volume** | Total Dataset GPS Fixes | **`81,169`** | Records |
| | B0 Associated Fixes | **`350`** | Records |
| | Sampling Frequency | **`30.0`** | Hz |
| **Geodetic Extents** | Latitude Range | `[47.3843563°, 47.3843807°]` | Degrees |
| | Longitude Range | `[8.5451774°, 8.5452293°]` | Degrees |
| | Altitude Range | `[464.91, 466.87]` | Meters |
| **Projected Metric Extents** | Local East Span ($\Delta E$) | **`3.84`** | Meters |
| | Local North Span ($\Delta N$) | **`2.72`** | Meters |
| | Local Altitude Span ($\Delta U$) | **`1.96`** | Meters |
| **Kinematic Steps** | Horizontal Step (Mean) | **`0.0242`** | Meters/frame |
| | Horizontal Step (Median) | **`0.0104`** | Meters/frame |
| | Horizontal Step (95th %) | **`0.0898`** | Meters/frame |
| | Vertical Step (Mean) | **`0.0056`** | Meters/frame |
| | Vertical Step (Max) | **`0.1400`** | Meters/frame |
| **Kinematic Speeds** | Horizontal Velocity (Mean) | **`0.72`** | m/s |
| | Horizontal Velocity (Max) | **`3.25`** | m/s |
| | Vertical Velocity (Mean) | **`0.17`** | m/s |

---

## 5. Observed GPS Variability & Noise Characteristics

* **Takeoff Pad Stationary Noise (Frames 1–30)**:
  * Horizontal Jitter ($\sigma_{\text{horiz}}$): **`0.0232 m`** ($2.3\text{ cm}$)
  * Vertical Jitter ($\sigma_{\text{vert}}$): **`0.0000 m`** (Quantized barometric / GNSS altitude hold)
* **Overall Stream Integrity**:
  * Duplicate Timestamps: **`0`**
  * Missing Epochs: **`0`**
  * Statistical Step Outliers ($> 3\sigma$): **`0`**

---

## 6. GPS $\leftrightarrow$ COLMAP Descriptive Relationship (Pre-Alignment)

| Metric | Raw GPS (Metric) | Raw COLMAP B0 (Reconstructed) | Ratio / Scale Indicator |
| :--- | :---: | :---: | :---: |
| **Associated Trajectory Fixes** | 350 frames | 350 cameras | 1:1 correspondences |
| **Cumulative Path Length** | **`9.13 m`** | **`18.22 units`** | Rough scale: **`0.500920 m/unit`** |
| **Spatial Extents ($X, Y, Z$)** | $[3.84, 2.72, 1.96]\text{ m}$ | $[1.46, 17.51, 0.44]\text{ units}$ | Anisotropic flight corridor |

---

## 7. Recommendations for B1 Metric Anchoring

1. **Continuous 30 Hz GPS Anchoring**: All 350 frames have valid, synchronized GPS fixes. B1 can anchor every camera station rather than relying solely on sparse keyframes.
2. **Robust $\text{Sim}(3)$ Transformation**: Because standalone commercial GNSS receivers have global offset biases ($\sim 3-5\text{ m}$ relative to surveyed ground truth), B1 should fit a global 7-DoF similarity transform (scale $s$, rotation $R$, translation $t$) across all 350 image-GPS pairs to align the trajectory and sparse point cloud into the real-world UTM Zone 32N coordinate frame.
3. **No Filtering Required for Initial Anchor**: The GPS stream has zero discontinuous spikes in this 350-frame sequence, confirming clean input data for B1.

---

## 8. Visualizations & Deliverables

* **Quality Report JSON**: [outputs/reports/zurich_mav/b1/gps_quality.json](file:///d:/SIH26158-single-pass-3D/outputs/reports/zurich_mav/b1/gps_quality.json)
* **Outliers Log CSV**: [outputs/reports/zurich_mav/b1/gps_outliers.csv](file:///d:/SIH26158-single-pass-3D/outputs/reports/zurich_mav/b1/gps_outliers.csv)
* **Pre-Alignment Comparison Plot**: [outputs/reports/zurich_mav/b1/gps_vs_colmap_raw.png](file:///d:/SIH26158-single-pass-3D/outputs/reports/zurich_mav/b1/gps_vs_colmap_raw.png)
* **Local ENU Flight Path & Altitude Plot**: [outputs/reports/zurich_mav/b1/gps_trajectory_local.png](file:///d:/SIH26158-single-pass-3D/outputs/reports/zurich_mav/b1/gps_trajectory_local.png)
