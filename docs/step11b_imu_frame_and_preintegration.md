# Step 11B: IMU Frame/Sign Validation & Discrete Preintegration Foundation

This report documents the rigorous mathematical frame verification, scale factor audit, and standalone on-manifold IMU preintegration engine developed for **Baseline B2** (Visual + GPS + IMU Trajectory Fusion) on the **Zurich Urban MAV Dataset**.

---

## 1. Verified Sensor Semantics & Measurement Models

### 1.1 Accelerometer Semantics
The raw accelerometer measurements represent **specific force** ($\mathbf{f}$), which is the difference between kinematic vehicle acceleration and Earth gravity:
$$\mathbf{f}_{\text{body}} = \mathbf{a}_{\text{motion}} - \mathbf{g}_{\text{body}} + \mathbf{b}_{\text{accel}} + \boldsymbol{\eta}_{\text{accel}}$$

* When the vehicle is resting stationary on the takeoff pad ($\mathbf{a}_{\text{motion}} = \mathbf{0}$), the sensor measures the **upward reaction force opposing gravity** ($-\mathbf{g}_{\text{body}}$).

### 1.2 Gyroscope Semantics
The raw gyroscope measurements represent **instantaneous 3-axis angular velocity** ($\boldsymbol{\omega}_{\text{body}}$) around the body principal axes:
$$\boldsymbol{\omega}_{\text{measured}} = \boldsymbol{\omega}_{\text{true}} + \mathbf{b}_{\text{gyro}} + \boldsymbol{\eta}_{\text{gyro}}$$

---

## 2. Native Frame Verification (FRD vs FLU)

* **Verified Native Sensor Frame**: **`Forward-Right-Down (FRD / NED Body Frame)`**
  * $+X_{\text{FRD}}$ points along the vehicle forward nose.
  * $+Y_{\text{FRD}}$ points along the right wing / arm.
  * $+Z_{\text{FRD}}$ points downward towards Earth.
* **Physical Gravity Consistency**:
  * In FRD, gravity vector $\mathbf{g}_{\text{world}} = [0, 0, +9.81]^T\text{ m/s}^2$ points down.
  * The reaction force opposing gravity is directed upwards along $-Z_{\text{FRD}}$, producing the observed stationary reading:
    $$\mathbf{a}_{\text{stationary, FRD}} \approx \begin{bmatrix} -0.1638 \\ -0.1654 \\ -9.1785 \end{bmatrix}\text{ m/s}^2$$
* **Conversion to Internal Robotics Frame (Forward-Left-Up / FLU)**:
  $$\mathbf{a}_{\text{FLU}} = \begin{bmatrix} 1 & 0 & 0 \\ 0 & -1 & 0 \\ 0 & 0 & -1 \end{bmatrix} \mathbf{a}_{\text{FRD}} = \begin{bmatrix} a_x \\ -a_y \\ -a_z \end{bmatrix}$$
  $$\boldsymbol{\omega}_{\text{FLU}} = \begin{bmatrix} 1 & 0 & 0 \\ 0 & -1 & 0 \\ 0 & 0 & -1 \end{bmatrix} \boldsymbol{\omega}_{\text{FRD}} = \begin{bmatrix} \omega_x \\ -\omega_y \\ -\omega_z \end{bmatrix}$$
  * In FLU, stationary upward reaction force becomes positive ($a_{z, \text{FLU}} \approx +9.18\text{ m/s}^2$), matching standard robotics and GTSAM conventions.

---

## 3. Scale Factor & Raw Extraction Validation

Verified against raw hardware ADC registers and dataset log headers (`RawAccel.csv`, `RawGyro.csv`):

| Channel | Physical Range | Bit Width / Scaling | Verification Formula |
| :--- | :---: | :---: | :--- |
| **Accelerometer** | $\pm 156.9064\text{ m/s}^2$ ($\pm 16g$) | $0.0047884034\text{ m/s}^2/\text{LSB}$ | $\text{accel} = \text{accel\_raw} \times \text{scaling}$ |
| **Gyroscope** | $\pm 34.90658\text{ rad/s}$ ($\pm 2000^\circ/\text{s}$) | $0.0010642195\text{ rad/s/LSB}$ | $\text{gyro} = \text{gyro\_raw} \times \text{scaling}$ |

---

## 4. Stationary Gyro Offset & Orientation Drift Analysis

* **Observed Stationary Gyro Offset**:
  $$\boldsymbol{\omega}_{\text{offset}} = [+0.011275, -0.039672, -0.024465]^T\text{ rad/s} \quad (\|\boldsymbol{\omega}_{\text{offset}}\| = 0.04797\text{ rad/s} \approx 2.748^\circ/\text{s})$$

### Naïve Integrated Orientation Drift Projection (Uncorrected Gyroscope):
$$\Delta \theta(T) = \|\boldsymbol{\omega}_{\text{offset}}\| \cdot T$$

| Integration Duration ($T$) | Predicted Attitude Drift (deg) | Predicted Attitude Drift (rad) | Operational Impact |
| :---: | :---: | :---: | :--- |
| **$1\text{ second}$** | **`2.75°`** | `0.0480 rad` | Negligible for intra-frame tracking |
| **$10\text{ seconds}$** | **`27.48°`** | `0.4797 rad` | Severe heading rotation |
| **$30\text{ seconds}$** | **`82.43°`** | `1.4391 rad` | Catastrophic attitude inversion |
| **$60\text{ seconds}$** | **`164.85°`** | `2.8782 rad` | Complete loss of orientation |

> [!IMPORTANT]
> The orientation drift analysis proves that standalone IMU dead-reckoning is fundamentally unstable over multi-second horizons. B2 sensor fusion must continuously estimate and correct gyroscope bias against visual feature tracks and GPS heading.

---

## 5. Discrete On-Manifold IMU Preintegration Engine

Implemented in `src/sensor_fusion/imu_preintegration.py` using double-precision arithmetic and actual non-uniform sample intervals ($\Delta t_i = t_{i+1} - t_i$):

### 5.1 Discrete Update Equations (Forster et al. Formulation)
$$\Delta \mathbf{R}_{i+1} = \Delta \mathbf{R}_i \cdot \text{Exp}((\boldsymbol{\omega}_i - \mathbf{b}_g) \Delta t_i)$$
$$\Delta \mathbf{v}_{i+1} = \Delta \mathbf{v}_i + \Delta \mathbf{R}_i (\mathbf{a}_i - \mathbf{b}_a) \Delta t_i$$
$$\Delta \mathbf{p}_{i+1} = \Delta \mathbf{p}_i + \Delta \mathbf{v}_i \Delta t_i + \frac{1}{2} \Delta \mathbf{R}_i (\mathbf{a}_i - \mathbf{b}_a) \Delta t_i^2$$

### 5.2 World Navigation State Prediction
$$\mathbf{R}_j = \mathbf{R}_i \cdot \Delta \mathbf{R}$$
$$\mathbf{v}_j = \mathbf{v}_i + \mathbf{g}_{\text{world}} \Delta T + \mathbf{R}_i \Delta \mathbf{v}$$
$$\mathbf{p}_j = \mathbf{p}_i + \mathbf{v}_i \Delta T + \frac{1}{2} \mathbf{g}_{\text{world}} \Delta T^2 + \mathbf{R}_i \Delta \mathbf{p}$$

---

## 6. Real-Data Preintegration Sanity Results

Preintegration was executed over 3 representative flight intervals in the Zurich dataset:

| Flight Segment | Time Interval (s) | IMU Samples | $\Delta \theta$ (deg) | $\Delta v$ (m/s) | $\Delta p$ (m) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Takeoff Ground Dwell** | $[7.091, 8.091]\text{ s}$ | 11 | `2.78°` | `9.19 m/s` | `4.59 m` |
| **Initial Climb Phase** | $[10.091, 12.091]\text{ s}$ | 21 | `8.41°` | `18.62 m/s` | `18.51 m` |
| **Corridor Flight Phase** | $[14.091, 16.091]\text{ s}$ | 21 | `5.92°` | `18.49 m/s` | `18.39 m` |

*Note: Velocity and displacement magnitudes reflect integrated specific force before subtracting world gravity $\mathbf{g}_{\text{world}} \Delta T$.*

---

## 7. Deliverables Inventory

| Deliverable File | Type | Description |
| :--- | :---: | :--- |
| [src/pose/imu_frames.py](file:///d:/SIH26158-single-pass-3D/src/pose/imu_frames.py) | Python | Explicit, reversible FRD $\leftrightarrow$ FLU frame transformation module |
| [src/sensor_fusion/imu_types.py](file:///d:/SIH26158-single-pass-3D/src/sensor_fusion/imu_types.py) | Python | Data classes for `IMUMeasurement` and `PreintegratedNavState` |
| [src/sensor_fusion/imu_preintegration.py](file:///d:/SIH26158-single-pass-3D/src/sensor_fusion/imu_preintegration.py) | Python | Standalone on-manifold discrete preintegration and state prediction module |
| [outputs/reports/zurich_mav/b2/imu_frame_validation.json](file:///d:/SIH26158-single-pass-3D/outputs/reports/zurich_mav/b2/imu_frame_validation.json) | JSON | Frame validation report, scale verification, and 60-second drift analysis |
| [outputs/reports/zurich_mav/b2/imu_preintegration_sanity.json](file:///d:/SIH26158-single-pass-3D/outputs/reports/zurich_mav/b2/imu_preintegration_sanity.json) | JSON | Real-data preintegration sanity execution metrics over flight intervals |
| [docs/step11b_imu_frame_and_preintegration.md](file:///d:/SIH26158-single-pass-3D/docs/step11b_imu_frame_and_preintegration.md) | Docs | Technical documentation of sensor semantics, frames, drift, and preintegration |
| [tests/unit/test_imu_frames.py](file:///d:/SIH26158-single-pass-3D/tests/unit/test_imu_frames.py) | Test | Unit tests for FRD $\leftrightarrow$ FLU conversions and gravity direction invariants |
| [tests/unit/test_imu_preintegration.py](file:///d:/SIH26158-single-pass-3D/tests/unit/test_imu_preintegration.py) | Test | Unit tests for mathematical preintegration invariants with irregular $\Delta t$ |
| [tests/integration/test_b2_imu_preintegration.py](file:///d:/SIH26158-single-pass-3D/tests/integration/test_b2_imu_preintegration.py) | Test | Integration tests validating frame validation and sanity JSON reports |
