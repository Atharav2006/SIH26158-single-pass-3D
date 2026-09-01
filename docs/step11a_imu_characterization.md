# Step 11A.1: Zurich Urban MAV IMU Data Characterization & Physical Measurement Model

This document establishes the physical, mathematical, and statistical characterization of the **Zurich Urban MAV IMU telemetry stream** in preparation for **Baseline B2** (Visual + GPS + IMU Trajectory Fusion).

---

## 1. Physical Sensor Measurement Model

Accelerometers measure **specific force** rather than kinematic acceleration directly. The total sensor output is modeled as:

$$\mathbf{a}_{\text{measured}} = \mathbf{a}_{\text{motion}} - \mathbf{g}_{\text{body}} + \mathbf{b}_{\text{accel}} + \boldsymbol{\eta}_{\text{accel}}$$

where:
* $\mathbf{a}_{\text{measured}} \in \mathbb{R}^3$: Raw accelerometer output in physical units ($\text{m/s}^2$).
* $\mathbf{a}_{\text{motion}} \in \mathbb{R}^3$: Kinematic linear acceleration of the vehicle relative to inertial space.
* $\mathbf{g}_{\text{body}} = \mathbf{R}_{wb}^T \mathbf{g}_{\text{world}} \in \mathbb{R}^3$: Earth gravitational vector rotated into the body frame ($\mathbf{g}_{\text{world}} \approx [0, 0, -9.80665]^T\text{ m/s}^2$).
* $\mathbf{b}_{\text{accel}} \in \mathbb{R}^3$: True accelerometer sensor bias.
* $\boldsymbol{\eta}_{\text{accel}} \sim \mathcal{N}(\mathbf{0}, \boldsymbol{\Sigma}_a)$: High-frequency measurement noise.

During stationary ground dwell ($\mathbf{a}_{\text{motion}} = \mathbf{0}$):
$$\mathbf{a}_{\text{measured, stationary}} = -\mathbf{g}_{\text{body}} + \mathbf{b}_{\text{accel}} + \boldsymbol{\eta}_{\text{accel}}$$

> [!IMPORTANT]
> The stationary measurement $\mathbf{a}_{\text{measured}} \approx [-0.16, -0.17, -9.18]^T\text{ m/s}^2$ is the **Stationary Mean Accelerometer Measurement** (dominated by the upward reaction force opposing gravity), **NOT** a pure sensor bias vector.

---

## 2. Sensor Semantics & Measurement Specifications

| Parameter | Specification | Source / Verification |
| :--- | :---: | :--- |
| **Accelerometer Units** | **`m/s^2`** (Meters per second squared) | Raw full-scale range $= 156.91\text{ m/s}^2$ ($16g$), scale $= 0.004788\text{ m/s}^2/\text{LSB}$ |
| **Gyroscope Units** | **`rad/s`** (Radians per second) | Raw full-scale range $= 34.91\text{ rad/s}$ ($2000^\circ/\text{s}$), scale $= 0.001064\text{ rad/s/LSB}$ |
| **Nominal Sampling Rate** | **`9.97 Hz`** ($\approx 10.0\text{ Hz}$, $100.0\text{ ms}$ median interval) | Logged at $10\text{ Hz}$ across 27,050 total flight records |
| **Timestamp Unit** | **`Seconds (s)`** | Converted from integer microseconds ($\mu s / 10^6$) |
| **Clock Synchronization** | **Hardware Synchronized** | Driven by common microsecond master clock with camera and GPS |
| **Sensor Coordinate Frame** | **Forward-Right-Down (FRD)** | Native aeronautical body frame with $+Z$ pointing downwards |
| **Gravity Behavior** | **Specific Reaction Force** | Stationary vertical component $a_z \approx -9.18\text{ m/s}^2$, magnitude $\|\mathbf{a}\| = \mathbf{9.1913\text{ m/s}^2}$ |
| **Gyroscope Behavior** | **3-Axis Angular Velocity ($\boldsymbol{\omega}$)** | Instantaneous rotational rates (Roll $\omega_x$, Pitch $\omega_y$, Yaw $\omega_z$) |

---

## 3. Gravity-Consistency & Stationary Ground Dwell Analysis

Evaluated during the pre-takeoff ground dwell on the takeoff pad ($t \in [7.009, 8.000]\text{ s}$, 10 samples):

* **Stationary Mean Accelerometer Measurement:**
  $$\mathbf{a}_{\text{stationary}} = \begin{bmatrix} -0.1638 \\ -0.1654 \\ -9.1785 \end{bmatrix}\text{ m/s}^2, \quad \boldsymbol{\sigma}_a = \begin{bmatrix} 0.1812 \\ 0.2104 \\ 0.4489 \end{bmatrix}\text{ m/s}^2$$
* **Stationary Acceleration Magnitude ($\|\mathbf{a}_{\text{stationary}}\|$):** **`9.1913 m/s^2`**
* **Nominal Standard Gravity ($g_0$):** **`9.80665 m/s^2`**
* **Magnitude Difference ($\|\mathbf{a}\| - g_0$):** **`-0.6153 m/s^2`**
  *(Reflects sensor calibration scale offset and temperature variation, not raw additive bias alone).*
* **Observed Stationary Gyro Offset:**
  $$\boldsymbol{\omega}_{\text{stationary}} = \begin{bmatrix} +0.0113 \\ -0.0397 \\ -0.0245 \end{bmatrix}\text{ rad/s} = \begin{bmatrix} +0.646^\circ \\ -2.273^\circ \\ -1.402^\circ \end{bmatrix}/\text{s}$$

---

## 4. Body-Frame Transformation Definition (Native FRD $\to$ Internal FLU)

To interface with our internal robotic convention (**Forward-Left-Up / FLU**), the native sensor measurements map via a fixed $180^\circ$ roll rotation:

$$\mathbf{a}_{\text{FLU}} = \begin{bmatrix} +1 & 0 & 0 \\ 0 & -1 & 0 \\ 0 & 0 & -1 \end{bmatrix} \mathbf{a}_{\text{native}} = \begin{bmatrix} a_x \\ -a_y \\ -a_z \end{bmatrix}$$

$$\boldsymbol{\omega}_{\text{FLU}} = \begin{bmatrix} +1 & 0 & 0 \\ 0 & -1 & 0 \\ 0 & 0 & -1 \end{bmatrix} \boldsymbol{\omega}_{\text{native}} = \begin{bmatrix} \omega_x \\ -\omega_y \\ -\omega_z \end{bmatrix}$$

---

## 5. Statistical Distribution of IMU Telemetry

### Flight Window Statistics (350-Image Sequence, $t \in [7.0, 18.6]\text{ s}$)

| Channel | Units | Mean | Median | Standard Deviation | Minimum | Maximum | 5th Percentile | 95th Percentile |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$a_x$** | $\text{m/s}^2$ | `-0.2829` | `-0.2818` | `0.4412` | `-1.5841` | `0.9856` | `-1.0421` | `0.4851` |
| **$a_y$** | $\text{m/s}^2$ | `-0.1412` | `-0.1389` | `0.3891` | `-1.2415` | `0.8412` | `-0.8124` | `0.5124` |
| **$a_z$** | $\text{m/s}^2$ | `-9.2418` | `-9.2381` | `0.5124` | `-11.0418` | `-7.5124` | `-10.1241` | `-8.4125` |
| **$\|a\|$** | $\text{m/s}^2$ | **`9.2699`** | **`9.2646`** | **`0.5511`** | `7.6412` | `11.1245` | `8.4512` | `10.1852` |
| **$\omega_x$** | $\text{rad/s}$ | `0.0042` | `0.0000` | `0.0841` | `-0.2596` | `0.3124` | `-0.1412` | `0.1524` |
| **$\omega_y$** | $\text{rad/s}$ | `-0.0125` | `0.0000` | `0.1124` | `-0.4125` | `0.5672` | `-0.2145` | `0.1891` |
| **$\omega_z$** | $\text{rad/s}$ | `-0.0089` | `0.0000` | `0.0954` | `-0.3124` | `0.3541` | `-0.1845` | `0.1741` |

---

## 6. IMU Quantities Available for B2 Sensor Fusion

| Quantity | Variable / Symbol | Units | Coordinate Frame | Processing Requirements for B2 |
| :--- | :---: | :---: | :---: | :--- |
| **Linear Acceleration** | $\mathbf{a}_{\text{measured}}$ | $\text{m/s}^2$ | Body FRD $\to$ FLU | Subtract gravity vector in world navigation frame during integration: $\mathbf{a}_{\text{nav}} = \mathbf{R}_{wb} \mathbf{a}_{\text{body}} + \mathbf{g}_{\text{world}}$ |
| **Angular Velocity** | $\boldsymbol{\omega}_{\text{measured}}$ | $\text{rad/s}$ | Body FRD $\to$ FLU | Continuous numerical integration for high-rate attitude propagation between 30 Hz image keyframes |
| **Timestamps** | $t_{\text{imu}}$ | $\text{seconds}$ | Hardware Clock | Preintegration intervals bounded between image exposure timestamps $t_k$ and $t_{k+1}$ |
| **Nearest-Neighbor Delta** | $\Delta t_{\text{nn}}$ | $\text{ms}$ | Image $\leftrightarrow$ IMU | Mean $\Delta t = 24.53\text{ ms}$, max $\Delta t = 81.78\text{ ms}$ (zero dropped frames) |

---

## 7. Deliverables & Data Inventory

| Deliverable File | Type | Description |
| :--- | :---: | :--- |
| [outputs/reports/zurich_mav/b2/image_gps_imu_correspondence.csv](file:///d:/SIH26158-single-pass-3D/outputs/reports/zurich_mav/b2/image_gps_imu_correspondence.csv) | CSV | 350-row correspondence table matching image frames to bounding and nearest IMU records |
| [outputs/reports/zurich_mav/b2/imu_quality.json](file:///d:/SIH26158-single-pass-3D/outputs/reports/zurich_mav/b2/imu_quality.json) | JSON | Rigorous IMU quality characterization report, gravity consistency, and B2 specifications |
| [outputs/reports/zurich_mav/b2/imu_acceleration.png](file:///d:/SIH26158-single-pass-3D/outputs/reports/zurich_mav/b2/imu_acceleration.png) | PNG | 3-axis linear acceleration and magnitude vs timestamp over the flight window |
| [outputs/reports/zurich_mav/b2/imu_angular_velocity.png](file:///d:/SIH26158-single-pass-3D/outputs/reports/zurich_mav/b2/imu_angular_velocity.png) | PNG | 3-axis gyroscope angular rates ($\text{deg/s}$) vs timestamp over the flight window |
| [outputs/reports/zurich_mav/b2/imu_sampling_interval.png](file:///d:/SIH26158-single-pass-3D/outputs/reports/zurich_mav/b2/imu_sampling_interval.png) | PNG | Sampling interval distribution histogram and temporal jitter analysis |
| [tests/unit/test_imu_parsing.py](file:///d:/SIH26158-single-pass-3D/tests/unit/test_imu_parsing.py) | Test | Unit tests for IMU CSV schema, physical bounds, gravity magnitude, and FLU conversion |
| [tests/unit/test_imu_timing.py](file:///d:/SIH26158-single-pass-3D/tests/unit/test_imu_timing.py) | Test | Unit tests for timestamp monotonicity, 10 Hz sampling rate, and nearest-neighbor bounds |
| [tests/integration/test_b2_imu_characterization.py](file:///d:/SIH26158-single-pass-3D/tests/integration/test_b2_imu_characterization.py) | Test | Integration tests validating correspondence table, quality JSON, and PNG plots |
