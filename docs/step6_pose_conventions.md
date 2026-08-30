# Step 6: Coordinate Systems, Reference Frames, and Pose Conventions

This document establishes the exact coordinate-frame definitions, transformation conventions, quaternion semantics, units, and spatial statistics for the **Zurich Urban MAV Dataset (Air-Ground Zurich AGZ)**, providing an unambiguous geometric foundation for downstream camera trajectory visualization, visual odometry, and 3D reconstruction.

---

## 1. Coordinate Systems & Reference Frames

### 1.1 World / Global Reference Frame ($\mathcal{F}_{\text{world}}$)
- **Convention**: **East-North-Up (ENU)** Cartesian frame rooted in the Universal Transverse Mercator (UTM Zone 32N) projection on the WGS84 ellipsoid.
- **Axes**:
  - **$+X$**: Pointing **East** (Easting in meters).
  - **$+Y$**: Pointing **North** (Northing in meters).
  - **$+Z$**: Pointing **Up** (Elevation AMSL in meters).
- **Handedness**: **Right-Handed** ($+X \times +Y = +Z$).
- **Origin**: Grid origin $(0, 0, 0)$ corresponds to the UTM Zone 32N baseline. Spatial coordinates in Zurich lie at $(X \approx 465,666\text{ m}, Y \approx 5,247,973\text{ m}, Z \approx 469\text{ m})$.
- **Evidence / Source**:
  - `GroundTruthAGL.csv` header: `x_gt, y_gt, z_gt` alongside `x_gps, y_gps, z_gps` in meters.
  - Coordinate values align with the Swiss Federal Office of Topography (swisstopo) / UTM projection for Zurich city centre ($47.384^\circ\text{ N}, 8.545^\circ\text{ E}$).

### 1.2 Body / Drone Reference Frame ($\mathcal{F}_{\text{body}}$)
- **Convention**: Standard Aerospace / Robotics **Forward-Left-Up (FLU)** body-fixed frame.
- **Axes**:
  - **$+X_{\text{body}}$**: Forward along the UAV longitudinal axis.
  - **$+Y_{\text{body}}$**: Left along the lateral axis.
  - **$+Z_{\text{body}}$**: Upward along the vertical axis.
- **Evidence / Source**:
  - `RawAccel.csv` records nominal gravity $z \approx -9.81\text{ m/s}^2$ to $-10.58\text{ m/s}^2$ when the drone is level on the ground, indicating $+Z$ points upward.
  - Forward velocity in `OnboardPose.csv` (`Vel_x`) increases during forward translation.

### 1.3 Camera Reference Frame ($\mathcal{F}_{\text{cam}}$)
- **Convention**: Standard Computer Vision / OpenCV / Photogrammetry frame:
  - **$+X_{\text{cam}}$**: Pointing **Right** across the image sensor.
  - **$+Y_{\text{cam}}$**: Pointing **Down** across the image sensor.
  - **$+Z_{\text{cam}}$**: Pointing **Forward** along the optical axis (principal ray).
- **Handedness**: **Right-Handed**.
- **Evidence / Source**:
  - `calibration_data.npz` stores the standard OpenCV pinhole calibration matrix $K = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}$ with positive focal lengths and principal point at $(951.13, 555.13)$.

---

## 2. Transformation Definitions & Pose Semantics

### 2.1 Camera-to-World ($T_{WC}$) vs. World-to-Camera ($T_{CW}$)
- **In `pose.csv`**:
  - The translation vector $\mathbf{t} = [t_x, t_y, t_z]^T$ represents the **position of the camera optical center expressed in World coordinates** (Camera-to-World translation $C_W$).
  - The rotation quaternion $\mathbf{q} = [q_x, q_y, q_z, q_w]^T$ represents the orientation of the vehicle/camera relative to the World ENU frame.
  - Explicit Transform:
    $$\mathbf{p}_{\text{world}} = \mathbf{R}_{WC} \mathbf{p}_{\text{cam}} + \mathbf{t}_{WC}$$
  - In our normalized `pose.csv`, $\mathbf{t} = \mathbf{t}_{WC}$ (meters in UTM).

### 2.2 Photogrammetric Euler Angles to Quaternion
- In the source `GroundTruthAGL.csv`, orientation is stored as photogrammetric Tait-Bryan angles: $\omega$ (omega / roll), $\phi$ (phi / pitch), $\kappa$ (kappa / yaw) in degrees.
- Transformation to Hamilton unit quaternion $\mathbf{q} = [q_x, q_y, q_z, q_w]$:
  - Yaw ($\kappa$ around $Z$), Pitch ($\phi$ around $Y$), Roll ($\omega$ around $X$).
  - Evaluated and verified: $\forall \mathbf{q}, \|\mathbf{q}\| = 1.0 \pm 10^{-4}$.

---

## 3. Quaternion Conventions & Ordering

| Context | Storage Order | Scalar Component Position | Hamilton / JPL |
| :--- | :--- | :--- | :--- |
| **Normalized `pose.csv`** | `qx, qy, qz, qw` | Scalar-Last ($w$ at index 3) | **Hamilton** ($i^2=j^2=k^2=ijk=-1$) |
| **Raw `OnboardPose.csv`** | `Attitude_w, Attitude_x, Attitude_y, Attitude_z` | Scalar-First ($w$ at index 0) | **Hamilton** |
| **PyTorch / SciPy / Open3D** | `qx, qy, qz, qw` | Scalar-Last | **Hamilton** |

---

## 4. GPS & Altitude Definitions

### 4.1 Geodetic Coordinate Convention
- **Datum**: WGS84 (EPSG:4326).
- **Latitude**: Geodetic Latitude in decimal degrees (North positive, Zurich: $47.38225^\circ\text{ N}$ to $47.38733^\circ\text{ N}$).
- **Longitude**: Geodetic Longitude in decimal degrees (East positive, Zurich: $8.54259^\circ\text{ E}$ to $8.54736^\circ\text{ E}$).

### 4.2 Altitude Definitions
1. **Ellipsoidal / AMSL Altitude (`alt` in `OnboardGPS.csv`)**:
   - Height above WGS84 reference ellipsoid / Mean Sea Level in meters.
   - Zurich baseline: $\sim 448.96\text{ m}$ to $519.30\text{ m AMSL}$.
2. **Barometric Altitude (`Altitude` in `BarometricPressure.csv`)**:
   - Pressure altitude derived from onboard barometer ($\sim 471.37\text{ m}$).
3. **Relative Flight Altitude (`Height` in `OnboardPose.csv`)**:
   - Altitude Above Ground Level (AGL) / height above takeoff point ($\sim 0.0\text{ m}$ to $30.0\text{ m}$).

---

## 5. Relationship Between Onboard Pose and Ground-Truth Pose

| Property | Onboard Pose (`OnboardPose.csv`) | Ground Truth Pose (`GroundTruthAGL.csv`) |
| :--- | :--- | :--- |
| **Source** | Real-time onboard state estimator | Offline photogrammetric bundle adjustment with GCPs |
| **Sampling Rate** | $\sim 50\text{ Hz}$ (135,098 rows) | $1\text{ Hz}$ (every 30th image frame, 2,708 rows) |
| **Position Accuracy**| Approximate dead-reckoning | Sub-meter metric ground truth (UTM meters) |
| **Orientation** | Real-time IMU/tether attitude ($q_w, q_x, q_y, q_z$) | High-accuracy photogrammetric angles ($\omega, \phi, \kappa$) |
| **Primary Usage** | High-frequency IMU fusion / rate control | Quantitative trajectory & 3D reconstruction benchmark |

---

## 6. Timestamp Relationship & Association

- **Image-to-GPS Association**: In `OnboardGPS.csv`, every record explicitly carries an integer `imgid` referencing the 30 FPS video frame sequence.
- **Ground-Truth Association**: In `GroundTruthAGL.csv`, each record contains `imgid` at 1 Hz intervals (`1, 31, 61, 91, ...`).
- **Association Method**:
  - For discrete image sets without frame tags, timestamps are associated via `TemporalSynchronizer` nearest-neighbor matching ($O(\log N)$ binary search).
  - All timestamps are converted to floating-point seconds: $t_{\text{seconds}} = \text{Timpstemp} / 10^6$.

---

## 7. Trajectory & Spatial Extent Statistics

The following empirical statistics were computed directly from the normalized dataset files:

### 7.1 Pose Trajectory Statistics (`pose.csv`)
- **Total Ground Truth Poses**: $2,708$ records.
- **Trajectory Start Time**: $7.009\text{ s}$
- **Trajectory End Time**: $2,707.033\text{ s}$
- **Total Duration**: $2,700.024\text{ s}$ ($\mathbf{45.00\text{ minutes}}$)
- **Spatial Extent**:
  - **$X$ (Easting)**: $[465,476.42\text{ m}, 465,834.27\text{ m}]$ $\implies \mathbf{\Delta X = 357.85\text{ m}}$
  - **$Y$ (Northing)**: $[5,247,742.01\text{ m}, 5,248,307.06\text{ m}]$ $\implies \mathbf{\Delta Y = 565.04\text{ m}}$
  - **$Z$ (Altitude)**: $[460.15\text{ m}, 488.96\text{ m}]$ $\implies \mathbf{\Delta Z = 28.81\text{ m}}$
- **Cumulative Trajectory Length**: $\mathbf{1,915.63\text{ m}}$ ($\mathbf{\approx 1.92\text{ km}}$)

### 7.2 GPS Spatial Statistics (`gps.csv`)
- **Total GPS Records**: $81,169$ records.
- **Latitude Span**: $[47.3822502^\circ\text{ N}, 47.3873331^\circ\text{ N}]$ ($\Delta\text{Lat} \approx 0.00508^\circ \approx 565\text{ m}$)
- **Longitude Span**: $[8.5425953^\circ\text{ E}, 8.5473580^\circ\text{ E}]$ ($\Delta\text{Lon} \approx 0.00476^\circ \approx 358\text{ m}$)
- **GPS Altitude Span**: $[448.96\text{ m}, 519.30\text{ m}]$ ($\Delta\text{Alt} = 70.34\text{ m}$)

---

## 8. Recommended Standard Representation for SIH26158

To maintain strict mathematical consistency across all pipeline modules (COLMAP, PyTorch, Open3D, and visual odometry), the following internal standards are adopted:

1. **Local Centered World Frame ($\mathcal{F}_{\text{local}}$)**:
   - Subtract the first trajectory position $\mathbf{t}_0 = [465666.06, 5247973.65, 469.02]^T$ from all UTM coordinates to prevent floating-point precision loss in 32-bit GPU computations.
   - Origin $(0, 0, 0)$ is the drone takeoff point in ENU meters.
2. **Camera Coordinate Convention**: Standard Computer Vision right-down-forward ($+X$ Right, $+Y$ Down, $+Z$ Optical Axis).
3. **Quaternion Storage**: Hamilton convention $[q_x, q_y, q_z, q_w]$ normalized to unit length.
4. **Time Representation**: Continuous floating-point elapsed seconds from mission zero ($t \ge 0.0\text{ s}$).
