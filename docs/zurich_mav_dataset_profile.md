# Zurich Urban MAV Dataset Profile

This document profiles the raw structure, contents, coordinate conventions, sensor streams, and telemetry of the **Zurich Urban MAV Dataset (Air-Ground Zurich AGZ)**, based on direct empirical inspection of the acquired dataset at `D:\SIH26158\datasets\zurich_mav`.

---

## 1. General Dataset Information

- **Dataset Name**: Zurich Urban Micro Aerial Vehicle Dataset (Air-Ground Zurich / AGZ)
- **Official Provider**: Robotics and Perception Group (RPG), University of Zurich & ETH Zurich (Andras Majdik, Yves Albers-Schoenberg, Davide Scaramuzza)
- **Official URL**: [http://rpg.ifi.uzh.ch/zurichmavdataset.html](http://rpg.ifi.uzh.ch/zurichmavdataset.html)
- **License**: Unrestricted open research and commercial use.
- **Reference Paper**: *Air-ground correspondence and collision-free navigation of a quadrotor in urban street canyons* (Majdik et al., Journal of Field Robotics 2015 / IROS 2013).
- **Physical Dataset Path**: `D:\SIH26158\datasets\zurich_mav\AGZ_subset`

---

## 2. Discovered Directory and File Structure

```
D:\SIH26158\datasets\zurich_mav\AGZ_subset/
├── AGZ.bag                               # Synchronized ROS bag file containing all topics
├── calibration_data.npz                  # NumPy archive containing intrinsic matrix & distortion coefficients
├── loadGroundTruthAGL.m                  # MATLAB script for importing GroundTruthAGL.csv
├── Log Files/                            # Raw comma-separated telemetry and sensor logs
│   ├── BarometricPressure.csv            # Barometric pressure, altitude, and temperature logs (27,052 rows)
│   ├── GroundTruthAGL.csv                # Aerial ground location ground truth in UTM/Swiss grid (2,708 rows)
│   ├── GroundTruthAGM.csv                # Aerial-Ground matches / correspondence IDs (81,169 rows)
│   ├── OnboardGPS.csv                    # Onboard GNSS receiver logs (81,169 rows)
│   ├── OnboardPose.csv                   # Onboard estimated state, velocities, and attitude (135,098 rows)
│   ├── RawAccel.csv                      # Raw 3-axis accelerometer sensor logs (27,050 rows)
│   ├── RawGyro.csv                       # Raw 3-axis rate gyroscope sensor logs (27,050 rows)
│   └── StreetViewGPS.csv                 # Ground Google Street View query poses (113 rows)
├── MAV Images/                           # Directory reserved for extracted aerial RGB frames
└── MAV Images Calib/                     # Synchronized camera calibration checkerboard images (30 PNG images)
```

---

## 3. Sensor Streams and Data Format Analysis

### 3.1 Timestamp Conventions
- **Timestamp Field Name**: `Timpstemp` (in log files).
- **Native Unit**: **Microseconds** ($\mu\text{s}$) represented as integer values from system start (e.g., `7009129`, `7042462` $\implies \Delta t \approx 33,333\mu\text{s} \approx 30.0\text{ Hz}$).
- **Normalized Representation**: Converted to standard floating-point seconds:
  $$t_{\text{seconds}} = \frac{\text{Timpstemp}}{1\,000\,000.0}$$

### 3.2 GPS / GNSS Log (`OnboardGPS.csv`)
- **Row Count**: 81,169 records (sampling rate: ~30 Hz).
- **Header**: `Timpstemp, imgid, lat, lon, alt, s_variance_m_s, c_variance_rad, fix_type, eph_m, epv_m, vel_n_m_s, vel_e_m_s, vel_d_m_s, num_sat`
- **Fields**:
  - `lat`, `lon`: WGS84 Geodetic Coordinates (degrees, e.g. `47.3843571`, `8.5451784`).
  - `alt`: WGS84 Ellipsoidal/MSL altitude in meters (e.g. `464.91 m`).
  - `imgid`: 1-based synchronized frame ID mapping directly to the video frame index.
  - `fix_type`: GNSS fix quality indicator (3 = 3D Fix).
  - `num_sat`: Number of satellites tracked (e.g. 6–10).

### 3.3 IMU Sensor Logs (`RawAccel.csv` & `RawGyro.csv`)
- **Row Count**: 27,050 records each (~10 Hz telemetry logging rate).
- **Header**: `Timpstemp, Error_count, x, y, z, temperature, range_rad_s, scaling, x_raw, y_raw, z_raw, temperature_raw`
- **Acceleration Units**: $x, y, z$ in $\text{m/s}^2$ (including gravity, $z \approx -10.58\text{ m/s}^2$).
- **Gyroscope Units**: $x, y, z$ in $\text{rad/s}$ (angular velocities).

### 3.4 Onboard Pose and State (`OnboardPose.csv`)
- **Row Count**: 135,098 records (~50 Hz state estimator).
- **Header**: `Timpstemp, Omega_x, Omega_y, Omega_z, Accel_x, Accel_y, Accel_z, Vel_x, Vel_y, Vel_z, AccBias_x, AccBias_y, AccBias_z, Azimuth, Attitude_w, Attitude_x, Attitude_y, Attitude_z, Height, Altitude, veh_pitch, Tether_angle, Tether_angle_dot, Tether_force, GPS_on`
- **Orientation**: Unit quaternion $[q_w, q_x, q_y, q_z]$ (`Attitude_w, Attitude_x, Attitude_y, Attitude_z`).
- **Altitude**: `Altitude` (meters AMSL) and relative flight `Height` (meters AGL).

### 3.5 Ground Truth Aerial Location (`GroundTruthAGL.csv`)
- **Row Count**: 2,708 records (sampled at 1 Hz, every 30th image frame: `imgid = 1, 31, 61, ...`).
- **Header**: `imgid, x_gt, y_gt, z_gt, omega_gt, phi_gt, kappa_gt, x_gps, y_gps, z_gps`
- **Position ($x_{\text{gt}}, y_{\text{gt}}, z_{\text{gt}}$)**: Metric Cartesian coordinates in the Swiss Grid (CH1903+ / LV95 projection) or local UTM (meters).
- **Orientation ($\omega_{\text{gt}}, \phi_{\text{gt}}, \kappa_{\text{gt}}$)**: Photogrammetric Tait-Bryan angles (roll, pitch, yaw) in degrees.

---

## 4. Camera Intrinsics and Calibration

From `calibration_data.npz`:
- **Camera Model**: Pinhole with Brown-Conrady 5-parameter radial/tangential distortion ($k_1, k_2, p_1, p_2, k_3$).
- **Resolution**: $1920 \times 1080$ pixels (1080p).
- **Focal Lengths**:
  - $f_x = 893.39010814$ pixels
  - $f_y = 898.32648616$ pixels
- **Principal Point**:
  - $c_x = 951.13100430$ pixels
  - $c_y = 555.13350077$ pixels
- **Distortion Coefficients**:
  $$[-0.28052513, 0.115806413, -0.000984336785, 0.000158479248, -0.0270215034]$$
- **Calibration Images**: 30 high-resolution checkerboard calibration images available in `MAV Images Calib/`.

---

## 5. Summary of Limitations and Adapter Strategy

1. **Typographical Field Names**: Header names in the raw CSVs contain typos and leading/trailing whitespace (e.g. `'Timpstemp'`, `' Pressure'`). The adapter trims all headers and maps `'Timpstemp'` to normalized `timestamp_seconds`.
2. **Multi-Rate Synchronization**: GPS operates at ~30 Hz, IMU raw streams at ~10 Hz, Onboard State at ~50 Hz, and Ground Truth at 1 Hz. Nearest-neighbor temporal matching is required to synchronize telemetry with camera frames.
3. **Metric Ground Truth Coordinates**: Position ground truth is stored in projected Cartesian grid coordinates (meters) rather than latitude/longitude, which is directly suitable for metric 3D reconstruction benchmarking.
