# Step 5D: Zurich Urban MAV Dataset Ingestion & Adapter Report

This document outlines the architecture, data schemas, synchronization strategy, validation results, and usage of the **Zurich Urban MAV Dataset Adapter** in SIH26158.

---

## 1. Source Dataset Overview

- **Dataset**: Zurich Urban Micro Aerial Vehicle Dataset (Air-Ground Zurich AGZ)
- **Official Provider**: Robotics and Perception Group (RPG), University of Zurich & ETH Zurich (Andras Majdik, Yves Albers-Schoenberg, Davide Scaramuzza)
- **Official URL**: [http://rpg.ifi.uzh.ch/zurichmavdataset.html](http://rpg.ifi.uzh.ch/zurichmavdataset.html)
- **License**: Open / Unrestricted Research and Commercial Use
- **Read-Only Location**: `D:\SIH26158\datasets\zurich_mav\AGZ_subset`

---

## 2. Directory Structure

```
D:\SIH26158\datasets\zurich_mav\AGZ_subset/
├── AGZ.bag                               # Synchronized ROS bag file
├── calibration_data.npz                  # Intrinsics and distortion coefficients
├── loadGroundTruthAGL.m                  # MATLAB parsing script
├── Log Files/                            # Raw telemetry logs
│   ├── BarometricPressure.csv            # 27,052 records
│   ├── GroundTruthAGL.csv                # 2,708 records (1 Hz metric 6DoF Ground Truth)
│   ├── GroundTruthAGM.csv                # 81,169 records (Air-ground correspondences)
│   ├── OnboardGPS.csv                    # 81,169 records (~30 Hz GNSS receiver logs)
│   ├── OnboardPose.csv                   # 135,098 records (~50 Hz Onboard state estimator)
│   ├── RawAccel.csv                      # 27,050 records (~10 Hz 3-axis Accelerometer)
│   ├── RawGyro.csv                       # 27,050 records (~10 Hz 3-axis Rate Gyroscope)
│   └── StreetViewGPS.csv                 # 113 records (Google Street View poses)
├── MAV Images/                           # Directory for flight frames
└── MAV Images Calib/                     # 30 Checkerboard calibration images (1920x1080 PNG)
```

---

## 3. Normalized Data Model (`datasets/normalized/zurich_mav_sample/`)

Without modifying or duplicating raw heavy binaries, the adapter produces a clean, lightweight metadata package adhering to the project's standard schemas:

```
datasets/normalized/zurich_mav_sample/
├── dataset.json                          # Dataset metadata and record counts
├── images.csv                            # Image index, filenames, dimensions, and timestamps
├── gps.csv                               # WGS84 coordinates, altitude, and timestamps
├── imu.csv                               # 3-axis acceleration and angular rate measurements
├── pose.csv                              # 6DoF position (tx, ty, tz) and quaternion (qx, qy, qz, qw)
└── camera.json                           # Pinhole focal lengths, principal point, and distortion
```

### 3.1 `images.csv`
| Column | Type | Description |
| :--- | :--- | :--- |
| `image_id` | Integer | One-based sequential image identifier |
| `filename` | String | Image file name (e.g. `Calibration_Image_01.png`) |
| `timestamp_seconds` | Float | Floating-point timestamp in seconds |
| `width` | Integer | Frame width in pixels (1920) |
| `height` | Integer | Frame height in pixels (1080) |

### 3.2 `gps.csv`
| Column | Type | Description |
| :--- | :--- | :--- |
| `timestamp_seconds` | Float | Synchronized timestamp in seconds |
| `latitude` | Float | WGS84 Geodetic Latitude (degrees, ~47.38°) |
| `longitude` | Float | WGS84 Geodetic Longitude (degrees, ~8.54°) |
| `altitude_if_available` | Float | Altitude AMSL / Ellipsoidal (meters, ~464.91m) |

### 3.3 `imu.csv`
| Column | Type | Description |
| :--- | :--- | :--- |
| `timestamp_seconds` | Float | Synchronized timestamp in seconds |
| `accel_x`, `accel_y`, `accel_z` | Float | Linear accelerations in $\text{m/s}^2$ |
| `gyro_x`, `gyro_y`, `gyro_z` | Float | Angular velocities in $\text{rad/s}$ |

### 3.4 `pose.csv`
| Column | Type | Description |
| :--- | :--- | :--- |
| `timestamp_seconds` | Float | Synchronized timestamp in seconds |
| `tx`, `ty`, `tz` | Float | Metric Cartesian position in Swiss Grid / UTM (meters) |
| `qx`, `qy`, `qz`, `qw` | Float | Normalized orientation quaternion |

### 3.5 `camera.json`
```json
{
    "model": "pinhole_radial_tangential",
    "fx": 893.39010814,
    "fy": 898.32648616,
    "cx": 951.1310043,
    "cy": 555.13350077,
    "distortion_parameters_if_available": [
        -0.28052513,
        0.115806413,
        -0.000984336785,
        0.000158479248,
        -0.0270215034
    ],
    "image_width": 1920,
    "image_height": 1080
}
```

---

## 4. Timestamp Handling & Synchronization

1. **Unit Conversion**: The native dataset logs timestamps under `'Timpstemp'` as integer microseconds ($\mu\text{s}$). The adapter rigorously converts all microsecond timestamps to standard floating-point seconds:
   $$t_{\text{seconds}} = \frac{\text{Timpstemp}}{1\,000\,000.0}$$
2. **Temporal Synchronization Utility ([synchronization.py](file:///d:/SIH26158-single-pass-3D/src/ingestion/synchronization.py))**:
   - Implements $O(\log N)$ binary-search nearest-neighbor matching across heterogeneous sensor sampling frequencies (GPS at ~30 Hz, IMU at ~10 Hz, Pose at ~1 Hz / 50 Hz).
   - Enforces a configurable `max_tolerance` window (default: $0.05\text{ s}$ to $0.1\text{ s}$) to associate telemetry records with camera frames without fabricating data or introducing unverified interpolations.

---

## 5. Pipeline CLI Usage

To profile, normalize, and validate the dataset:
```powershell
python -m pipelines.baseline.inspect_zurich_mav --dataset D:\SIH26158\datasets\zurich_mav --output outputs/reports/zurich_mav
```

### Execution Telemetry Output
```text
============================================================
SIH26158: Zurich Urban MAV Dataset Ingestion & Validation
============================================================
Dataset Path:     D:\SIH26158\datasets\zurich_mav
Output Directory: D:\SIH26158-single-pass-3D\outputs\reports\zurich_mav

--- Validating & Parsing Dataset ---

--- Exported Normalized Files ---
  dataset : D:\SIH26158-single-pass-3D\outputs\reports\zurich_mav\dataset.json
  images  : D:\SIH26158-single-pass-3D\outputs\reports\zurich_mav\images.csv
  gps     : D:\SIH26158-single-pass-3D\outputs\reports\zurich_mav\gps.csv
  imu     : D:\SIH26158-single-pass-3D\outputs\reports\zurich_mav\imu.csv
  pose    : D:\SIH26158-single-pass-3D\outputs\reports\zurich_mav\pose.csv
  camera  : D:\SIH26158-single-pass-3D\outputs\reports\zurich_mav\camera.json

--- Validation Summary ---
  Status:             PASS
  Images Found:       350
  GPS Records:        81169
  IMU Records:        27050
  Pose Records:       2708
  Camera Model:       pinhole_radial_tangential (1920x1080)
  Focal Length (fx):  893.39 px
  Image-GPS Sync:     350 matched (mean dt: 0.00 ms)

  Validation Report:  D:\SIH26158-single-pass-3D\outputs\reports\zurich_mav\zurich_mav_validation.json
============================================================
RESULT: SUCCESS
```

---

## 6. Test Suite Verification

Run all integration and unit tests:
```powershell
pytest -v
```
**Results**:
```text
tests/integration/test_zurich_mav_dataset.py::test_dataset_root_validation PASSED [  5%]
tests/integration/test_zurich_mav_dataset.py::test_image_discovery_and_readability PASSED [ 11%]
tests/integration/test_zurich_mav_dataset.py::test_gps_parsing PASSED    [ 16%]
tests/integration/test_zurich_mav_dataset.py::test_imu_parsing PASSED    [ 22%]
tests/integration/test_zurich_mav_dataset.py::test_pose_parsing PASSED   [ 27%]
tests/integration/test_zurich_mav_dataset.py::test_camera_calibration PASSED [ 33%]
tests/integration/test_zurich_mav_dataset.py::test_timestamp_parsing_and_monotonicity PASSED [ 38%]
tests/integration/test_zurich_mav_dataset.py::test_synchronization PASSED [ 44%]
tests/integration/test_zurich_mav_dataset.py::test_normalized_output_schema PASSED [ 50%]
tests/test_project_structure.py::test_directories_exist PASSED           [ 55%]
tests/test_project_structure.py::test_imports PASSED                     [ 61%]
tests/test_project_structure.py::test_project_version PASSED             [ 66%]
tests/test_project_structure.py::test_config_system PASSED               [ 72%]
tests/test_project_structure.py::test_logging_system PASSED              [ 77%]
tests/unit/test_frame_extractor.py::test_frame_extractor_full_fps PASSED [ 83%]
tests/unit/test_frame_extractor.py::test_frame_extractor_custom_fps_and_resize PASSED [ 88%]
tests/unit/test_frame_extractor.py::test_frame_extractor_invalid_inputs PASSED [ 94%]
tests/unit/test_video_metadata.py::test_video_metadata_extraction PASSED [100%]

============================= 18 passed in 12.18s =============================
```

---

## 7. Known Limitations

- **Microsecond GPS Jitter**: In the raw `OnboardGPS.csv` log, 6 microsecond-level packet jitter occurrences were empirically identified over 81,169 records (jumps of $-0.28\text{ ms}$ to $-7.4\text{ ms}$). These represent raw hardware driver timing artifacts and are preserved without synthetic alterations.
- **Asymmetric Sensor Sampling Rates**: GPS (~30 Hz), IMU (~10 Hz), and Ground Truth (1 Hz) require nearest-neighbor lookup via `TemporalSynchronizer` rather than assuming synchronized index matching.
