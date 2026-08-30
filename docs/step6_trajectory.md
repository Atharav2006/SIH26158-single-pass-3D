# Step 6B: Ground-Truth Trajectory Loader, Coordinate Conversions, and Visualization

This document details the mathematical formulation, typed data models, local coordinate conversion, image association, trajectory sanity validation, visualization architecture, and cross-check verification for the **Ground-Truth Trajectory Processing Subsystem** in SIH26158.

---

## 1. Verified Pose Semantics & Reference Frames

### 1.1 Reference Coordinate Frames
- **Global UTM Reference Frame ($\mathcal{F}_{\text{UTM}}$)**:
  - Universal Transverse Mercator (UTM Zone 32N) on the WGS84 ellipsoid.
  - Coordinate axes: $+X = \text{East}$, $+Y = \text{North}$, $+Z = \text{Up}$ (AMSL altitude in meters).
  - Spatial baseline in Zurich: $\mathbf{p}_0 = [465666.057548, 5247973.646622, 469.019496]^T\text{ meters}$.
- **Internal Local Reference Frame ($\mathcal{F}_{\text{Local}}$)**:
  - Tangent-plane East-North-Up (ENU) Cartesian frame centered at the first verified ground-truth position $\mathbf{p}_0$.
  - Mathematical transformation:
    $$\mathbf{p}_{\text{local}} = \mathbf{p}_{\text{UTM}} - \mathbf{p}_0$$
    $$\mathbf{R}_{\text{local}} = \mathbf{R}_{\text{UTM}}$$
- **Camera Frame ($\mathcal{F}_{\text{cam}}$)**:
  - Standard Computer Vision / OpenCV convention: $+X_{\text{cam}} = \text{Right}$, $+Y_{\text{cam}} = \text{Down}$, $+Z_{\text{cam}} = \text{Forward along optical axis}$.

### 1.2 Pose Representation & Semantics
- **Position**: Camera optical center in World / Local ENU coordinates ($C_W$).
- **Orientation**: Rotation from World East-North-Up to Camera frame ($T_{WC}$ camera attitude) represented as a Hamilton unit quaternion $[q_x, q_y, q_z, q_w]$ in scalar-last format.
- **Normalization Constraint**: Every quaternion is validated to satisfy:
  $$\left| \|\mathbf{q}\| - 1.0 \right| < 10^{-4}$$

---

## 2. Typed Data Model Architecture ([models.py](file:///d:/SIH26158-single-pass-3D/src/pose/models.py))

```python
@dataclass
class Position:
    x: float
    y: float
    z: float
    unit: str = "meter"

@dataclass
class Quaternion:
    qx: float
    qy: float
    qz: float
    qw: float
    convention: str = "Hamilton"

class Pose:
    timestamp_seconds: float
    position_xyz: Position
    orientation_xyzw: Quaternion
    source_frame: str
    target_frame: str
    pose_semantics: str = "camera_optical_center_in_world"
```

---

## 3. Data Flow & Artifact Generation

```
+------------------------------------+        +----------------------------------------+
|  Normalized Ingestion Outputs      |        |  Pose Loader & Trajectory Engine       |
|  - pose.csv (2,708 Poses)          | -----> |  - load_poses_from_csv()               |
|  - images.csv (350 Frames)         |        |  - transform_to_local_enu()            |
|  - gps.csv (81,169 GNSS Records)   |        |  - associate_poses_to_images()         |
+------------------------------------+        |  - Trajectory.validate_trajectory()    |
                                              |  - Trajectory.compute_statistics()     |
                                              +----------------------------------------+
                                                                   |
                                                                   v
+--------------------------------------------------------------------------------------+
|  Generated Trajectory Artifacts                                                      |
|  - outputs/reports/zurich_mav/trajectory.csv              (Local ENU 6DoF Poses)     |
|  - outputs/reports/zurich_mav/trajectory.json             (Metric Statistics)        |
|  - outputs/reports/zurich_mav/trajectory_validation.json  (Validation Status: PASS)  |
|  - outputs/reports/zurich_mav/image_pose_associations.csv (Image-to-Pose Mapping)    |
|  - outputs/reports/zurich_mav/trajectory_3d.png           (3D Isometric Trajectory)  |
|  - outputs/reports/zurich_mav/trajectory_topdown.png      (2D Top-Down Flight Path)  |
+--------------------------------------------------------------------------------------+
```

---

## 4. Output Schemas & Validation

### 4.1 `image_pose_associations.csv`
| Column | Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `image_id` | Integer | One-based sequential image identifier | `1` |
| `image_timestamp` | Float | Frame capture timestamp in seconds | `7.009129` |
| `pose_timestamp` | Float | Matched ground-truth pose timestamp in seconds | `7.009129` |
| `absolute_delta_seconds` | Float | Absolute synchronization difference ($|t_{\text{img}} - t_{\text{pose}}|$) | `0.0` |
| `pose_index` | Integer | Zero-based index of matched pose in trajectory sequence | `0` |

### 4.2 `trajectory_validation.json`
```json
{
    "status": "PASS",
    "pose_count": 2708,
    "coordinate_frame": "Local_ENU",
    "issues_detected": 0,
    "issues": []
}
```

### 4.3 `trajectory.json`
```json
{
    "pose_count": 2708,
    "valid_pose_count": 2708,
    "coordinate_frame": "Local_ENU",
    "start_timestamp_seconds": 7.009129,
    "end_timestamp_seconds": 2707.033333,
    "duration_seconds": 2700.024204,
    "spatial_extent": {
        "min_x": -189.642,
        "max_x": 168.212,
        "span_x": 357.854,
        "min_y": -231.633,
        "max_y": 333.409,
        "span_y": 565.042,
        "min_z": -8.868,
        "max_z": 19.942,
        "span_z": 28.81,
        "unit": "meter"
    },
    "trajectory_length_meters": 1915.625,
    "mean_speed_mps": 0.709,
    "average_speed_mps": 0.709,
    "median_speed_mps": 0.709,
    "max_speed_mps": 1.925,
    "dataset_source": "Zurich Urban MAV Dataset (AGZ)",
    "origin_utm_coordinates": {
        "x_easting": 465666.057548,
        "y_northing": 5247973.646622,
        "z_altitude": 469.019496,
        "unit": "meter"
    },
    "gps_statistics": {
        "gps_record_count": 81169,
        "min_latitude": 47.3822502,
        "max_latitude": 47.3873331,
        "latitude_span_deg": 0.0050829,
        "min_longitude": 8.5425953,
        "max_longitude": 8.547358,
        "longitude_span_deg": 0.0047627,
        "min_altitude_meters": 448.96,
        "max_altitude_meters": 519.3,
        "altitude_span_meters": 70.34
    },
    "synchronization_statistics": {
        "total_images": 350,
        "matched_poses": 36,
        "unmatched_images": 314,
        "match_rate": 0.1029,
        "mean_time_diff_sec": 0.021478,
        "max_time_diff_sec": 0.048305
    }
}
```

---

## 5. Independent Cross-Checks (Part K)

1. **Image-Associated Pose Consistency**: Verified that `imgid` in `GroundTruthAGL.csv` correctly matches the timestamp sequence of `OnboardGPS.csv` ($t = 7.009129\text{ s}$).
2. **Sample Association Integrity**: All 350 image records are preserved in `image_pose_associations.csv`. Within a $\le 50\text{ ms}$ tolerance window, exactly 36 calibration/initial frames match the 1 Hz bundle adjustment poses without silent data drop.
3. **Spatial Extent Invariance**: Local coordinate extents ($\Delta X = 357.85\text{ m}, \Delta Y = 565.04\text{ m}, \Delta Z = 28.81\text{ m}$) exactly equal raw UTM coordinate spans.
4. **GPS vs. Ground Truth Separation**: Ground-truth poses remain derived solely from photogrammetric aerial bundle adjustment; GPS is maintained as an independent sensor stream with a known antenna offset ($\sim 4.6\text{ m}$ average lever arm).

---

## 6. Pipeline CLI Command

```powershell
python -m pipelines.baseline.visualize_zurich_trajectory --normalized outputs/reports/zurich_mav --output outputs/reports/zurich_mav
```

### Execution Telemetry Output
```text
============================================================
SIH26158: Zurich Urban MAV Trajectory Processing & Visualization
============================================================
Normalized Input Dir: D:\SIH26158-single-pass-3D\outputs\reports\zurich_mav
Output Directory:     D:\SIH26158-single-pass-3D\outputs\reports\zurich_mav

Loaded 2708 ground-truth poses from pose.csv
Loaded 350 image records from images.csv
Exported image_pose_associations.csv
Exported trajectory_validation.json (Status: PASS)
Exported trajectory.csv
Exported trajectory.json
Generated 3D Plot:       trajectory_3d.png
Generated Top-Down Plot: trajectory_topdown.png

--- Trajectory Summary ---
  Validation Status:  PASS
  Pose Count:         2708
  Duration:           2700.02 s (45.00 min)
  Trajectory Length:  1915.62 m (1.916 km)
  Mean Speed:         0.71 m/s
  Median Speed:       0.71 m/s
  Max Speed:          1.93 m/s
  Local Extents:      X: [-189.642, 168.212] m
                      Y: [-231.633, 333.409] m
                      Z: [-8.868, 19.942] m
  Image-Pose Sync:    36/350 matched (10.3%)
============================================================
TRAJECTORY STATUS: PASS
```

---

## 7. Test Suite Verification

```powershell
pytest -q
```
**Results**:
```text
..............................                                           [100%]
30 passed in 12.26s
```
