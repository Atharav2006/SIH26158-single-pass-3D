# SIH26158: Strategic Dataset Selection & Acquisition Plan

Based on the multi-dimensional dataset survey in [dataset_landscape.md](file:///d:/SIH26158-single-pass-3D/docs/dataset_landscape.md), this document establishes the official dataset selection strategy for developing, verifying, and demonstrating the **Single-Pass Drone Video to Accurate 3D Model Generation System** (SIH26158).

---

## 1. Executive Selection Summary

| Role | Selected Dataset | Provider | Key Characteristics | Target Pipeline Milestone |
| :--- | :--- | :--- | :--- | :--- |
| **1. Primary Development Dataset** | **Mid-Air (Low-Altitude Flights)** | Univ. of Liège | Continuous drone trajectories, ground-truth depth, 6DoF poses, synthetic GPS/IMU. | Step 6 (Pose), Step 7 (Depth), Step 8 (Sensor Fusion) |
| **2. Primary Metric Evaluation Dataset**| **ETH3D (High-Res Outdoor/SLAM)** | ETH Zurich / Microsoft | Millimeter TLS laser scans, laser-registered camera poses, standard evaluation protocols. | Step 9 (Reconstruction Benchmarking & Accuracy Metrics) |
| **3. Dynamic-Object Test Dataset** | **UAVid (4K Oblique Sequences)** | Univ. of Twente | 4K 30 FPS drone video, explicit `Moving car` and `Human` semantic annotations. | Step 10 (Dynamic Object Filtering & Masking) |
| **4. Real-Drone Demo Dataset** | **Zurich Urban MAV Dataset** | Univ. of Zurich (RPG) | Real ~2 km drone flight through urban streets, forward HD video, synchronized GPS/IMU. | Step 11 (End-to-End Single-Pass Demo & Telemetry Fusion) |
| **5. Backup Dataset** | **EuRoC MAV & VisDrone** | ETH Zurich / Tianjin Univ. | High-speed aerial maneuvers, dense urban traffic clips, Vicon 6DoF pose ground truth. | Failover validation & robustness stress testing |

---

## 2. Detailed Technical Justification for Each Selection

### 1. Primary Development Dataset: Mid-Air
* **Official URL**: [https://midair.ulg.ac.be](https://midair.ulg.ac.be)
* **License**: MIT / Open Research
* **Why Selected**:
  1. **Ground-Truth Completeness**: Provides exact per-frame 6DoF camera poses, dense pixel-aligned depth maps, surface normal maps, and simulated GPS/IMU logs without sensor noise.
  2. **Fast Debugging Loops**: Enables unit testing of keyframe selection, depth fusion, and pose estimation in isolation, guaranteeing that algorithmic regressions are not masked by real-world camera motion blur or GPS multipath errors.
  3. **Drone-Specific Dynamics**: Synthetic trajectories are generated with realistic drone physics (low-altitude forward flights, bank angles, variable speeds).

---

### 2. Primary Metric Evaluation Dataset: ETH3D
* **Official URL**: [https://www.eth3d.net/](https://www.eth3d.net/)
* **License**: CC BY-NC-SA 4.0
* **Why Selected**:
  1. **Sub-Millimeter Ground Truth**: Captured with high-precision Terrestrial Laser Scanners (TLS), establishing an unassailable geometric reference for quantitative evaluation.
  2. **Standardized Evaluation Metrics**: Directly supports industry-standard evaluation scripts for:
     - **Chamfer Distance (Accuracy & Completeness)**
     - **F-score** at varying distance thresholds ($\le 2\text{ cm}, \le 5\text{ cm}, \le 10\text{ cm}$)
     - **Absolute Trajectory Error (ATE RMSE)** for camera pose solvers.
  3. **Outdoor Architectural Complexity**: Includes complex multi-view facade structures and courtyard geometries that match drone photogrammetry targets.

---

### 3. Dynamic-Object Test Dataset: UAVid
* **Official URL**: [https://uavid.nl/](https://uavid.nl/)
* **License**: CC BY-NC-SA 4.0
* **Why Selected**:
  1. **High-Resolution Drone Video**: 4K oblique video sequences recorded at 30 FPS from real multirotor UAVs flying over city roads, roundabouts, and intersections.
  2. **Explicit Transient vs. Static Ground Truth**: Dense per-pixel annotations clearly distinguish `Moving car` and `Human` from `Static car`, `Building`, `Road`, and `Tree`.
  3. **Artifact-Free Reconstruction Validation**: Provides the ground truth required to verify that moving vehicles and walking pedestrians are cleanly segmented and inpainted without leaving "ghosting" artifacts in reconstructed 3D point clouds.

---

### 4. Real-Drone Demo Dataset: Zurich Urban MAV Dataset
* **Official URL**: [http://rpg.ifi.uzh.ch/zurichmavdataset.html](http://rpg.ifi.uzh.ch/zurichmavdataset.html)
* **License**: Open / Unrestricted Research & Commercial
* **Why Selected**:
  1. **Single-Pass Urban Flight**: Captures a continuous ~2 km single-pass drone flight at street canyon level, exactly matching the problem statement of SIH26158.
  2. **Synchronized Multi-Sensor Streams**: Contains time-synchronized forward-facing HD video (720p @ 30 FPS), high-res stills (20 MP), onboard GPS coordinates, and 3-axis IMU telemetry.
  3. **Real-World Metric Scale Recovery**: Allows end-to-end demonstration of fusing monocular drone video with GPS/telemetry to produce metric-scaled, geo-registered 3D reconstructions.

---

### 5. Backup & Stress-Testing Datasets: EuRoC MAV & VisDrone
* **EuRoC MAV**: [https://projects.asl.ethz.ch/datasets/euroc-mav/](https://projects.asl.ethz.ch/datasets/euroc-mav/) (CC BY 4.0)
  - Provides millisecond-synchronized visual-inertial data with Vicon millimeter ground truth for high-speed dynamic drone motion.
* **VisDrone**: [http://aiskyeye.com/](http://aiskyeye.com/) (CC BY-NC-SA 3.0)
  - Provides 288 video clips across multiple cities and lighting conditions, offering a rich fallback library for stress-testing frame extraction and feature tracking under heavy motion blur.

---

## 3. Storage and Acquisition Guidelines

Per project policy:
1. **Isolated Storage**: All downloaded dataset archives must reside outside the Git repository in:
   `D:\SIH26158\datasets\<dataset_name>\`
2. **Zero Repository Bloat**: No raw dataset files, video archives, or large point clouds will be committed to Git.
3. **Reproducible Fetch Scripts**: In Step 5B/6, automated download and extraction scripts will be implemented under `scripts/datasets/` to fetch specific sequences on demand.
