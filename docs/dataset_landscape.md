# SIH26158: Dataset Landscape for Single-Pass Drone Video to 3D Model Generation

This document provides a comprehensive survey and evaluation of real-world and synthetic datasets for developing, validating, and benchmarking the **Single-Pass Drone Video to Accurate 3D Model Generation System**.

---

## 1. Evaluation Criteria

For a single-pass UAV 3D reconstruction pipeline, candidate datasets are evaluated across key functional criteria:
1. **Video vs. Discrete Images**: Availability of continuous, single-pass video streams vs. unordered photo collections.
2. **Telemetry & Sensor Data**: Synchronized GPS/GNSS, IMU, barometric altitude, and camera telemetry.
3. **Ground Truth Quality**: Availability of millimeter/centimeter-accurate 6DoF camera poses and dense 3D reference geometry (LiDAR, laser scans, survey GCPs).
4. **Scene Realism**: Urban structures, infrastructure, terrain, vegetation, and realistic drone flight altitudes (10m–100m).
5. **Dynamic Objects**: Presence of moving cars, pedestrians, and cyclists to evaluate transient object masking/filtering.
6. **Licensing & Accessibility**: Open research accessibility without legal or restrictive distribution blockers.

---

## 2. Dataset Comparison Matrix

| Attribute | **ETH3D** | **EuRoC MAV** | **UAVid** | **VisDrone** | **UrbanScene3D** | **Zurich Urban MAV** | **Mid-Air** | **OpenDroneMap (ODM)** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Provider** | ETH Zurich / Microsoft | ETH Zurich (ASL) | Univ. of Twente | Tianjin Univ. (AISKYEYE) | Shenzhen Univ. / Tsinghua | Univ. of Zurich (RPG) | Univ. of Liège | OpenDroneMap / SenseFly |
| **URL** | [eth3d.net](https://www.eth3d.net/) | [projects.asl.ethz.ch](https://projects.asl.ethz.ch/datasets/euroc-mav/) | [uavid.nl](https://uavid.nl/) | [aiskyeye.com](http://aiskyeye.com/) | [vcc.tech/UrbanScene3D](https://vcc.tech/UrbanScene3D) | [rpg.ifi.uzh.ch](http://rpg.ifi.uzh.ch/zurichmavdataset.html) | [midair.ulg.ac.be](https://midair.ulg.ac.be) | [opendronemap.org](https://www.opendronemap.org/odm/datasets/) |
| **Video** | Synchronized video SLAM sets (734x504 @ 14 FPS) | Stereo video sequences (752x480 @ 20 FPS) | 4K continuous oblique video (3840x2160 @ 30 FPS) | 288 video clips (up to 4K / 1080p @ 30 FPS) | Oblique video / high-overlap photo sequences | 720p HD forward video @ 30 FPS + 20MP stills | 54 continuous synthetic video tracks (1024x1024 @ 25 FPS) | Drone flight video / high-overlap aerial stills |
| **Images** | High-res DSLRs + low-res frames | Monochromatic stereo pairs | 4K extracted video frames | 260k+ video frames + 10k static photos | 128k+ high-res aerial images | 20MP stills (2 Hz) + 720p frames | 420k+ rendered RGB frames | 50–1000+ aerial GeoTIFF / JPEG images |
| **GPS / GNSS** | ❌ No (Indoor/Outdoor local) | ❌ No (Indoor Vicon/Leica) | ❌ No | ❌ No | ❌ Partial (EXIF GPS tags) | ✔️ Synchronized GPS receiver | ✔️ Simulated GPS (lat/lon/alt) | ✔️ Survey-grade RTK/PPK GPS + GCPs |
| **IMU** | ❌ No | ✔️ 200 Hz ADIS16448 MEMS IMU | ❌ No | ❌ No | ❌ No | ✔️ Synchronized IMU | ✔️ Simulated IMU (Acc/Gyro 100 Hz) | ❌ Drone flight logs (Pitch/Roll/Yaw in EXIF) |
| **Altitude** | ❌ No | ❌ No | ❌ No | ❌ Approximate metadata | ❌ Partial | ✔️ GPS Altitude | ✔️ Exact ground truth altitude | ✔️ Flight altitude + relative AGL |
| **Camera Calibration** | ✔️ Full pinhole + radial distortion | ✔️ Full stereo intrinsics + extrinsics | ❌ Uncalibrated (estimated via SfM) | ❌ Uncalibrated consumer drone cameras | ✔️ Full camera intrinsic parameters | ✔️ Calibrated camera models | ✔️ Perfect synthetic pinhole intrinsics | ✔️ EXIF focal length / Sensor sizes |
| **Pose Ground Truth** | ✔️ Laser-scan registered 6DoF poses | ✔️ Vicon / Leica millimeter 6DoF poses | ❌ No | ❌ No | ✔️ High-precision reconstructed poses | ✔️ GPS/SfM-aligned 6DoF trajectories | ✔️ Exact 6DoF trajectories | ✔️ Surveyed GCP ground coordinates |
| **3D Ground Truth** | ✔️ Sub-millimeter TLS laser point clouds | ✔️ Leica MS50 3D laser point clouds | ❌ No | ❌ No | ✔️ Dense aerial & terrestrial LiDAR clouds/meshes | ❌ Google Earth / SfM reference | ✔️ Exact dense depth maps + surface normals | ✔️ Surveyed DEM / Orthophoto / LiDAR |
| **Scene Type** | Indoor rooms, outdoor buildings & courtyards | Industrial machine halls & Vicon rooms | Dense urban streets, intersections, greenery | Multi-city urban intersections, traffic, crowds | Large-scale urban cities, university campus, towers | Urban street canyons (~2 km drone flight) | Forest, rural, agricultural, mountains | Rural, urban infrastructure, quarries, buildings |
| **Dynamic Objects** | ❌ Static scenes only | ❌ Static rooms only | ✔️ Moving cars, trucks, pedestrians, cyclists | ✔️ Dense vehicles, pedestrians, crowds | ❌ Static urban scans | ✔️ Real urban traffic & pedestrians | ❌ Static environments | ❌ Mostly static surveying scenes |
| **License** | CC BY-NC-SA 4.0 | CC BY 4.0 | CC BY-NC-SA 4.0 | CC BY-NC-SA 3.0 | Non-Commercial Research Use | Open / Unrestricted Research & Commercial | MIT / Open Research | Open / CC-BY / Public Domain (ODMdata) |
| **Approx. Size** | ~10 GB – 50 GB | ~20 GB (all sequences) | ~30 GB (4K sequences) | ~40 GB | ~150 GB (selective subset ~15 GB) | ~15 GB | ~80 GB (selective subsets ~10 GB) | ~2 GB – 10 GB per scene |

---

## 3. Suitability Assessment Matrix

| Dataset | Frame Extraction | Pose Estimation | Metric 3D Reconstruction | GPS / Scale Alignment | Dynamic-Object Research | Standard Benchmarking | Overall Category |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **ETH3D** | High | Excellent | **Outstanding** | Low | None | **Industry Standard** | Metric Evaluation Benchmark |
| **EuRoC MAV** | High | **Outstanding** | High (Indoor) | None | None | **Industry Standard** | Visual-Inertial Odometry / SLAM |
| **UAVid** | **Outstanding** | Moderate | Low | Low | **Outstanding** | High (Segmentation) | Dynamic Object Masking |
| **VisDrone** | **Outstanding** | Moderate | Low | Low | **Outstanding** | High (Tracking) | Dynamic Object Detection |
| **UrbanScene3D**| High | Excellent | **Outstanding** | Moderate | Low | **Outstanding** | Large-Scale Aerial 3D Reconstruction |
| **Zurich Urban MAV**| **Outstanding** | High | Moderate | **Outstanding** | Moderate | Moderate | Real Single-Pass Drone Trajectory & GPS |
| **Mid-Air** | **Outstanding** | **Outstanding** | Excellent (Synthetic)| **Outstanding** | Low | High | Synthetic Algorithm Development |
| **OpenDroneMap (ODM)**| High | High | High | **Outstanding** | Low | High | Real-World Aerial Photogrammetry & Demo |

---

## 4. Functional Category Breakdown

### Category A: Datasets for Algorithm Development
* **Characteristics**: Deterministic data, high frame rates, perfect or highly controlled ground truth, rich intermediate modalities (depth, normals, trajectory), fast debugging loops.
* **Top Candidates**: **Mid-Air**, **EuRoC MAV (easy sequences)**.
* **Role in Pipeline**: Validates keyframe selection algorithms, depth estimation wrappers, optical flow, and sensor fusion logic before encountering real-world sensor noise.

### Category B: Datasets for Metric Evaluation
* **Characteristics**: Millimeter-to-centimeter precision LiDAR ground truth, rigorous error metrics (Chamfer distance, Point-to-Plane, F-score, Absolute Trajectory Error (ATE)), established multi-view evaluation protocols.
* **Top Candidates**: **ETH3D (Outdoor Multi-View / SLAM)**, **UrbanScene3D**, **EuRoC MAV**.
* **Role in Pipeline**: Evaluates point cloud accuracy, completeness, and camera trajectory drift against laser-scanned ground truth.

### Category C: Datasets for Dynamic-Object Testing
* **Characteristics**: High-resolution continuous UAV video capturing dense urban traffic, pedestrians, and moving vehicles from oblique drone camera angles.
* **Top Candidates**: **UAVid (4K sequences)**, **VisDrone**.
* **Role in Pipeline**: Evaluates moving-object segmentation, dynamic keypoint removal, epipolar outlier filtering, and artifact-free background 3D reconstruction.

### Category D: Datasets for Final SIH Demonstration
* **Characteristics**: Real drone flights over large-scale infrastructure, buildings, or terrain with realistic drone forward motion, gimbal tilt, GPS logs, and visually compelling 3D mesh outputs.
* **Top Candidates**: **Zurich Urban MAV Dataset**, **OpenDroneMap (e.g. Park, Bellus, Quarry datasets)**, **UrbanScene3D (Campus scene)**.
* **Role in Pipeline**: Demonstrates the end-to-end single-pass drone-to-3D capabilities in real-world scenarios.
