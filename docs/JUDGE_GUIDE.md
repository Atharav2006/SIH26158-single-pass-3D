# SIH 2026 Evaluator & Judge Guide

**Problem Statement PS 26158 (NTRO):** Single-Pass Drone Video to Accurate 3D Model Generation System  
**Repository:** `SIH26158-single-pass-3D`  
**Team Evaluation Reference:** SIH 2026

---

## 1. Problem Solved by This Prototype

Under operational constraints (e.g. reconnaissance, disaster relief, defense mapping), a drone typically makes a **single flight pass** over a target area. Standard photogrammetry workflows require extensive multi-loop overlaps, cross-grid flight patterns, and manual Ground Control Points (GCPs).

This prototype addresses the core challenges of **single-pass monocular reconstruction**:
1. **Scale Drift & Ambiguity:** Monocular Structure-from-Motion (SfM) has unobservable metric scale ($s$). We implement closed-form 7-DoF $\mathrm{Sim}(3)$ Umeyama alignment, metric geodesy (WGS84 $\to$ UTM/ENU), and GPS trajectory conditioning.
2. **Temporal & Spatial Frame Ingestion:** Automated frame extraction, timestamp alignment, and camera intrinsic calibration handling.
3. **Rigorous Trajectory & Reconstruction Benchmarking:** Automated calculation of Absolute Trajectory Error (ATE RMSE), Relative Pose Error (RPE), scale factor drift, and 3D point cloud generation.

---

## 2. 60-Second Quickstart (Zero-Download Sample Demo)

A sample drone video clip is committed directly to `data/samples/controlled_test/test_video.mp4` (2.9 MB). Judges can verify the full core pipeline locally in **less than 1 minute** without downloading external datasets.

### Step 1: Clone & Setup Environment
```bash
# 1. Clone repository
git clone https://github.com/Atharav2006/SIH26158-single-pass-3D.git
cd SIH26158-single-pass-3D

# 2. Create and activate virtual environment (Python >= 3.10)
python -m venv venv
# Windows:
.\venv\Scripts\Activate.ps1
# Linux/macOS:
source venv/bin/activate

# 3. Install package and dependencies
pip install -e .
pip install pytest
```

### Step 2: Run the One-Line Demo Runner
```bash
python scripts/run_sample_demo.py
```
**Expected Outcome:**
* `Stage 1: Video Ingestion & Stream Inspection` $\to$ **PASS**
* `Stage 2: Keyframe Extraction` $\to$ **PASS**
* `Stage 3: Geodetic WGS84 -> UTM32N -> Local ENU Projection` $\to$ **PASS**
* `Stage 4: 7-DoF Sim(3) Alignment & ATE Error Calculation` $\to$ **PASS**

### Step 3: Run the Full Test Suite
```bash
pytest --verbose
```
**Expected Outcome:** All **81 unit and integration tests PASS** in ~15 seconds.

---

## 3. Recommended Evaluation Flow

### Test 1: Full Baseline B0 Trajectory Benchmark Evaluation
To inspect the real evaluation metrics computed against real UAV flight ground truth:
```bash
# Runs Sim(3) trajectory evaluation on reconstructed camera poses
python -m pipelines.evaluation.evaluate_b0_trajectory
```

### Test 2: GPS Quality & Anchorability Conditioning
```bash
# Analyzes raw drone GPS quality, DOP/satellite degradation, and metric consistency
python -m pipelines.evaluation.analyze_gps_quality
```

### Test 3: Camera Trajectory & Point Cloud Visualization
```bash
# Generates 3D trajectory comparison figures and error distributions
python -m pipelines.baseline.visualize_zurich_trajectory
```

---

## 4. Genuinely Implemented Features vs. Roadmap

| Pipeline Component | Status in Prototype | Evidence / Source Code |
| :--- | :--- | :--- |
| **Video Ingestion & Demuxing** | **Fully Implemented** | `src/ingestion/video_metadata.py`, `src/ingestion/frame_extractor.py` |
| **Coordinate Frame Conventions (SE3/Sim3)** | **Fully Implemented** | `src/pose/coordinate_frames.py`, `src/pose/models.py` |
| **SfM Reconstruction Engine** | **Fully Implemented** | `src/reconstruction/colmap_wrapper.py`, `colmap_parser.py` |
| **WGS84 $\to$ UTM $\to$ ENU Geodesy** | **Fully Implemented** | `src/geodesy/projection.py` |
| **Sim(3) Umeyama Trajectory Alignment** | **Fully Implemented** | `src/metrics/alignment.py` |
| **ATE / RPE / Drift Trajectory Metrics** | **Fully Implemented** | `src/metrics/trajectory_metrics.py` |
| **GPS Anchorability & Dilution Analysis** | **Fully Implemented** | `pipelines/evaluation/analyze_gps_quality.py` |
| *GPS-IMU Factor Graph Fusion (B1)* | *In Progress / Roadmap* | `src/sensor_fusion/` |
| *Monocular Depth / MVS Mesh Generation* | *Roadmap* | `src/depth/`, `src/reconstruction/` |
| *Dynamic Vehicle/Pedestrian Masking* | *Roadmap* | `src/dynamic_objects/` |
| *Voxel Occlusion & Confidence Labeling* | *Roadmap* | `src/confidence/`, `src/occlusion/` |

---

## 5. Honest Known Limitations

1. **Monocular Scale Observability:** Pure monocular video without metric priors (GPS/IMU/GCPs) is subject to scale gauge freedom. The pipeline uses GPS position anchorability and 7-DoF Sim(3) alignment to resolve scale.
2. **Exhaustive Matching Latency:** Exhaustive feature matching across all image pairs scales quadratically ($O(N^2)$). For production runs on 300+ frames, sequential or spatial matching is recommended over exhaustive matching.
3. **Hardware Acceleration:** Structure-from-Motion bundle adjustment runs on CPU (Ceres Solver), while SIFT feature extraction and matching utilize CUDA when available.
