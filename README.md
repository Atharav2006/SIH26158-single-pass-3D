# SIH26158: Single-Pass Drone Video to Accurate 3D Model Generation System

## Project Purpose
This project is dedicated to building an automated pipeline that reconstructs high-fidelity 3D models (dense point clouds, textured meshes, or neural fields) from a **single-pass** continuous video stream captured by a drone. By resolving camera poses, depth maps, and fusing spatial points efficiently, the system targets high-speed mapping and analysis of geographical terrains, infrastructure, and object assets without requiring dense multi-angle photo surveys.

* **SIH Problem ID**: 26158
* **Current Development Stage**: Initial Scaffolding / Pre-Baseline B0

---

## Repository Structure

The project follows a modular packaging structure to enforce separation of concerns:

```
SIH26158-single-pass-3D/
│
├── README.md                           # Main project documentation
├── pyproject.toml                     # Project packaging and metadata config
├── .gitignore                         # Git exclusion rules
│
├── docs/                              # Project documentation
│   ├── problem_statement.md           # Details of SIH Problem Statement 26158
│   ├── architecture.md               # Detailed pipeline architecture
│   ├── environment_inventory.md       # Target system hardware/software specifications
│   └── environment_status.md          # Dependency statuses (READY, MISSING, etc.)
│
├── configs/                           # Project configuration files
│   └── default_config.json            # Default pipeline configuration
│
├── data/                              # Data directories (ignored by git except structure)
│   ├── raw/                           # Raw drone video inputs and logs
│   ├── processed/                     # Preprocessed frames, masks, and telemetry
│   ├── samples/                       # Sample/test video files
│   └── ground_truth/                  # Ground truth meshes or point clouds for comparison
│
├── src/                               # Python package source code
│   ├── __init__.py                    # Exports version, configuration and log tools
│   ├── version.py                     # Version definition
│   ├── config.py                      # JSON config loader and provider
│   ├── logger.py                      # Console and file logger configuration
│   ├── ingestion/                     # Ingestion of video feeds and drone logs
│   ├── preprocessing/                 # Frame enhancement, stabilization, and deblurring
│   ├── frame_selection/               # Sharpness assessment and keyframe filtering
│   ├── pose/                          # Camera trajectory and intrinsics solvers
│   ├── depth/                         # Dense/sparse depth estimation
│   ├── reconstruction/                # Dense point cloud and mesh generators
│   ├── sensor_fusion/                 # Merging camera outputs with GPS/IMU data
│   ├── dynamic_objects/               # Removing moving foreground elements
│   ├── occlusion/                     # Dealing with obscured regions
│   ├── confidence/                    # Point/pixel uncertainty mapping
│   ├── metrics/                       # Reconstruction evaluation utilities
│   └── visualization/                 # Renderers for 3D outputs
│
├── pipelines/                         # Executable pipelines
│   ├── baseline/                      # Baseline pipeline (B0, B1, etc.)
│   ├── experiments/                   # Proof-of-concept pipelines
│   └── production/                    # Final optimized execution pipeline
│
├── tests/                             # Unit, integration and regression tests
│   ├── test_project_structure.py      # Checks structural integrity and imports
│   ├── unit/                          # Isolated module tests
│   ├── integration/                   # Pipeline coupling tests
│   └── regression/                    # Version consistency tests
│
├── scripts/                           # Auxiliary scripts
│   └── verify_environment.py          # Verifies presence of environment requirements
│
├── notebooks/                         # Jupyter notebooks for visual analysis
│
└── outputs/                           # Generated pipeline files (ignored by git)
    ├── pointclouds/                   # Exported PLY/LAS point clouds
    ├── meshes/                        # Exported OBJ/PLY meshes
    ├── renders/                       # Output pictures, walkthrough videos
    └── reports/                       # Log files and evaluation sheets
```

---

## Setup Instructions

### 1. Clone the Repository
```powershell
git clone <repository_url>
cd SIH26158-single-pass-3D
```

### 2. Verify Your Environment
Before running the project, verify that the base environment meets the requirements by running:
```powershell
python scripts/verify_environment.py
```
This script will report whether essential tools (Python, pip, Git, PyTorch, OpenCV, pytest) are installed and ready, and print informational status messages about external binaries like COLMAP, FFmpeg, and CMake.

* Verification Script: [verify_environment.py](file:///d:/SIH26158-single-pass-3D/scripts/verify_environment.py)
* Environment Inventory: [environment_inventory.md](file:///d:/SIH26158-single-pass-3D/docs/environment_inventory.md)
* Environment Status details: [environment_status.md](file:///d:/SIH26158-single-pass-3D/docs/environment_status.md)

### 3. Install the Package
You can install this package locally in editable mode with development and testing dependencies:
```powershell
# For basic usage
pip install -e .

# For testing and development
pip install -e .[test]
```

---

## Testing Instructions

To run the structural tests verifying correct file layouts and import paths, run `pytest`:

```powershell
pytest
```

* Structural Tests File: [test_project_structure.py](file:///d:/SIH26158-single-pass-3D/tests/test_project_structure.py)
