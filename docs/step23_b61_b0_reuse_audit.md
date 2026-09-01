# B6.1: B0 Reuse Audit

## Purpose
The purpose of this audit is to identify which parts of the existing B0 COLMAP implementation (`src/reconstruction/colmap_wrapper.py` and `src/reconstruction/colmap_parser.py`) can be reused for the generic Video-to-3D engine, specifically for the automatic pose and calibration providers in B6.1.

## Identified Reusable Components

### 1. Execution Wrapper (`colmap_wrapper.py`)
- `find_colmap_executable()`: Reliable PATH/binary search logic.
- `COLMAPRunner`: Abstraction class for executing CLI commands securely.
  - `extract_features()`: Extractor wrapper handling GPU usage and SIFT defaults.
  - `match_exhaustive()`: Feature matcher.
  - `run_mapper()`: Incremental mapping wrapper.
  - `convert_model()`: BIN to TXT conversion.
- **Conversion Math**: `invert_colmap_pose()` converts from COLMAP's World-to-Camera ($X_c = R_{cw} X_w + T_{cw}$) to the required Camera-in-World ($X_w = R_{wc} X_c + C_w$) and outputs properly flipped quaternions (`q_wc`).

### 2. Output Parsers (`colmap_parser.py`)
- `parse_colmap_cameras_txt()`: Reads intrinsic camera parameters from `cameras.txt`.
- `parse_colmap_images_txt()`: Reads $Q, T$ and converts them automatically using `invert_colmap_pose()`.
- `parse_colmap_points3D_txt()`: Extracts tie-point errors and tracks.
- `compute_colmap_metrics()`: Consolidates registration rates, reprojection errors, and track lengths.

## Integration Strategy for B6.1
We will **NOT** rewrite COLMAP execution or parsing logic. Instead, we will wrap the existing `COLMAPRunner` and `colmap_parser.py` within the `ColmapPoseProvider` and `ColmapCalibrationProvider` implementations. 

We will ensure isolated execution by instantiating `COLMAPRunner` strictly with the `session/colmap/` directory path, explicitly preventing any leak into `D:\SIH26158\colmap_workspace\smoke_test` or `zurich_mav_b0`.
