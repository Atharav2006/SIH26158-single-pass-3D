# SHERBROOKE_UAV_3D Dataset Audit

## A. Acquisition & Extraction Status
* **Dataset**: Unmanned aerial image dataset: Ready for 3D reconstruction (Sherbrooke)
* **Archive**: `Part3-RawImages.zip`, `mmc2.zip`, `mmc3.zip`
* **Acquisition Status**: DOWNLOADED and extracted.
* **Extracted Path**: `D:\SIH26158-single-pass-3D\datasets\external_validation\SHERBROOKE_UAV_3D\`

## B. File Inventory & Image Statistics
* **Images**: 158 TIFF images (8-bit, 4872 x 3248).
* **Metadata Archives**:
    * `mmc2.zip` (481,028 bytes) | SHA256: `045d3a58b47d5cb3fde97839982617efc3efafd9ee1fa15b03e9469b28b36b35`
    * `mmc3.zip` (57,603,941 bytes) | SHA256: `58f3c9feb05475d3a9614763e27944a01fccebd8fdd55e33bea2e18748aeb339`
* **All Required Files Present**: Yes.

## C. Calibration Findings
* **Status**: PRESENT
* **Details**: `RefinedIOPs.txt` is present and successfully parsed. Contains 9 lines documenting focal length, principal point, pixel size, and dimensions.

## D. Pose / GPS / GCP Findings
* **Status**: PRESENT
* **Details**: 
    * `DirectEOPs.txt`: Approximate GNSS/INS direct observations for 158 images.
    * `RefinedEOPs.txt`: Photogrammetric adjusted trajectory for 158 images.
    * `TimeStamps.txt`: Accurate timing for 158 images.
    * `GCPs.txt`: 109 independently surveyed ground control points in a projected metric coordinate system.

## E. Tie Points Findings
* **Status**: PRESENT 
* **Details**: `TiePoints.txt` (41,346 observations) and `ObjectPoints.txt` (3,682 3D points) are present.

## F. Reference Geometry Findings
* **Status**: PRESENT
* **Details**: 
    * `DenseMatching.txt`: Defines a 24-image stereo sub-block suitable for dense comparison.
    * `LaserScan.xyz`: A massive 8.3 million point cloud representing the physical terrain.

## G. Provenance
* **GCPs**: Independently surveyed. Can be used as absolute scale anchors or held-out checkpoints for validation.
* **TLS Laser Scan**: Captured independently using a Faro Focus 3D scanner. Registered to the same global geodetic frame using surveyed targets. This is a TRUE independent metric ground truth.
* **Refined Poses/Object Points**: These are same-data photogrammetric outputs. They are valid for baseline software comparisons but do not constitute independent ground truth.

## H. License / Permission Findings
* **Public License**: CC BY.
* **Commercial Restriction**: None specified for CC BY.
* **SIH Project Compatibility**: Compatible for academic research and validation.

## I. Metric vs Relative Classification
* **Current Intake Status**: `READY_FOR_METRIC_VALIDATION`.
* **Validation Outcome**: With the inclusion of independent GCPs and an independent TLS point cloud, the B6 pipeline can now execute absolute metric scaling, trajectory ATE measurement, and dense point-cloud RMSE evaluations against true physical ground truth.

## J. Recommended Next Experiment
**EXPERIMENT 1: The Dense Subset Metric Validation**. 
Process the 24 images specified in `DenseMatching.txt`. Run the B6.1 automatic calibration and pose backend (constrained by the `DirectEOPs` or a small subset of `GCPs`). Output the dense 3D point cloud and quantitatively compare it against `LaserScan.xyz` (RMSE/Chamfer distance). Do not align the output to the TLS data before computing the error if absolute scale is being tested.
