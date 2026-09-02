# IIT_GANDHINAGAR_UAV_PHOTOGRAMMETRY Dataset Audit

## A. Dataset Identity & Acquisition Status
* **Dataset**: IIT Gandhinagar UAV Photogrammetry Dataset
* **Source URL**: https://data.mendeley.com/datasets/nmxysw8ybx/1
* **Acquisition Status**: DOWNLOADED and extracted.
* **Archive Path**: `D:\SIH26158-single-pass-3D\datasets\india_validation\IIT_GANDHINAGAR_UAV_PHOTOGRAMMETRY\Utilizing UAV-based photogrammetry to develop a 3D Model of IITGN.zip`
* **Extracted Path**: `D:\SIH26158-single-pass-3D\datasets\india_validation\IIT_GANDHINAGAR_UAV_PHOTOGRAMMETRY\extracted\`

## B. File Inventory & Image Statistics
* **ZIP Size**: 2.42 GB (2,422,669,983 bytes)
* **ZIP Hash (SHA256)**: `390b185a0581cc6f0ed4a3b7e1f7a3a781bfafef85d6e64ec3db1b91f8bde5d9`
* **Total files**: 2
* **Image count (Input UAV)**: **0**. The archive contains absolutely no input UAV photographs.
* **Corrupted files**: The orthophoto is a 1.55 billion pixel JPEG (535 MB), which exceeds standard OpenCV/PIL decompression bomb limits.

## C. Calibration Findings
* **Status**: NOT_AVAILABLE. (No images, no metadata, no calibration reports).

## D. Pose / GPS / GCP Findings
* **Status**: NOT_AVAILABLE. (The public description states GCPs were used, but no GCP coordinates or pose files are included in the archive).

## E. Reference Geometry Findings
* **Status**: SAME-DATA RECONSTRUCTION PRODUCT.
* **Files**:
    * `IITGN_Campus_3D_Model.bin`: CloudCompare Binary format 3D model (4.6 GB).
    * `IITGN_Ortho_Photograph.jpg`: Massive JPEG orthophoto (535 MB).
* **Provenance**: The public description explicitly states the 3D model was developed *by utilizing the UAV-based photogrammetry*. Therefore, it is a derived reconstruction product generated from the 4,100 missing images, NOT an independent ground-truth survey (like terrestrial laser scanning).
* **Validation Suitability**: UNUSABLE for metric evaluation.

## F. License / Permission Findings
* **Public License**: CC BY-NC 3.0. No LICENSE file is included in the ZIP.
* **Commercial Restriction**: Yes (Non-Commercial).

## G. Metric vs Relative Classification
* **Current Intake Status**: `NOT_READY`.
* **Validation Outcome**: The dataset fails intake validation. It has 0 valid input images to reconstruct, and its reference geometry is a same-data derived product without independent GCP files. It cannot be used for any B6.3 reconstruction experiment.

## H. Recommended Next Experiment
DO NOT run reconstruction on this dataset. It is invalid for Video-to-3D evaluation. Discard from active pipeline tests and proceed to evaluate the `UASG2023_DELHI_DENSE_URBAN` or `UASG2019_ROORKEE` datasets once obtained.
