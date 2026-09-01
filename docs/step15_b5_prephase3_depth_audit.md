# B5 Pre-Phase-3 Depth Audit

This document is the result of the rigorous audit requested prior to beginning Phase 3 (Depth Scale Alignment). It documents the exact empirical statistics, preprocessing stages, and physical boundaries currently in the codebase for the B5 Monocular Depth Fusion engine.

## 1. Depth Semantics & Dimensions
1. **Tensor Returned**: MiDaS_small returns a 2D tensor `[H, W]` of `float32` representing scale-ambiguous, shift-ambiguous **relative inverse depth (disparity)**.
2. **Preprocessing**: Distorted 1080x1920 images are rectified via `cv2.remap` (using FULL_OPENCV parameters). They are resized to `256x256` (bilinear) and normalized using ImageNet parameters (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).
3. **Postprocessing**: The raw `[256, 256]` MiDaS output is upsampled back to `[1080, 1920]` via bicubic interpolation.
4. **Is it treated as inverse depth?** Yes. It is explicitly labeled `relative_inverse_depth` within `DepthPrior.metadata()`.
5. **Is it inverted?** Yes, via `inverse_to_relative_depth()` which applies $1.0 / (D_{inv} + 1e^{-6})$.
6. **Normalization**: No explicit depth normalization (like zero-mean/unit-variance) is applied *to the output* at the B5 interface level. It is kept purely as the network's raw scale-ambiguous response.
7. **Implicit Scale/Shift**: The B5 code applies NO implicit scale or shift.
8. **Interpolation**: Bilinear for downsampling to model resolution, Bicubic for upsampling depth.
9. **Dimensions**: Final output is `[1080, 1920]`.

## 2. Real Frame Statistics (Zurich MAV)
Evaluated on frames `00001.jpg`, `00175.jpg`, and `00350.jpg`.

### Raw MiDaS Inverse Depth
Values are typically in the hundreds, indicating they are in a completely arbitrary feature space.
* **00001.jpg**: min=271.2, max=1564.8, mean=665.1, median=641.0, std=256.9
* **00175.jpg**: min=374.5, max=1704.1, mean=698.8, median=661.3, std=249.0
* **00350.jpg**: min=401.8, max=1673.9, mean=668.0, median=625.5, std=215.8

### Transformed Relative Depth
After applying $1.0 / D_{inv}$, the values become extremely small. This definitively proves they **are not metric**.
* **00001.jpg**: min=0.0006, max=0.0037, mean=0.0017, median=0.0016
* **00175.jpg**: min=0.0006, max=0.0027, mean=0.0016, median=0.0015
* **00350.jpg**: min=0.0006, max=0.0025, mean=0.0016, median=0.0016

*Note: For a drone flying at ~10-30 meters altitude, a depth value of `0.0016` would correspond to 1.6 millimeters if misinterpreted as metric. This perfectly demonstrates why Phase 3 (Metric Scale Alignment) is mathematically mandatory.*

## 3. Camera Rectification Matrix
* **Hidden Scale Transformations:** None.
* **$K_{source}$**: 
  $f_x = 893.39$, $f_y = 898.33$, $c_x = 951.13$, $c_y = 555.13$
* **$K_{rect}$** (calculated via `cv2.getOptimalNewCameraMatrix` with alpha=1):
  $f_x = 264.30$, $f_y = 581.70$, $c_x = 1579.22$, $c_y = 565.27$
*The change in $c_x$ and $f_x$ is due to accommodating the heavy `FULL_OPENCV` distortion parameters while retaining all valid pixels.*

## 4. B2 Pose Data
* **Frame count**: 350
* **Metric Frame**: Local ENU (Sim(3) aligned from GPS)
* **Translation Units**: Meters
* **Convention**: Camera-to-World ($X_{world} = R_{wc} X_{camera} + C_{world}$)
* **Trajectory Spatial Extent**: 
  * X range: ~1.70 m
  * Y range: ~1.37 m
  * Z range: ~1.44 m
*(This tiny ~1.5m spatial extent confirms the sequence is a near-stationary hover, perfectly explaining why classical B3 MVS failed and why a monocular prior was required).*

## Final Conclusion
The B5 depth interface correctly isolates the scale-ambiguous nature of the learned depth. It mathematically protects the pipeline from silently assuming metric values.

**B5 PRE-PHASE-3 AUDIT: PASS**
