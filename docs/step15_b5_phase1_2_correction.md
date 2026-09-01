# B5 Phase 1–2 Correction Report

This document records the exact fixes applied to resolve two test failures uncovered during the B5 Phase 1–2 validation process, ensuring that the codebase is completely green before proceeding to B5 Phase 3 (Metric Depth Alignment).

## 1. Camera Preprocessing Invariant Fix

### The Failure
`tests/unit/test_b5_camera_preprocessing.py::test_camera_preprocessor_initialization` failed because it contained the assertion:
`assert not np.allclose(K_source, K_rect)`

### Root Cause
This was a flawed assumption. We updated the preprocessing to use `alpha=0` (crop to valid pixels only) rather than `alpha=1`. For the specific synthetic dummy calibration in the test (and the real `FULL_OPENCV` fallback behavior), `cv2.getOptimalNewCameraMatrix` can determine that the optimal crop box is the full native image, effectively making $K_{rect} = K_{source}$.

### Exact Fix
Removed the flawed assertion. Replaced it with mathematically sound physical invariants:
* `assert K_rect.shape == (3, 3)`
* `assert np.all(np.isfinite(K_rect))`
* `assert K_rect[0, 0] > 0 and K_rect[1, 1] > 0`
* `assert K_rect[2, 0] == 0 and K_rect[2, 1] == 0 and K_rect[2, 2] == 1`

### Scientific Validity
This scientifically verifies that the generated matrix is a valid pinhole projection model without artificially forcing OpenCV to warp the focal length just to satisfy a test.

---

## 2. B4 Synthetic Sanity Test Fix

### The Failure
`tests/integration/test_b4_neural_reconstruction.py::test_synthetic_sanity_overfit` failed because the final loss equalled the initial loss exactly (`0.33333...`), meaning the network learned absolutely nothing over 200 iterations.

### Root Cause
A classic NeRF "Zero-Gradient Initialization Trap".
* The synthetic test environment placed a solid red object `[1, 0, 0]` against a pure black background `[0, 0, 0]`.
* The `VolumetricRenderer` was initialized with `bg_color=(0, 0, 0)`.
* TinyNeRF initializes volume density very close to `0.0`. Therefore, transmittance is `1.0`, and the rendered color is identical to `bg_color` (`[0, 0, 0]`).
* In NeRF volume rendering, the gradient of the loss with respect to density is proportional to `(rgb_net - bg_color)`.
* Because `rgb_net` initializes small and `bg_color` was exactly `0`, the gradient was practically `0.0`. The optimizer froze on step 1.

### Exact Fix
In `test_b4_neural_reconstruction.py`, changed the synthetic test renderer background to white:
`renderer = VolumetricRenderer(bg_color=(1.0, 1.0, 1.0)).to(device)`

### Scientific Validity
This ensures `(rgb_net - bg_color) \neq 0`, feeding strong gradients to the density network to block the background and reveal the red object. This fix operates entirely within the isolated synthetic test space; it does not change the TinyNeRF model, its architecture, or the B4 experimental results on the real Zurich dataset.

---

## 3. Test Results Before & After

**Before Fixes:**
* `pytest -q` resulted in `2 failed, 146 passed`.

**After Fixes:**
* `pytest tests/unit/test_b5_*.py -q`: `11 passed`
* `pytest tests/integration/test_b4_neural_reconstruction.py -q`: `1 passed`
* `pytest -q`: `148 passed` (Complete repository is green)

---

## 4. Camera Validation Before & After

We re-ran the exact camera validation script to confirm no physical degradation occurred to the B5 ray pipeline.

* **alpha**: `0`
* **newImgSize**: `(1920, 1080)`
* **Interpolation**: `cv2.INTER_LINEAR`
* **$K_{source}$**: Valid
* **$K_{rect}$**: Valid and mathematically bounded.

**Maximum Round-Trip Pixel Error:**
* Before: `0.000048 px`
* After: `0.000048 px`

The camera numerical pipeline is identically robust.

---

## 5. Status
All tests pass. No scientific models were weakened. B5 Metric Depth remains explicitly NOT ESTABLISHED.

**B5 PHASE 1–2: PASS**
