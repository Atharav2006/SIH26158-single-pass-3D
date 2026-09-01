# Phase 0: Audit of B4-B for Monocular Depth Fusion (B5)

This document satisfies the explicit requirements of **Phase 0** in the B5 master prompt. Before implementing the multi-frame Monocular Depth Fusion point-cloud engine, we must explicitly verify the B4-B interfaces and data formats to ensure physical and mathematical consistency.

## 1. Depth Source
* **Model Used:** `MiDaS_small` loaded via `torch.hub`.
* **Output Semantics:** The output is **relative inverse depth** (disparity space). High values indicate objects closer to the camera; lower values indicate objects farther away.
* **Metric Nature:** Strictly **relative**. The depth maps are scale-ambiguous and shift-ambiguous. It is mathematically invalid to directly assign `meters = midas_output`.

## 2. Image Preprocessing & Conventions
* **Camera Model:** `FULL_OPENCV` (1920x1080 native resolution).
* **Undistortion:** `dataset.py` explicitly applies `cv2.remap` using `cv2.getOptimalNewCameraMatrix` to produce rectilinear/pinhole-equivalent images before any depth inference.
* **Resolution:** In B4-B, images were downscaled to `256x144` for training. B5 will need to configure its own resolution.
* **Normalization:** ImageNet mean (`[0.485, 0.456, 0.406]`) and standard deviation (`[0.229, 0.224, 0.225]`) are explicitly applied to the `[0, 1]` RGB tensors inside `MiDaSDepthPrior` before inference.

## 3. Pose & Ray Conventions
* **Poses used:** B2 fused metric trajectory (`b2_fused_trajectory.csv`).
* **Format:** Camera-to-World transform. `X_w = R_wc * X_c + C_w`. 
* **Camera Coordinates:** OpenCV standard convention (`+X` Right, `+Y` Down, `+Z` Forward). Ray origins are $C_w$, and world directions are $R_{wc} \cdot [ (u-c_x)/f_x, (v-c_y)/f_y, 1 ]^T$.

## 4. Depth Loss Formulation (B4-B)
In B4-B, the depth loss was defined in `src/neural_reconstruction/depth_loss.py`. Because MiDaS depth is relative, the formulation solved a per-batch least-squares alignment using zero-mean, unit-variance standardization:
$$ s \cdot D_{nerf} + t \approx D_{midas} $$
The final loss was normalized by the variance of $D_{midas}$ to bound it as a Pearson-like correlation loss between `[0, 1]`.

## 5. B4-B Results Measured
From `b4b_experiment_comparison.json`:
* **B4 (Photometric):** Peak VRAM 0.89 GB, Loss ~0.096. 
* **B4-B (+Depth):** Peak VRAM 1.05 GB, Loss ~1.079. 
* **B4-B+ (+Depth + Smoothness):** Peak VRAM 1.05 GB, Loss ~1.079.

*Note: The depth loss in B4-B converged to roughly `0.999` while the RGB loss converged successfully.*

## Conclusion for B5 Design
Because MiDaS depth is **relative inverse depth**, Phase 3 (Depth Scale Alignment) and Phase 5 (Unprojection) of the B5 engine cannot simply unproject $1 / D_{midas}$. We must explicitly invert it and then calculate the true metric scale ($s$) and shift ($t$) to align it to the B2 trajectory frame (meters) before generating the dense metric point cloud.
