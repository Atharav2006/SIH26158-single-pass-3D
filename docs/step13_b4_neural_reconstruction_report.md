# SIH26158 B4-A Baseline Report: Neural/AI Reconstruction Feasibility

## 1. Objective
Determine whether a VRAM-constrained, purely photometric Neural Radiance Field (TinyNeRF) can recover useful 3D scene geometry from the Zurich Urban MAV dataset, given the severe geometric degeneracy (hover sequence, 1.7m baseline, 50m altitude) that caused the classical B3 MVS pipeline to fail.

## 2. Dataset Characteristics & Why B3 Failed
* **Sequence**: 350 frames, 11.6 seconds.
* **Baseline Degeneracy**: Maximum horizontal camera motion is ~1.7 meters.
* **B3 Failure**: Classical MVS relies on multi-view epipolar constraints. At a 50m depth scale, a 1.7m baseline yields extreme low parallax (<2°). The uncalibrated B2 trajectory error (~1.9m) eclipses the physical baseline, causing COLMAP Stereo Fusion to reject 100% of the computed depth maps.

## 3. Hardware Constraints & Selected Architecture
* **GPU**: NVIDIA RTX 3050 Laptop GPU (4.29 GB VRAM limit).
* **Selection**: Tiny Neural Radiance Field (Pure PyTorch). Heavy frameworks like Gaussian Splatting and Instant-NGP were rejected due to extreme VRAM usage during densification and complex CUDA extensions.
* **Architecture**: A lightweight 6-layer MLP with 128 hidden units. Uses positional encodings for 3D coordinates and viewing directions. B2 Poses and FULL_OPENCV intrinsics were mapped directly into unit-normalized world-space rays.

## 4. Training Configuration & Smoke Test
* **Smoke Test**: 10 frames, 128x72 resolution, 100 iterations. Passed successfully with a peak VRAM of **0.19 GB** and monotonically decreasing loss.
* **Full Run (B4-A)**: 280 Train frames, 35 Val frames, 35 Test frames at 256x144 resolution.

## 5. Scientific Results & Geometric Interpretation
* **Photometric Loss**: The TinyNeRF successfully optimized the photometric MSE loss on the training set, correctly learning to predict the RGB values of the input views.
* **Novel View Generalization**: The validation views rendered smoothly but presented high-frequency blurring ("foggy" rendering).
* **Geometric Reliability (Depth Maps)**: Unlike classical MVS which produces *no* geometry (0 points), the neural field produced a *continuous* volume. However, the density is smeared along the ray axes.
* **Conclusion vs B3**: B3 failed abruptly due to hard geometric thresholds. B4-A succeeded in creating a continuous mathematical representation, but explicitly proved that **photometric loss alone cannot resolve sharp geometry on a noisy hover sequence**. The resulting volume is physically uncertain (fog).

## 6. Next Recommended Phase
B4 verifies that the AI system requires explicit priors to resolve this sequence. The pipeline and dataset APIs are now fully established. We recommend moving to **B4-B (Neural Optimization with Depth Regularization)** or **B5 (Monocular Depth Fusion)** where learned scale-invariant depth priors (e.g., ZoeDepth/DepthAnything) are injected to supervise the neural geometry.
