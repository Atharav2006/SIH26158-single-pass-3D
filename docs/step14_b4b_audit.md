# SIH26158 B4-B Audit & Mathematical Design

## 1. Audit of Existing B4
* **Ray Generation**: OpenCV images are undistorted using `cv2.getOptimalNewCameraMatrix` and scaled. Pixel coordinates `(u, v)` yield camera-frame rays `[(u-cx)/fx, (v-cy)/fy, 1.0]`.
* **Pose Conversion**: B2 poses are `Camera-to-World` ($R_{wc}, C_w$). We compute global ray directions via $ray\_d = R_{wc} \cdot ray\_d_{cam}$ and normalize them to length 1.
* **Density Representation**: TinyNeRF MLP predicts volumetric density (ReLU-activated) from 3D coordinates.
* **Rendered Depth**: Calculated as the expected termination depth along the ray: $D = \sum w_i z_i$. Since ray directions are unit-normalized, $z_i$ represents physical metric depth (distance from camera center).
* **RGB Photometric Loss**: Standard Mean Squared Error (MSE) over RGB pixels.
* **Outputs**: Model checkpoints, diagnostic JSON files, and rendered images/depth maps (as colored PNGs). No `.ply` point cloud is exported yet.

## 2. Depth Source Selection
The Zurich MAV sequence lacks ground-truth metric depth (e.g., LiDAR). Furthermore, the classical B3 MVS pipeline failed, yielding no geometric depth.
Therefore, our only legitimate source of independent structural information is a **Monocular Relative-Depth Prior**. We will use `MiDaS_small` (via `torch.hub`). This prior is strictly *relative* (scale-ambiguous), and we must explicitly formulate a scale-invariant depth loss to avoid forcing the metric NeRF to fit arbitrary scale/shift coefficients.

## 3. Mathematical Design for B4-B
We formulate the total loss as:

$$ L_{total} = L_{rgb} + \lambda_{depth} L_{depth} + \lambda_{smooth} L_{smooth} + \lambda_{reg} L_{reg} $$

### L_rgb
Standard photometric MSE:
$$ L_{rgb} = || \hat{C} - C_{gt} ||^2 $$

### L_depth (Scale-Invariant Depth Loss)
Because the MiDaS depth $D_{midas}$ is scale-ambiguous and often inverted (disparity space), we align the expected NeRF depth $\hat{D}$ to $D_{midas}$ using a Scale-and-Shift invariant loss (e.g., Pearson correlation, or solving for scale/shift per batch).
Let $D^* = s \hat{D} + t$.
$$ L_{depth} = \frac{1}{N} \sum (\hat{D}_{normalized} - D_{midas\_normalized})^2 $$
*where normalization is zero-mean and unit-variance.*

### L_smooth (Edge-Aware Smoothness)
$$ L_{smooth} = \sum |\nabla \hat{D}| \cdot e^{-|\nabla I_{gt}|} $$
Encourages geometric continuity where the RGB image lacks edges.

### L_reg (Density Regularization)
$$ L_{reg} = \sum \log(1 + \sigma) $$
Prevents the network from filling empty space with "fog" to cheat the photometric loss.

## 4. Controlled Experiments
1. **Experiment A (B4)**: $\lambda_{depth}=0, \lambda_{smooth}=0, \lambda_{reg}=0$
2. **Experiment B (B4-B)**: $\lambda_{depth}=0.1, \lambda_{smooth}=0, \lambda_{reg}=0$
3. **Experiment C (B4-B+)**: $\lambda_{depth}=0.1, \lambda_{smooth}=0.01, \lambda_{reg}=0.001$
