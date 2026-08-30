# Step 8: Pose Evaluation Conventions & Coordinate System Formulations

This document specifies the exact coordinate frames, quaternion representations, rotation directions, camera center derivations, and reference transformations utilized in the **Step 8A B0 Camera Trajectory Evaluation**.

---

## 1. COLMAP Reconstructed Pose Semantics

### 1.1 World-to-Camera Transformation ($T_{CW}$)
In standard COLMAP output (`images.txt` / `images.bin`), camera poses are represented as the transformation that maps 3D points from the world coordinate frame into the camera coordinate frame:
$$X_C = R_{CW} X_W + t_{CW}$$

* **Rotation Quaternion**: Stored in scalar-first format in `images.txt`: $\mathbf{q}_{CW} = [q_w, q_x, q_y, q_z]$.
* **Translation Vector**: $\mathbf{t}_{CW} = [t_x, t_y, t_z]^T$ (expresses the world origin in the camera coordinate frame).

### 1.2 Camera Optical Center in World Coordinates ($C_W$)
The physical optical center of the camera in reconstructed world coordinates ($C_W$) is derived by setting $X_C = \mathbf{0}$:
$$\mathbf{0} = R_{CW} C_W + t_{CW} \implies C_W = - R_{CW}^T t_{CW} = - R_{WC} t_{CW}$$

### 1.3 Camera-to-World Attitude ($R_{WC}$ / $q_{WC}$)
The camera attitude in world space is the matrix transpose / quaternion conjugate:
$$R_{WC} = R_{CW}^T = R_{CW}^{-1}$$
In standard Hamilton scalar-last quaternion representation $[q_x, q_y, q_z, q_w]$:
$$\mathbf{q}_{WC} = [-q_{CW,x}, -q_{CW,y}, -q_{CW,z}, q_{CW,w}]$$

---

## 2. Zurich Urban MAV Ground-Truth Pose Semantics

### 2.1 Spatial Reference Frame
* **Coordinate System**: WGS 84 / UTM Zone 32N (EPSG:32632)
* **Translation Units**: Metric (Meters)
* **Origin**: Global UTM projection coordinates:
  * Keyframe 1 Reference Origin: $X_0 = 465,666.057548\text{ m}, Y_0 = 5,247,973.646622\text{ m}, Z_0 = 469.019496\text{ m}$.

### 2.2 Local Metric Coordinate Normalization (Local ENU)
To avoid numerical catastrophic cancellation when calculating distances with large global UTM offsets ($\approx 5 \times 10^6\text{ m}$), ground-truth positions are expressed relative to the first evaluated ground-truth keyframe:
$$\mathbf{p}_{\text{GT,local}} = \mathbf{p}_{\text{GT,UTM}} - \mathbf{p}_{\text{GT,UTM}, 0}$$

### 2.3 Ground-Truth Orientation
Ground truth orientations in `GroundTruthAGL.csv` are defined by photogrammetrically calibrated Euler angles ($\omega, \phi, \kappa$ in degrees: yaw, pitch, roll) converted to Hamilton quaternions:
$$\mathbf{q}_{\text{GT}} = [q_x, q_y, q_z, q_w]$$

---

## 3. Transformation Alignment Pipeline: $\text{Sim}(3)$ & $\text{SE}(3)$

### 3.1 Closed-Form Umeyama Similarity Alignment ($\text{Sim}(3)$)
To align the scale-free monocular COLMAP trajectory $C_W$ to the metric ground truth $\mathbf{p}_{\text{GT,local}}$, we solve for $(s, R, t)$ minimizing:
$$\min_{s > 0, R \in SO(3), t \in \mathbb{R}^3} \frac{1}{N} \sum_{i=1}^N \|\mathbf{p}_{\text{GT,local}, i} - (s R C_{W, i} + t)\|^2$$

Where:
* $s \in \mathbb{R}^+$: Global scale ratio (COLMAP units $\to$ Meters).
* $R \in SO(3)$: $3 \times 3$ rotation matrix with $\det(R) = +1$ (pure spatial rotation, no reflections).
* $t \in \mathbb{R}^3$: $3 \times 1$ translation vector in meters.

$$\mathbf{p}_{\text{aligned}} = s R C_W + t$$

### 3.2 Pure Rigid Alignment ($\text{SE}(3)$)
To isolate and measure scale error independently without distorting metric discrepancies, rigid alignment is also evaluated with $s \equiv 1.0$:
$$\mathbf{p}_{\text{SE3}} = R_{\text{SE3}} C_W + t_{\text{SE3}}$$

---

## 4. Evaluation Error Definitions

### 4.1 Absolute Trajectory Error (ATE)
Measures the global root-mean-square positional discrepancy across all evaluated keyframes:
$$\text{ATE}_{\text{RMSE}} = \sqrt{\frac{1}{N} \sum_{i=1}^N \|\mathbf{p}_{\text{aligned}, i} - \mathbf{p}_{\text{GT}, i}\|^2}$$

### 4.2 Relative Pose Error (RPE)
Measures the local trajectory drift per step ($\Delta = 1$ interval between 1 Hz keyframes):
* **Translational RPE**:
  $$\text{RPE}_{\text{trans}, i} = \|(\mathbf{p}_{\text{aligned}, i+1} - \mathbf{p}_{\text{aligned}, i}) - (\mathbf{p}_{\text{GT}, i+1} - \mathbf{p}_{\text{GT}, i})\|$$
* **Rotational RPE**:
  $$\theta_i = \arccos\left(\frac{\text{tr}(R_{\text{aligned}, \text{rel}, i}^T R_{\text{GT}, \text{rel}, i}) - 1}{2}\right) \times \frac{180^\circ}{\pi}$$
