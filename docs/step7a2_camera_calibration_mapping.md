# Step 7A.2: Zurich Urban MAV Camera Distortion Model & COLMAP Calibration Mapping

This document provides the mathematical derivation, dataset source provenance, coordinate frame conventions, and verified parameter mapping between the **Zurich Urban MAV camera calibration** and **COLMAP Structure-from-Motion** camera models.

---

## 1. Source Provenance & Dataset Evidence

The camera calibration is defined directly in the official Zurich Urban MAV dataset files located at:
* **NumPy Archive**: `D:\SIH26158\datasets\zurich_mav\AGZ_subset\calibration_data.npz`
* **ROS Ingestion Script**: `D:\SIH26158\datasets\zurich_mav\AGZ_subset\write_ros_bag.py` (lines 120–132)
* **Dataset Documentation**: `D:\SIH26158\datasets\zurich_mav\AGZ_subset\readme.txt` (lines 45–53)

### Evidence from `write_ros_bag.py`:
```python
# Lines 120-132
Caminfo.D = np.asarray(data['distCoeff']).reshape(-1)
Caminfo.K = np.asarray(data['intrinsic_matrix']).reshape(-1)
Caminfo.distortion_model = 'plumb_bob'
```

In ROS (`sensor_msgs/CameraInfo`), the `plumb_bob` distortion model is explicitly defined as the standard 5-parameter OpenCV / Brown-Conrady pinhole radial-tangential lens model:
$$D = [k_1, k_2, p_1, p_2, k_3]$$

---

## 2. Source Calibration Matrices & Mathematical Formulation

### 2.1 Intrinsic Matrix $K$
```text
[[893.3901081378665,   0.0,               951.1310042974931],
 [  0.0,               898.3264861625313, 555.1335007742958],
 [  0.0,                 0.0,               1.0              ]]
```
* **Image Dimensions**: $W = 1920\text{ px}$, $H = 1080\text{ px}$
* **Horizontal Focal Length**: $f_x = 893.3901081378665\text{ px}$
* **Vertical Focal Length**: $f_y = 898.3264861625313\text{ px}$
* **Principal Point**: $c_x = 951.1310042974931\text{ px}$, $c_y = 555.1335007742958\text{ px}$

### 2.2 Distortion Coefficients Vector
```text
distCoeff = [-0.2805251302544365, 0.1158064134556822, -0.0009843367849156311, 0.0001584792476978901, -0.027021503433937236]
```
The exact 5 distortion coefficients are ordered as:
1. $k_1 = -0.2805251302544365$ (Radial distortion 1, $r^2$ coefficient)
2. $k_2 = +0.1158064134556822$ (Radial distortion 2, $r^4$ coefficient)
3. $p_1 = -0.0009843367849156311$ (Tangential distortion 1)
4. $p_2 = +0.0001584792476978901$ (Tangential distortion 2)
5. $k_3 = -0.027021503433937236$ (Radial distortion 3, $r^6$ coefficient)

### 2.3 OpenCV Pinhole + Plumb-Bob Projection Equations
Given a 3D point in camera frame $X_C = [x, y, z]^T$, let normalized coordinates be $\tilde{x} = x/z, \tilde{y} = y/z$, and $r^2 = \tilde{x}^2 + \tilde{y}^2$:
$$\begin{aligned}
\tilde{x}_{\text{dist}} &= \tilde{x} (1 + k_1 r^2 + k_2 r^4 + k_3 r^6) + 2 p_1 \tilde{x} \tilde{y} + p_2 (r^2 + 2 \tilde{x}^2) \\
\tilde{y}_{\text{dist}} &= \tilde{y} (1 + k_1 r^2 + k_2 r^4 + k_3 r^6) + p_1 (r^2 + 2 \tilde{y}^2) + 2 p_2 \tilde{x} \tilde{y} \\
u &= f_x \tilde{x}_{\text{dist}} + c_x \\
v &= f_y \tilde{y}_{\text{dist}} + c_y
\end{aligned}$$

---

## 3. Pixel Coordinate & Principal Point Convention

### 3.1 Source Convention (OpenCV / Kalibr)
* Origin $(0.0, 0.0)$ is defined at the **top-left outer corner** of the top-left pixel.
* The optical center of pixel $(0, 0)$ is at coordinate $(0.5, 0.5)$.
* Continuous coordinates span $[0, W] \times [0, H] = [0, 1920] \times [0, 1080]$.

### 3.2 Target Convention (COLMAP)
* COLMAP similarly defines origin $(0.0, 0.0)$ at the **top-left outer corner** of the image grid.
* The center of the top-left pixel is located at $(0.5, 0.5)$.

### 3.3 Principal Point Shift Calculation
$$\Delta c_x = c_{x,\text{colmap}} - c_{x,\text{opencv}} = 0.0000\text{ px}$$
$$\Delta c_y = c_{y,\text{colmap}} - c_{y,\text{opencv}} = 0.0000\text{ px}$$
* **Conclusion**: No coordinate translation or index adjustment is required. The raw values $c_x = 951.1310042974931, c_y = 555.1335007742958$ map directly into COLMAP.

---

## 4. Evaluation of COLMAP Camera Models & Impact of $k_3$

### 4.1 Significance of the $k_3$ Higher-Order Radial Term
The Zurich MAV camera utilizes a wide-angle lens (horizontal FOV $\approx 94^\circ$, diagonal FOV $\approx 102^\circ$). At the sensor corners:
$$r^2 = \left(\frac{1920 - 951.13}{893.39}\right)^2 + \left(\frac{1080 - 555.13}{898.33}\right)^2 \approx 1.176 + 0.341 = 1.5175$$
At $r^2 \approx 1.52$, the $k_3 r^6$ contribution is:
$$\Delta_{\text{radial}, k_3} = k_3 (r^2)^3 = -0.0270215 \times (1.5175)^3 \approx -0.09442$$
This contributes approximately **$103.9\text{ pixels}$ of geometric correction** at the sensor perimeter.

### 4.2 Model Selection Comparison

| Feature / Model | `OPENCV` (Model ID 6) | `FULL_OPENCV` (Model ID 8) |
| :--- | :---: | :---: |
| **Total Parameters** | 8 | 12 |
| **Parameter Ordering** | `fx, fy, cx, cy, k1, k2, p1, p2` | `fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6` |
| **Supports $k_3$?** | ❌ Discarded | ✅ Preserved ($k_3 = -0.0270215$) |
| **Higher Order ($k_4, k_5, k_6$)** | N/A | Set to `0.0, 0.0, 0.0` |
| **Corner Ray Accuracy** | Degraded by $\sim 100\text{ px}$ | Exact ($< 0.01\text{ px}$) |
| **Mapper Bundle Adjustment** | Refines $(f_x, f_y, k_1, k_2, p_1, p_2)$ | Refines $(f_x, f_y, k_1 \dots k_6)$ |
| **Primary Recommendation** | Simplified baseline / Smoke test | **Primary Full Baseline Model** |

---

## 5. Comprehensive Parameter Mapping Table

| Source Parameter | Physical Meaning | Source Value (Unrounded) | COLMAP Parameter | `FULL_OPENCV` Index | Conversion Applied |
| :--- | :--- | :--- | :--- | :---: | :--- |
| `intrinsic_matrix[0,0]` | Horizontal Focal Length ($f_x$) | `893.3901081378665` | `fx` | 0 | None (Exact) |
| `intrinsic_matrix[1,1]` | Vertical Focal Length ($f_y$) | `898.3264861625313` | `fy` | 1 | None (Exact) |
| `intrinsic_matrix[0,2]` | Principal Point X ($c_x$) | `951.1310042974931` | `cx` | 2 | None ($0.0\text{ px}$ shift) |
| `intrinsic_matrix[1,2]` | Principal Point Y ($c_y$) | `555.1335007742958` | `cy` | 3 | None ($0.0\text{ px}$ shift) |
| `distCoeff[0]` | Radial Distortion 1 ($k_1$) | `-0.2805251302544365` | `k1` | 4 | None (Exact) |
| `distCoeff[1]` | Radial Distortion 2 ($k_2$) | `0.1158064134556822` | `k2` | 5 | None (Exact) |
| `distCoeff[2]` | Tangential Distortion 1 ($p_1$) | `-0.0009843367849156311` | `p1` | 6 | None (Exact) |
| `distCoeff[3]` | Tangential Distortion 2 ($p_2$) | `0.0001584792476978901` | `p2` | 7 | None (Exact) |
| `distCoeff[4]` | Radial Distortion 3 ($k_3$) | `-0.027021503433937236` | `k3` | 8 | None (Preserved in `FULL_OPENCV`) |
| *N/A* | Radial Distortion 4 ($k_4$) | `0.0` | `k4` | 9 | Assigned constant `0.0` |
| *N/A* | Radial Distortion 5 ($k_5$) | `0.0` | `k5` | 10 | Assigned constant `0.0` |
| *N/A* | Radial Distortion 6 ($k_6$) | `0.0` | `k6` | 11 | Assigned constant `0.0` |

---

## 6. Exact COLMAP CLI Parameter Strings

### Recommended Model: `FULL_OPENCV`
```text
--ImageReader.camera_model FULL_OPENCV
--ImageReader.camera_params "893.3901081378665,898.3264861625313,951.1310042974931,555.1335007742958,-0.2805251302544365,0.1158064134556822,-0.0009843367849156311,0.0001584792476978901,-0.027021503433937236,0,0,0"
```

### Simplified Model: `OPENCV`
```text
--ImageReader.camera_model OPENCV
--ImageReader.camera_params "893.3901081378665,898.3264861625313,951.1310042974931,555.1335007742958,-0.2805251302544365,0.1158064134556822,-0.0009843367849156311,0.0001584792476978901"
```

---

## 7. Machine-Readable Artifact

Created at [outputs/reports/zurich_mav/camera_calibration_mapping.json](file:///d:/SIH26158-single-pass-3D/outputs/reports/zurich_mav/camera_calibration_mapping.json).
