import math
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Union, Tuple

def render_b2_trajectory_comparison(
    b1_pts: np.ndarray,
    b2_pts: np.ndarray,
    gt_pts: np.ndarray,
    gps_pts: np.ndarray,
    output_path: Union[str, Path],
    img_width: int = 1400,
    img_height: int = 900
) -> None:
    """
    Render 2D top-down comparison of B1 (GPS-georeferenced), B2 (Fused), Raw GPS, and GT keyframes.
    """
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    canvas = np.full((img_height, img_width, 3), 255, dtype=np.uint8)

    all_pts = np.vstack([b1_pts[:, :2], b2_pts[:, :2], gt_pts[:, :2], gps_pts[:, :2]])
    min_x, min_y = np.min(all_pts, axis=0)
    max_x, max_y = np.max(all_pts, axis=0)

    span_x = max(1.0, max_x - min_x)
    span_y = max(1.0, max_y - min_y)

    margin_left, margin_right = 100, 80
    margin_top, margin_bottom = 110, 80
    plot_w = img_width - margin_left - margin_right
    plot_h = img_height - margin_top - margin_bottom

    scale = min(plot_w / span_x, plot_h / span_y) * 0.85
    cx = (min_x + max_x) / 2.0
    cy = (min_y + max_y) / 2.0
    canvas_cx = margin_left + plot_w / 2.0
    canvas_cy = margin_top + plot_h / 2.0

    def to_screen(x: float, y: float) -> Tuple[int, int]:
        return int(canvas_cx + (float(x) - cx) * scale), int(canvas_cy - (float(y) - cy) * scale)

    # Background and Grid
    cv2.rectangle(canvas, (margin_left, margin_top), (margin_left + plot_w, margin_top + plot_h), (244, 246, 248), -1)
    cv2.rectangle(canvas, (margin_left, margin_top), (margin_left + plot_w, margin_top + plot_h), (190, 198, 206), 2)

    for i in range(1, 6):
        lx = int(margin_left + int(plot_w * i / 6))
        ly = int(margin_top + int(plot_h * i / 6))
        cv2.line(canvas, (lx, margin_top), (lx, margin_top + plot_h), (225, 230, 236), 1)
        cv2.line(canvas, (margin_left, ly), (margin_left + plot_w, ly), (225, 230, 236), 1)

    # 1. Raw GPS (Blue)
    g_scr = [to_screen(p[0], p[1]) for p in gps_pts]
    for i in range(len(g_scr) - 1):
        cv2.line(canvas, g_scr[i], g_scr[i + 1], (180, 80, 20), 1, cv2.LINE_AA)
    for sx, sy in g_scr:
        cv2.circle(canvas, (sx, sy), 2, (180, 80, 20), -1)

    # 2. B1 Georeferenced (Orange)
    b1_scr = [to_screen(p[0], p[1]) for p in b1_pts]
    for i in range(len(b1_scr) - 1):
        cv2.line(canvas, b1_scr[i], b1_scr[i + 1], (40, 140, 255), 2, cv2.LINE_AA)

    # 3. B2 Fused Trajectory (Green)
    b2_scr = [to_screen(p[0], p[1]) for p in b2_pts]
    for i in range(len(b2_scr) - 1):
        cv2.line(canvas, b2_scr[i], b2_scr[i + 1], (34, 139, 34), 3, cv2.LINE_AA)
    for sx, sy in b2_scr[::10]:
        cv2.circle(canvas, (sx, sy), 3, (34, 139, 34), -1)

    # 4. Ground Truth Keyframes (Red)
    gt_scr = [to_screen(p[0], p[1]) for p in gt_pts]
    for sx, sy in gt_scr:
        cv2.circle(canvas, (sx, sy), 6, (0, 0, 220), -1, cv2.LINE_AA)
        cv2.circle(canvas, (sx, sy), 8, (0, 0, 220), 1, cv2.LINE_AA)

    # Header and Legend
    cv2.putText(canvas, "STEP 12A: Baseline B2 Trajectory Fusion (Visual + GPS + IMU)", (40, 45),
                cv2.FONT_HERSHEY_DUPLEX, 0.9, (25, 30, 40), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Multimodal Fusion Trajectory (Green) vs B1 GPS Georeferenced (Orange) vs Ground Truth (Red)",
                (40, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (90, 100, 110), 1, cv2.LINE_AA)

    # Legend Card
    leg_x = margin_left + plot_w - 320
    leg_y = margin_top + 15
    cv2.rectangle(canvas, (leg_x, leg_y), (leg_x + 305, leg_y + 110), (255, 255, 255), -1)
    cv2.rectangle(canvas, (leg_x, leg_y), (leg_x + 305, leg_y + 110), (190, 198, 206), 1)

    entries = [
        ("Raw GPS Fixes", (180, 80, 20), 2, 25),
        ("B1 GPS Georeferenced", (40, 140, 255), 2, 50),
        ("B2 Fused Trajectory", (34, 139, 34), 3, 75),
        ("Ground Truth Keyframes", (0, 0, 220), 6, 100)
    ]
    for label, col, rad, y_off in entries:
        cv2.circle(canvas, (leg_x + 20, leg_y + y_off), rad, col, -1)
        cv2.putText(canvas, label, (leg_x + 35, leg_y + y_off + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (30, 30, 30), 1)

    cv2.imwrite(str(output_path), canvas)

def render_b2_sensor_residuals(
    vis_res: np.ndarray,
    gps_res: np.ndarray,
    imu_res: np.ndarray,
    output_path: Union[str, Path],
    img_width: int = 1400,
    img_height: int = 800
) -> None:
    """
    Render comparative bar charts of residual RMSE across the 3 sensor modalities.
    """
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    canvas = np.full((img_height, img_width, 3), 255, dtype=np.uint8)

    margin_left, margin_right = 100, 60
    margin_top, margin_bottom = 110, 80
    plot_w = img_width - margin_left - margin_right
    plot_h = img_height - margin_top - margin_bottom

    cv2.rectangle(canvas, (margin_left, margin_top), (margin_left + plot_w, margin_top + plot_h), (246, 248, 250), -1)
    cv2.rectangle(canvas, (margin_left, margin_top), (margin_left + plot_w, margin_top + plot_h), (190, 198, 206), 2)

    categories = [
        ("Visual Relative Factor", float(np.sqrt(np.mean(vis_res**2))), "norm", (40, 140, 255)),
        ("GPS Position Factor", float(np.sqrt(np.mean(gps_res**2))), "meters", (180, 80, 20)),
        ("IMU Preintegration Factor", float(np.sqrt(np.mean(imu_res**2))), "norm", (34, 139, 34))
    ]

    max_val = max(v for _, v, _, _ in categories) * 1.35
    bar_w = 140
    spacing = plot_w // 3

    for i, (name, val, unit, color) in enumerate(categories):
        bx = margin_left + int((i + 0.5) * spacing) - bar_w // 2
        bh = int(plot_h * (val / max_val))
        by = margin_top + plot_h - bh - 2

        cv2.rectangle(canvas, (bx, by), (bx + bar_w, margin_top + plot_h - 2), color, -1)
        cv2.rectangle(canvas, (bx, by), (bx + bar_w, margin_top + plot_h - 2), (40, 40, 40), 1)

        cv2.putText(canvas, f"RMSE = {val:.3f} {unit}", (bx - 5, by - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1)
        cv2.putText(canvas, name, (bx - 15, margin_top + plot_h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (30, 30, 30), 1)

    # Header
    cv2.putText(canvas, "STEP 12A: Multimodal Sensor Residual Error Distributions", (40, 45),
                cv2.FONT_HERSHEY_DUPLEX, 0.9, (25, 30, 40), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Optimized Residuals Across Visual Relative Constraints, GPS Observations, and IMU Dynamics",
                (40, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (90, 100, 110), 1, cv2.LINE_AA)

    cv2.imwrite(str(output_path), canvas)

def render_b2_gps_robustness(
    noise_results: List[Dict[str, Any]],
    dropout_results: List[Dict[str, Any]],
    output_path: Union[str, Path],
    img_width: int = 1400,
    img_height: int = 800
) -> None:
    """
    Render GPS noise sensitivity and GPS dropout tolerance curves.
    """
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    canvas = np.full((img_height, img_width, 3), 255, dtype=np.uint8)

    panel_w = (img_width - 140) // 2
    panel_h = img_height - 180

    p1_x1, p1_y1 = 60, 110
    p2_x1, p2_y1 = 60 + panel_w + 20, 110

    # Panel 1: GPS Noise Sensitivity
    cv2.rectangle(canvas, (p1_x1, p1_y1), (p1_x1 + panel_w, p1_y1 + panel_h), (246, 248, 250), -1)
    cv2.rectangle(canvas, (p1_x1, p1_y1), (p1_x1 + panel_w, p1_y1 + panel_h), (190, 198, 206), 2)
    cv2.putText(canvas, "1. ATE vs Injected GPS Noise (σ)", (p1_x1 + 15, p1_y1 + 30),
                cv2.FONT_HERSHEY_DUPLEX, 0.65, (25, 30, 40), 1)

    sigmas = [r["gps_noise_sigma_m"] for r in noise_results]
    ates = [r["ate_rmse_m"] for r in noise_results]
    max_ate = max(ates) * 1.25

    def to_p1(sig: float, val: float) -> Tuple[int, int]:
        sx = int(p1_x1 + 60 + (sig / 2.0) * (panel_w - 90))
        sy = int(p1_y1 + panel_h - 40 - (val / max_ate) * (panel_h - 90))
        return sx, sy

    pts1 = [to_p1(s, v) for s, v in zip(sigmas, ates)]
    for i in range(len(pts1) - 1):
        cv2.line(canvas, pts1[i], pts1[i + 1], (180, 50, 20), 2, cv2.LINE_AA)
    for sx, sy in pts1:
        cv2.circle(canvas, (sx, sy), 4, (180, 50, 20), -1)

    cv2.putText(canvas, "GPS Noise σ (0.1m - 2.0m)", (p1_x1 + panel_w // 2 - 80, p1_y1 + panel_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 40), 1)

    # Panel 2: GPS Dropout Tolerance
    cv2.rectangle(canvas, (p2_x1, p2_y1), (p2_x1 + panel_w, p2_y1 + panel_h), (246, 248, 250), -1)
    cv2.rectangle(canvas, (p2_x1, p2_y1), (p2_x1 + panel_w, p2_y1 + panel_h), (190, 198, 206), 2)
    cv2.putText(canvas, "2. ATE vs GPS Dropout Outage Duration", (p2_x1 + 15, p2_y1 + 30),
                cv2.FONT_HERSHEY_DUPLEX, 0.65, (25, 30, 40), 1)

    durs = [r["dropout_duration_s"] for r in dropout_results]
    ates_drop = [r["ate_rmse_m"] for r in dropout_results]
    max_drop_ate = max(ates_drop) * 1.25

    def to_p2(dur: float, val: float) -> Tuple[int, int]:
        sx = int(p2_x1 + 60 + (dur / 5.0) * (panel_w - 90))
        sy = int(p2_y1 + panel_h - 40 - (val / max_drop_ate) * (panel_h - 90))
        return sx, sy

    pts2 = [to_p2(d, v) for d, v in zip(durs, ates_drop)]
    for i in range(len(pts2) - 1):
        cv2.line(canvas, pts2[i], pts2[i + 1], (34, 139, 34), 2, cv2.LINE_AA)
    for sx, sy in pts2:
        cv2.circle(canvas, (sx, sy), 4, (34, 139, 34), -1)

    cv2.putText(canvas, "Dropout Duration (1.0s - 5.0s)", (p2_x1 + panel_w // 2 - 80, p2_y1 + panel_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 40), 1)

    # Header
    cv2.putText(canvas, "STEP 12A: B2 Fusion Robustness under GPS Noise & Outages", (40, 45),
                cv2.FONT_HERSHEY_DUPLEX, 0.9, (25, 30, 40), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Visual-Inertial Dead-Reckoning Hold During Synthetic GNSS Degradation and Blackouts",
                (40, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (90, 100, 110), 1, cv2.LINE_AA)

    cv2.imwrite(str(output_path), canvas)

def render_b2_imu_robustness(
    bias_results: List[Dict[str, Any]],
    output_path: Union[str, Path],
    img_width: int = 1200,
    img_height: int = 700
) -> None:
    """
    Render bar chart of ATE RMSE across small, medium, and large IMU bias perturbations.
    """
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    canvas = np.full((img_height, img_width, 3), 255, dtype=np.uint8)

    margin_left, margin_right = 100, 60
    margin_top, margin_bottom = 110, 80
    plot_w = img_width - margin_left - margin_right
    plot_h = img_height - margin_top - margin_bottom

    cv2.rectangle(canvas, (margin_left, margin_top), (margin_left + plot_w, margin_top + plot_h), (246, 248, 250), -1)
    cv2.rectangle(canvas, (margin_left, margin_top), (margin_left + plot_w, margin_top + plot_h), (190, 198, 206), 2)

    categories = [(r["perturbation_level"], r["ate_rmse_m"]) for r in bias_results]
    max_val = max(v for _, v in categories) * 1.3
    bar_w = 120
    spacing = plot_w // len(categories)

    colors = [(34, 139, 34), (40, 140, 255), (180, 50, 20)]

    for i, ((name, val), color) in enumerate(zip(categories, colors)):
        bx = margin_left + int((i + 0.5) * spacing) - bar_w // 2
        bh = int(plot_h * (val / max_val))
        by = margin_top + plot_h - bh - 2

        cv2.rectangle(canvas, (bx, by), (bx + bar_w, margin_top + plot_h - 2), color, -1)
        cv2.rectangle(canvas, (bx, by), (bx + bar_w, margin_top + plot_h - 2), (40, 40, 40), 1)

        cv2.putText(canvas, f"{val:.3f} m", (bx + 15, by - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (20, 20, 20), 1)
        cv2.putText(canvas, name.capitalize() + " Perturbation", (bx - 20, margin_top + plot_h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (30, 30, 30), 1)

    cv2.putText(canvas, "STEP 12A: B2 Trajectory Accuracy under IMU Bias Perturbations", (40, 45),
                cv2.FONT_HERSHEY_DUPLEX, 0.9, (25, 30, 40), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Evaluation of Visual-GPS Anchor Stability Against Uncalibrated Inertial Drift",
                (40, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (90, 100, 110), 1, cv2.LINE_AA)

    cv2.imwrite(str(output_path), canvas)
