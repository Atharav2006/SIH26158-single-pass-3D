import math
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Union, Tuple

def render_gps_colmap_correspondence(
    gps_pts: np.ndarray,
    colmap_pts: np.ndarray,
    output_path: Union[str, Path],
    img_width: int = 1400,
    img_height: int = 800
) -> None:
    """
    Render dual trajectory plot showing 350 GPS points (local ENU) and scaled COLMAP trajectory with correspondence lines.
    """
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    canvas = np.full((img_height, img_width, 3), 255, dtype=np.uint8)

    # Scale COLMAP roughly for visualization overlay
    c_mean = np.mean(colmap_pts, axis=0)
    c_d = colmap_pts - c_mean
    g_mean = np.mean(gps_pts, axis=0)
    
    # Simple bounding scale for plotting
    scale_factor = np.ptp(gps_pts[:, 1]) / max(1e-3, np.ptp(colmap_pts[:, 1]))
    colmap_scaled = c_d * scale_factor + g_mean

    all_pts = np.vstack([gps_pts[:, :2], colmap_scaled[:, :2]])
    min_x, min_y = np.min(all_pts, axis=0)
    max_x, max_y = np.max(all_pts, axis=0)

    span_x = max(1e-3, max_x - min_x)
    span_y = max(1e-3, max_y - min_y)

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

    # Draw correspondence tie-lines (sampled every 10 frames)
    for i in range(0, len(gps_pts), 10):
        gx, gy = to_screen(gps_pts[i, 0], gps_pts[i, 1])
        cx_pt, cy_pt = to_screen(colmap_scaled[i, 0], colmap_scaled[i, 1])
        cv2.line(canvas, (int(gx), int(gy)), (int(cx_pt), int(cy_pt)), (200, 205, 210), 1, cv2.LINE_AA)

    # Draw GPS path
    g_scr = [to_screen(p[0], p[1]) for p in gps_pts]
    for i in range(len(g_scr) - 1):
        cv2.line(canvas, (int(g_scr[i][0]), int(g_scr[i][1])), (int(g_scr[i+1][0]), int(g_scr[i+1][1])), (180, 50, 20), 2, cv2.LINE_AA)
    for sx, sy in g_scr:
        cv2.circle(canvas, (int(sx), int(sy)), 2, (180, 50, 20), -1)

    # Draw COLMAP path
    c_scr = [to_screen(p[0], p[1]) for p in colmap_scaled]
    for i in range(len(c_scr) - 1):
        cv2.line(canvas, (int(c_scr[i][0]), int(c_scr[i][1])), (int(c_scr[i+1][0]), int(c_scr[i+1][1])), (30, 140, 255), 2, cv2.LINE_AA)
    for sx, sy in c_scr:
        cv2.circle(canvas, (int(sx), int(sy)), 2, (30, 140, 255), -1)

    # Header and Legend
    cv2.putText(canvas, "STEP 9B: Image-GPS 1:1 Correspondences (350 Pairs)", (40, 45),
                cv2.FONT_HERSHEY_DUPLEX, 0.9, (25, 30, 40), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Local ENU GPS Stream (Blue) vs Reconstructed COLMAP Centers (Orange)",
                (40, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (90, 100, 110), 1, cv2.LINE_AA)

    # Legend Card
    leg_x = margin_left + plot_w - 280
    leg_y = margin_top + 15
    cv2.rectangle(canvas, (leg_x, leg_y), (leg_x + 265, leg_y + 80), (255, 255, 255), -1)
    cv2.rectangle(canvas, (leg_x, leg_y), (leg_x + 265, leg_y + 80), (190, 198, 206), 1)

    cv2.circle(canvas, (leg_x + 20, leg_y + 25), 4, (180, 50, 20), -1)
    cv2.putText(canvas, "GPS Positions (Local ENU)", (leg_x + 35, leg_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (30, 30, 30), 1)

    cv2.circle(canvas, (leg_x + 20, leg_y + 55), 4, (30, 140, 255), -1)
    cv2.putText(canvas, "COLMAP Camera Centers", (leg_x + 35, leg_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (30, 30, 30), 1)

    cv2.imwrite(str(output_path), canvas)

def render_gps_conditioning_plot(
    gps_cond: Dict[str, Any],
    colmap_cond: Dict[str, Any],
    output_path: Union[str, Path],
    img_width: int = 1400,
    img_height: int = 800
) -> None:
    """
    Render bar charts of eigenvalues, singular values, and explained variance for GPS and COLMAP point clouds.
    """
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    canvas = np.full((img_height, img_width, 3), 255, dtype=np.uint8)

    panel_w = (img_width - 140) // 2
    panel_h = img_height - 200

    p1_x1, p1_y1 = 60, 130
    p2_x1, p2_y1 = 60 + panel_w + 20, 130

    # Panel 1: GPS Explained Variance & Eigenvalues
    cv2.rectangle(canvas, (p1_x1, p1_y1), (p1_x1 + panel_w, p1_y1 + panel_h), (246, 248, 250), -1)
    cv2.rectangle(canvas, (p1_x1, p1_y1), (p1_x1 + panel_w, p1_y1 + panel_h), (190, 198, 206), 2)

    cv2.putText(canvas, "1. GPS Point Cloud Conditioning (Local ENU)", (p1_x1 + 15, p1_y1 + 30),
                cv2.FONT_HERSHEY_DUPLEX, 0.65, (25, 30, 40), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"Condition Number: {gps_cond['condition_number_svd']:.2f} | Rank: 3 (Full 3D)",
                (p1_x1 + 15, p1_y1 + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (80, 85, 95), 1, cv2.LINE_AA)

    # Bars for GPS explained variance
    var_gps = gps_cond["explained_variance_percent"]
    eval_gps = gps_cond["eigenvalues"]
    bar_w = 90

    for i in range(3):
        bx = p1_x1 + 70 + i * 160
        bh = int((panel_h - 140) * (var_gps[i] / 100.0))
        by = p1_y1 + panel_h - 40 - bh
        cv2.rectangle(canvas, (bx, by), (bx + bar_w, p1_y1 + panel_h - 40), (220, 140, 40), -1)
        cv2.rectangle(canvas, (bx, by), (bx + bar_w, p1_y1 + panel_h - 40), (180, 100, 20), 1)

        cv2.putText(canvas, f"PC {i+1}", (bx + 20, p1_y1 + panel_h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (30, 30, 30), 1)
        cv2.putText(canvas, f"{var_gps[i]:.1f}%", (bx + 15, by - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (20, 20, 20), 1)
        cv2.putText(canvas, f"λ={eval_gps[i]:.2f}", (bx + 5, by - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (90, 95, 105), 1)

    # Panel 2: COLMAP Explained Variance & Eigenvalues
    cv2.rectangle(canvas, (p2_x1, p2_y1), (p2_x1 + panel_w, p2_y1 + panel_h), (246, 248, 250), -1)
    cv2.rectangle(canvas, (p2_x1, p2_y1), (p2_x1 + panel_w, p2_y1 + panel_h), (190, 198, 206), 2)

    cv2.putText(canvas, "2. COLMAP Camera Centers Conditioning (C_w)", (p2_x1 + 15, p2_y1 + 30),
                cv2.FONT_HERSHEY_DUPLEX, 0.65, (25, 30, 40), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"Condition Number: {colmap_cond['condition_number_svd']:.2f} | Rank: 3 (Full 3D)",
                (p2_x1 + 15, p2_y1 + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (80, 85, 95), 1, cv2.LINE_AA)

    var_colmap = colmap_cond["explained_variance_percent"]
    eval_colmap = colmap_cond["eigenvalues"]

    for i in range(3):
        bx = p2_x1 + 70 + i * 160
        bh = int((panel_h - 140) * (var_colmap[i] / 100.0))
        by = p2_y1 + panel_h - 40 - bh
        cv2.rectangle(canvas, (bx, by), (bx + bar_w, p2_y1 + panel_h - 40), (40, 150, 240), -1)
        cv2.rectangle(canvas, (bx, by), (bx + bar_w, p2_y1 + panel_h - 40), (20, 110, 200), 1)

        cv2.putText(canvas, f"PC {i+1}", (bx + 20, p2_y1 + panel_h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (30, 30, 30), 1)
        cv2.putText(canvas, f"{var_colmap[i]:.1f}%", (bx + 15, by - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (20, 20, 20), 1)
        cv2.putText(canvas, f"λ={eval_colmap[i]:.2f}", (bx + 5, by - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (90, 95, 105), 1)

    # Header
    cv2.putText(canvas, "STEP 9B: Spatial Covariance & Principal Component Conditioning", (40, 45),
                cv2.FONT_HERSHEY_DUPLEX, 0.9, (25, 30, 40), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Principal Component Analysis & Variance Decomposition for Sim(3) Observability",
                (40, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (90, 100, 110), 1, cv2.LINE_AA)

    cv2.imwrite(str(output_path), canvas)

def render_sim3_noise_sensitivity(
    sensitivity_results: List[Dict[str, Any]],
    output_path: Union[str, Path],
    img_width: int = 1400,
    img_height: int = 800
) -> None:
    """
    Render sensitivity curves showing scale uncertainty and rotation error vs synthetic GPS noise level sigma.
    """
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    canvas = np.full((img_height, img_width, 3), 255, dtype=np.uint8)

    panel_w = (img_width - 140) // 2
    panel_h = img_height - 180

    p1_x1, p1_y1 = 60, 110
    p2_x1, p2_y1 = 60 + panel_w + 20, 110

    # Panel 1: Scale Standard Deviation vs Noise
    cv2.rectangle(canvas, (p1_x1, p1_y1), (p1_x1 + panel_w, p1_y1 + panel_h), (246, 248, 250), -1)
    cv2.rectangle(canvas, (p1_x1, p1_y1), (p1_x1 + panel_w, p1_y1 + panel_h), (190, 198, 206), 2)

    cv2.putText(canvas, "Scale Sensitivity vs GPS Noise Level (σ)", (p1_x1 + 15, p1_y1 + 30),
                cv2.FONT_HERSHEY_DUPLEX, 0.65, (25, 30, 40), 1, cv2.LINE_AA)

    # Grid
    for i in range(1, 5):
        gy = p1_y1 + int((panel_h - 70) * i / 5) + 30
        cv2.line(canvas, (p1_x1 + 50, gy), (p1_x1 + panel_w - 20, gy), (225, 230, 236), 1)

    sigmas = [r["noise_sigma_m"] for r in sensitivity_results]
    scale_stds = [r["scale_std"] for r in sensitivity_results]
    max_scale_std = max(scale_stds) if scale_stds else 0.05
    y_max1 = max(0.02, math.ceil(max_scale_std * 100.0) / 100.0)

    def to_p1(sig: float, val: float) -> Tuple[int, int]:
        sx = int(p1_x1 + 60 + (sig / 1.0) * (panel_w - 90))
        sy = int(p1_y1 + panel_h - 40 - (val / y_max1) * (panel_h - 90))
        return sx, sy

    pts1 = [to_p1(s, v) for s, v in zip(sigmas, scale_stds)]
    for i in range(len(pts1) - 1):
        cv2.line(canvas, pts1[i], pts1[i + 1], (180, 50, 20), 2, cv2.LINE_AA)
    for sx, sy in pts1:
        cv2.circle(canvas, (sx, sy), 4, (180, 50, 20), -1, cv2.LINE_AA)

    # X and Y axis labels
    cv2.putText(canvas, f"{y_max1:.3f}", (p1_x1 + 10, p1_y1 + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (80, 85, 95), 1)
    cv2.putText(canvas, "0.000", (p1_x1 + 10, p1_y1 + panel_h - 38), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (80, 85, 95), 1)
    cv2.putText(canvas, "GPS Noise σ (0.0 to 1.0 m)", (p1_x1 + panel_w // 2 - 80, p1_y1 + panel_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 40), 1)

    # Panel 2: Rotation Error vs Noise
    cv2.rectangle(canvas, (p2_x1, p2_y1), (p2_x1 + panel_w, p2_y1 + panel_h), (246, 248, 250), -1)
    cv2.rectangle(canvas, (p2_x1, p2_y1), (p2_x1 + panel_w, p2_y1 + panel_h), (190, 198, 206), 2)

    cv2.putText(canvas, "Rotation Sensitivity vs GPS Noise Level (σ)", (p2_x1 + 15, p2_y1 + 30),
                cv2.FONT_HERSHEY_DUPLEX, 0.65, (25, 30, 40), 1, cv2.LINE_AA)

    for i in range(1, 5):
        gy = p2_y1 + int((panel_h - 70) * i / 5) + 30
        cv2.line(canvas, (p2_x1 + 50, gy), (p2_x1 + panel_w - 20, gy), (225, 230, 236), 1)

    rot_errs = [r["rotation_error_deg_mean"] for r in sensitivity_results]
    max_rot_err = max(rot_errs) if rot_errs else 5.0
    y_max2 = max(3.0, math.ceil(max_rot_err * 1.2))

    def to_p2(sig: float, val: float) -> Tuple[int, int]:
        sx = int(p2_x1 + 60 + (sig / 1.0) * (panel_w - 90))
        sy = int(p2_y1 + panel_h - 40 - (val / y_max2) * (panel_h - 90))
        return sx, sy

    pts2 = [to_p2(s, v) for s, v in zip(sigmas, rot_errs)]
    for i in range(len(pts2) - 1):
        cv2.line(canvas, pts2[i], pts2[i + 1], (30, 140, 255), 2, cv2.LINE_AA)
    for sx, sy in pts2:
        cv2.circle(canvas, (sx, sy), 4, (30, 140, 255), -1, cv2.LINE_AA)

    cv2.putText(canvas, f"{y_max2:.1f}°", (p2_x1 + 10, p2_y1 + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (80, 85, 95), 1)
    cv2.putText(canvas, "0.0°", (p2_x1 + 10, p2_y1 + panel_h - 38), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (80, 85, 95), 1)
    cv2.putText(canvas, "GPS Noise σ (0.0 to 1.0 m)", (p2_x1 + panel_w // 2 - 80, p2_y1 + panel_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 40), 1)

    # Header
    cv2.putText(canvas, "STEP 9B: Sim(3) Alignment Numerical Noise Sensitivity Analysis", (40, 45),
                cv2.FONT_HERSHEY_DUPLEX, 0.9, (25, 30, 40), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Monte Carlo Perturbation Analysis across 50 Trials per Noise Standard Deviation",
                (40, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (90, 100, 110), 1, cv2.LINE_AA)

    cv2.imwrite(str(output_path), canvas)
