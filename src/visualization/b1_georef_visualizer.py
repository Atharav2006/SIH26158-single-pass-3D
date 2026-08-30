import math
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Union, Tuple

def render_b1_gps_georeferenced_trajectory(
    gps_pts: np.ndarray,
    b1_metric_pts: np.ndarray,
    output_path: Union[str, Path],
    img_width: int = 1400,
    img_height: int = 900
) -> None:
    """
    Render top-down comparison of raw GPS stream vs B1 Georeferenced COLMAP camera trajectory.
    """
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    canvas = np.full((img_height, img_width, 3), 255, dtype=np.uint8)

    all_pts = np.vstack([gps_pts[:, :2], b1_metric_pts[:, :2]])
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

    # Draw GPS path (Cyan/Blue)
    g_scr = [to_screen(p[0], p[1]) for p in gps_pts]
    for i in range(len(g_scr) - 1):
        cv2.line(canvas, (g_scr[i][0], g_scr[i][1]), (g_scr[i+1][0], g_scr[i+1][1]), (180, 70, 20), 2, cv2.LINE_AA)
    for sx, sy in g_scr:
        cv2.circle(canvas, (sx, sy), 2, (180, 70, 20), -1)

    # Draw B1 Georeferenced path (Orange/Red)
    b1_scr = [to_screen(p[0], p[1]) for p in b1_metric_pts]
    for i in range(len(b1_scr) - 1):
        cv2.line(canvas, (b1_scr[i][0], b1_scr[i][1]), (b1_scr[i+1][0], b1_scr[i+1][1]), (30, 140, 255), 3, cv2.LINE_AA)
    for sx, sy in b1_scr:
        cv2.circle(canvas, (sx, sy), 2, (30, 140, 255), -1)

    # Start and End
    cv2.circle(canvas, (b1_scr[0][0], b1_scr[0][1]), 8, (34, 139, 34), -1, cv2.LINE_AA)
    cv2.putText(canvas, "START", (b1_scr[0][0] + 12, b1_scr[0][1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (34, 139, 34), 2)

    cv2.circle(canvas, (b1_scr[-1][0], b1_scr[-1][1]), 8, (0, 0, 220), -1, cv2.LINE_AA)
    cv2.putText(canvas, "END (Frame 350)", (b1_scr[-1][0] + 12, b1_scr[-1][1] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 220), 2)

    # Header and Legend
    cv2.putText(canvas, "STEP 9C: Baseline B1 GPS Metric Georeferenced Camera Trajectory", (40, 45),
                cv2.FONT_HERSHEY_DUPLEX, 0.9, (25, 30, 40), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Raw GPS Stream (Blue) vs B1 Sim(3) Georeferenced COLMAP Trajectory (Orange)",
                (40, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (90, 100, 110), 1, cv2.LINE_AA)

    # Legend Card
    leg_x = margin_left + plot_w - 300
    leg_y = margin_top + 15
    cv2.rectangle(canvas, (leg_x, leg_y), (leg_x + 285, leg_y + 75), (255, 255, 255), -1)
    cv2.rectangle(canvas, (leg_x, leg_y), (leg_x + 285, leg_y + 75), (190, 198, 206), 1)

    cv2.circle(canvas, (leg_x + 20, leg_y + 25), 4, (180, 70, 20), -1)
    cv2.putText(canvas, "Raw GPS Positions (Meters)", (leg_x + 35, leg_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (30, 30, 30), 1)

    cv2.circle(canvas, (leg_x + 20, leg_y + 55), 4, (30, 140, 255), -1)
    cv2.putText(canvas, "B1 Georeferenced COLMAP", (leg_x + 35, leg_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (30, 30, 30), 1)

    cv2.imwrite(str(output_path), canvas)

def render_b1_gps_residuals(
    residuals_vec: np.ndarray,
    output_path: Union[str, Path],
    img_width: int = 1400,
    img_height: int = 800
) -> None:
    """
    Render per-frame GPS residual magnitudes and individual East/North/Up components.
    """
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    canvas = np.full((img_height, img_width, 3), 255, dtype=np.uint8)

    margin_left, margin_right = 100, 60
    margin_top, margin_bottom = 110, 80
    plot_w = img_width - margin_left - margin_right
    plot_h = img_height - margin_top - margin_bottom

    res_mag = np.linalg.norm(residuals_vec, axis=1)
    max_res = float(np.max(res_mag))
    y_max = max(1.5, math.ceil(max_res * 1.25 * 10.0) / 10.0)

    # Background and Grid
    cv2.rectangle(canvas, (margin_left, margin_top), (margin_left + plot_w, margin_top + plot_h), (246, 248, 250), -1)
    cv2.rectangle(canvas, (margin_left, margin_top), (margin_left + plot_w, margin_top + plot_h), (190, 198, 206), 2)

    for i in range(5):
        val = y_max * (4 - i) / 4.0
        gy = margin_top + int(plot_h * i / 4)
        cv2.line(canvas, (margin_left, gy), (margin_left + plot_w, gy), (225, 230, 236), 1)
        cv2.putText(canvas, f"{val:.2f} m", (margin_left - 80, gy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 85, 95), 1)

    n = len(res_mag)
    pts_mag = []
    for i in range(n):
        sx = int(margin_left + (i / max(1, n - 1)) * plot_w)
        sy = int(margin_top + plot_h * (1.0 - res_mag[i] / y_max))
        pts_mag.append((sx, sy))

    for i in range(len(pts_mag) - 1):
        cv2.line(canvas, pts_mag[i], pts_mag[i + 1], (180, 40, 20), 2, cv2.LINE_AA)

    mean_res = float(np.mean(res_mag))
    rmse_res = float(np.sqrt(np.mean(res_mag**2)))
    mean_y = margin_top + int(plot_h * (1.0 - mean_res / y_max))
    cv2.line(canvas, (margin_left, mean_y), (margin_left + plot_w, mean_y), (0, 0, 220), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"Mean Residual = {mean_res:.3f} m (RMSE = {rmse_res:.3f} m)", (margin_left + 15, mean_y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 220), 1)

    # Header
    cv2.putText(canvas, "STEP 9C: Per-Frame GPS Alignment Residuals (350 Frames)", (40, 45),
                cv2.FONT_HERSHEY_DUPLEX, 0.9, (25, 30, 40), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"RMSE: {rmse_res:.4f} m | Mean: {mean_res:.4f} m | Max: {max_res:.4f} m | East/North/Up Fits",
                (40, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (90, 100, 110), 1, cv2.LINE_AA)

    cv2.imwrite(str(output_path), canvas)

def render_b1_scale_comparison(
    b0_len: float,
    gps_len: float,
    b1_len: float,
    scale_s: float,
    output_path: Union[str, Path],
    img_width: int = 1200,
    img_height: int = 700
) -> None:
    """
    Render bar chart comparing trajectory path lengths and scale parameters.
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
        ("B0 COLMAP (Arbitrary)", b0_len, "units", (40, 140, 255)),
        ("Raw GPS (Metric)", gps_len, "meters", (180, 70, 20)),
        ("B1 Georeferenced", b1_len, "meters", (34, 139, 34))
    ]

    max_val = max(b0_len, gps_len, b1_len) * 1.3
    bar_w = 120
    spacing = plot_w // 3

    for i, (name, val, unit, color) in enumerate(categories):
        bx = margin_left + int((i + 0.5) * spacing) - bar_w // 2
        bh = int(plot_h * (val / max_val))
        by = margin_top + plot_h - bh - 2

        cv2.rectangle(canvas, (bx, by), (bx + bar_w, margin_top + plot_h - 2), color, -1)
        cv2.rectangle(canvas, (bx, by), (bx + bar_w, margin_top + plot_h - 2), (40, 40, 40), 1)

        cv2.putText(canvas, f"{val:.2f} {unit}", (bx + 10, by - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (20, 20, 20), 1)
        cv2.putText(canvas, name, (bx - 15, margin_top + plot_h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (30, 30, 30), 1)

    # Info Overlay
    cv2.putText(canvas, f"Estimated Scale Factor (s): {scale_s:.6f} m/unit (Inverse: {1.0/scale_s:.4f} units/m)",
                (margin_left + 20, margin_top + 35), cv2.FONT_HERSHEY_DUPLEX, 0.55, (30, 35, 45), 1)

    cv2.putText(canvas, "STEP 9C: Baseline B1 Trajectory Scale & Path Length Comparison", (40, 45),
                cv2.FONT_HERSHEY_DUPLEX, 0.9, (25, 30, 40), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Comparison of Arbitrary-Scale B0, Input GPS Path Length, and Georeferenced B1 Length",
                (40, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (90, 100, 110), 1, cv2.LINE_AA)

    cv2.imwrite(str(output_path), canvas)
