import math
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Union, Tuple

def render_b0_gt_vs_colmap_topdown(
    gt_pts: np.ndarray,
    raw_colmap_pts: np.ndarray,
    aligned_colmap_pts: np.ndarray,
    keyframe_indices: List[int],
    output_path: Union[str, Path],
    img_width: int = 1400,
    img_height: int = 1000
) -> None:
    """
    Render 2D top-down comparison between Ground Truth, Raw COLMAP, and Sim(3)-Aligned COLMAP trajectories.
    """
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    canvas = np.full((img_height, img_width, 3), 255, dtype=np.uint8)

    # Focus bounding box on Ground Truth and Aligned COLMAP (metric coordinates)
    combined = np.vstack([gt_pts[:, :2], aligned_colmap_pts[:, :2]])
    min_x, min_y = np.min(combined, axis=0)
    max_x, max_y = np.max(combined, axis=0)

    span_x = max(1.0, max_x - min_x)
    span_y = max(1.0, max_y - min_y)

    margin_left = 120
    margin_right = 80
    margin_top = 110
    margin_bottom = 100

    plot_w = img_width - margin_left - margin_right
    plot_h = img_height - margin_top - margin_bottom

    scale = min(plot_w / span_x, plot_h / span_y) * 0.85
    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0
    canvas_cx = margin_left + plot_w / 2.0
    canvas_cy = margin_top + plot_h / 2.0

    def world_to_screen(wx: float, wy: float) -> Tuple[int, int]:
        sx = int(canvas_cx + (wx - center_x) * scale)
        sy = int(canvas_cy - (wy - center_y) * scale)
        return sx, sy

    # 1. Background Grid & Framing
    cv2.rectangle(canvas, (margin_left, margin_top), (margin_left + plot_w, margin_top + plot_h), (244, 246, 248), -1)
    cv2.rectangle(canvas, (margin_left, margin_top), (margin_left + plot_w, margin_top + plot_h), (190, 198, 206), 2)

    # Metric Grid lines
    for i in range(1, 6):
        gx = margin_left + int(plot_w * i / 6)
        gy = margin_top + int(plot_h * i / 6)
        cv2.line(canvas, (gx, margin_top), (gx, margin_top + plot_h), (228, 232, 238), 1)
        cv2.line(canvas, (margin_left, gy), (margin_left + plot_w, gy), (228, 232, 238), 1)

    # 2. Draw Ground Truth Flight Path (Dark Blue Polyline + Green/Red Endpoints)
    gt_screen = [world_to_screen(p[0], p[1]) for p in gt_pts]
    for i in range(len(gt_screen) - 1):
        cv2.line(canvas, gt_screen[i], gt_screen[i + 1], (180, 50, 20), 3, cv2.LINE_AA)

    for idx, (sx, sy) in enumerate(gt_screen):
        cv2.circle(canvas, (sx, sy), 5, (180, 50, 20), -1, cv2.LINE_AA)
        cv2.circle(canvas, (sx, sy), 6, (0, 0, 0), 1, cv2.LINE_AA)

    # 3. Draw Sim(3)-Aligned COLMAP Flight Path (Bright Magenta/Orange + Markers)
    colmap_screen = [world_to_screen(p[0], p[1]) for p in aligned_colmap_pts]
    for i in range(len(colmap_screen) - 1):
        cv2.line(canvas, colmap_screen[i], colmap_screen[i + 1], (30, 140, 255), 2, cv2.LINE_AA)

    # 4. Draw Error Vectors between Evaluation Keyframes
    for i in range(len(gt_pts)):
        gx, gy = gt_screen[i]
        cx, cy = colmap_screen[i]
        cv2.line(canvas, (gx, gy), (cx, cy), (0, 0, 220), 2, cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy), 5, (30, 140, 255), -1, cv2.LINE_AA)

    # Start & End markers
    cv2.circle(canvas, gt_screen[0], 9, (34, 139, 34), -1, cv2.LINE_AA)
    cv2.putText(canvas, "START (KF 1)", (gt_screen[0][0] + 12, gt_screen[0][1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (34, 139, 34), 2)

    cv2.circle(canvas, gt_screen[-1], 9, (0, 0, 200), -1, cv2.LINE_AA)
    cv2.putText(canvas, f"END (KF {keyframe_indices[-1]})", (gt_screen[-1][0] + 12, gt_screen[-1][1] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 2)

    # 5. Header and Legend
    cv2.putText(canvas, "COLMAP B0 Baseline vs Photogrammetric Ground Truth (Top-Down)", (40, 45),
                cv2.FONT_HERSHEY_DUPLEX, 0.9, (25, 30, 40), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"Evaluated Keyframes: {len(gt_pts)} exact imgid pairs | Alignment: Closed-form Umeyama Sim(3)",
                (40, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (90, 100, 110), 1, cv2.LINE_AA)

    # Legend Card
    leg_x = margin_left + plot_w - 280
    leg_y = margin_top + 15
    cv2.rectangle(canvas, (leg_x, leg_y), (leg_x + 265, leg_y + 105), (255, 255, 255), -1)
    cv2.rectangle(canvas, (leg_x, leg_y), (leg_x + 265, leg_y + 105), (190, 198, 206), 1)

    cv2.line(canvas, (leg_x + 15, leg_y + 25), (leg_x + 45, leg_y + 25), (180, 50, 20), 3)
    cv2.circle(canvas, (leg_x + 30, leg_y + 25), 4, (180, 50, 20), -1)
    cv2.putText(canvas, "Ground Truth (1 Hz Keyframes)", (leg_x + 55, leg_y + 29), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (30, 30, 30), 1)

    cv2.line(canvas, (leg_x + 15, leg_y + 55), (leg_x + 45, leg_y + 55), (30, 140, 255), 2)
    cv2.circle(canvas, (leg_x + 30, leg_y + 55), 4, (30, 140, 255), -1)
    cv2.putText(canvas, "COLMAP B0 (Sim(3) Aligned)", (leg_x + 55, leg_y + 59), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (30, 30, 30), 1)

    cv2.line(canvas, (leg_x + 15, leg_y + 85), (leg_x + 45, leg_y + 85), (0, 0, 220), 2)
    cv2.putText(canvas, "Displacement Residual Vector", (leg_x + 55, leg_y + 89), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 200), 1)

    cv2.imwrite(str(output_path), canvas)

def render_b0_position_error_plot(
    keyframe_indices: List[int],
    per_frame_errors_m: List[float],
    output_path: Union[str, Path],
    img_width: int = 1200,
    img_height: int = 700
) -> None:
    """
    Render bar and line plot of per-keyframe Absolute Trajectory Error (ATE) in meters.
    """
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    canvas = np.full((img_height, img_width, 3), 255, dtype=np.uint8)

    margin_left = 100
    margin_right = 60
    margin_top = 110
    margin_bottom = 100

    plot_w = img_width - margin_left - margin_right
    plot_h = img_height - margin_top - margin_bottom

    max_err = max(per_frame_errors_m) if per_frame_errors_m else 1.0
    y_max = math.ceil(max_err * 1.25 * 10.0) / 10.0
    if y_max < 0.5:
        y_max = 0.5

    # 1. Background and Axis
    cv2.rectangle(canvas, (margin_left, margin_top), (margin_left + plot_w, margin_top + plot_h), (246, 248, 250), -1)
    cv2.rectangle(canvas, (margin_left, margin_top), (margin_left + plot_w, margin_top + plot_h), (190, 198, 206), 2)

    # Horizontal Grid Lines & Y-ticks
    for i in range(5):
        val = y_max * (4 - i) / 4.0
        gy = margin_top + int(plot_h * i / 4)
        cv2.line(canvas, (margin_left, gy), (margin_left + plot_w, gy), (225, 230, 236), 1)
        cv2.putText(canvas, f"{val:.2f} m", (margin_left - 80, gy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 85, 95), 1)

    # 2. Draw Bars and Points
    n = len(keyframe_indices)
    bar_slot_w = plot_w / float(n)
    bar_w = int(bar_slot_w * 0.55)

    pts_line = []
    for i, (k_idx, err) in enumerate(zip(keyframe_indices, per_frame_errors_m)):
        cx = int(margin_left + (i + 0.5) * bar_slot_w)
        cy = margin_top + int(plot_h * (1.0 - err / y_max))

        # Bar
        x1 = cx - bar_w // 2
        x2 = cx + bar_w // 2
        cv2.rectangle(canvas, (x1, cy), (x2, margin_top + plot_h - 2), (240, 150, 40), -1)
        cv2.rectangle(canvas, (x1, cy), (x2, margin_top + plot_h - 2), (200, 100, 20), 1)

        # Point & Label
        cv2.circle(canvas, (cx, cy), 4, (180, 40, 20), -1, cv2.LINE_AA)
        cv2.putText(canvas, f"{err:.2f}m", (cx - 18, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (30, 30, 30), 1)

        # X-tick label
        cv2.putText(canvas, f"imgid {k_idx}", (cx - 24, margin_top + plot_h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (40, 40, 40), 1)
        pts_line.append((cx, cy))

    # Connect points
    for i in range(len(pts_line) - 1):
        cv2.line(canvas, pts_line[i], pts_line[i + 1], (180, 40, 20), 2, cv2.LINE_AA)

    # Mean error line
    mean_err = float(np.mean(per_frame_errors_m))
    mean_y = margin_top + int(plot_h * (1.0 - mean_err / y_max))
    cv2.line(canvas, (margin_left, mean_y), (margin_left + plot_w, mean_y), (0, 0, 220), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"Mean ATE = {mean_err:.3f} m", (margin_left + 15, mean_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 220), 1)

    # Header
    cv2.putText(canvas, "COLMAP B0 Baseline - Absolute Trajectory Positional Error (ATE)", (40, 45),
                cv2.FONT_HERSHEY_DUPLEX, 0.9, (25, 30, 40), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"Mean ATE: {mean_err:.4f} m | RMSE: {float(np.sqrt(np.mean(np.array(per_frame_errors_m)**2))):.4f} m | Max: {max_err:.4f} m",
                (40, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (90, 100, 110), 1, cv2.LINE_AA)

    cv2.imwrite(str(output_path), canvas)

def render_b0_trajectory_comparison_3d(
    gt_pts: np.ndarray,
    raw_colmap_pts: np.ndarray,
    aligned_colmap_pts: np.ndarray,
    output_path: Union[str, Path],
    img_width: int = 1400,
    img_height: int = 1000
) -> None:
    """
    Render 3D isometric view comparing Ground Truth vs Raw COLMAP vs Sim(3)-Aligned COLMAP.
    """
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    canvas = np.full((img_height, img_width, 3), (20, 24, 28), dtype=np.uint8)

    # Center around GT
    combined = np.vstack([gt_pts, aligned_colmap_pts])
    cx, cy, cz = np.mean(combined, axis=0)

    # Isometric angles
    az = math.radians(45)
    el = math.radians(25)
    cos_az, sin_az = math.cos(az), math.sin(az)
    cos_el, sin_el = math.cos(el), math.sin(el)

    def project_3d(x: float, y: float, z: float) -> Tuple[float, float, float]:
        dx, dy, dz = x - cx, y - cy, z - cz
        rx = dx * cos_az - dy * sin_az
        ry = dx * sin_az + dy * cos_az
        px = rx
        py = ry * cos_el - dz * sin_el
        depth = ry * sin_el + dz * cos_el
        return px, py, depth

    gt_proj = [project_3d(p[0], p[1], p[2]) for p in gt_pts]
    colmap_proj = [project_3d(p[0], p[1], p[2]) for p in aligned_colmap_pts]

    all_px = [p[0] for p in gt_proj + colmap_proj]
    all_py = [p[1] for p in gt_proj + colmap_proj]

    span_x = max(1.0, max(all_px) - min(all_px))
    span_y = max(1.0, max(all_py) - min(all_py))

    scale = min((img_width - 240) / span_x, (img_height - 240) / span_y) * 0.88
    canvas_cx = img_width / 2.0
    canvas_cy = img_height / 2.0 + 20

    def screen_coord(px: float, py: float) -> Tuple[int, int]:
        return int(canvas_cx + px * scale), int(canvas_cy - py * scale)

    # Draw Ground Truth Path in 3D (Teal/Cyan)
    gt_scr = [screen_coord(p[0], p[1]) for p in gt_proj]
    for i in range(len(gt_scr) - 1):
        cv2.line(canvas, gt_scr[i], gt_scr[i + 1], (220, 180, 40), 3, cv2.LINE_AA)
    for sx, sy in gt_scr:
        cv2.circle(canvas, (sx, sy), 5, (220, 180, 40), -1, cv2.LINE_AA)

    # Draw Aligned COLMAP Path in 3D (Orange)
    colmap_scr = [screen_coord(p[0], p[1]) for p in colmap_proj]
    for i in range(len(colmap_scr) - 1):
        cv2.line(canvas, colmap_scr[i], colmap_scr[i + 1], (40, 140, 255), 2, cv2.LINE_AA)
    for sx, sy in colmap_scr:
        cv2.circle(canvas, (sx, sy), 4, (40, 140, 255), -1, cv2.LINE_AA)

    # Draw 3D error vectors
    for i in range(len(gt_scr)):
        cv2.line(canvas, gt_scr[i], colmap_scr[i], (80, 80, 240), 1, cv2.LINE_AA)

    # Header and Legend
    cv2.putText(canvas, "COLMAP B0 Baseline - 3D Trajectory Reconstruction Comparison", (40, 45),
                cv2.FONT_HERSHEY_DUPLEX, 0.9, (240, 245, 250), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Isometric 3D Projection | Ground Truth (Cyan) vs Sim(3) Aligned COLMAP (Orange)",
                (40, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 170, 180), 1, cv2.LINE_AA)

    cv2.imwrite(str(output_path), canvas)
