import math
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Union, Tuple

def render_gps_vs_colmap_raw(
    gps_local_pts: np.ndarray,
    colmap_pts: np.ndarray,
    output_path: Union[str, Path],
    img_width: int = 1500,
    img_height: int = 800
) -> None:
    """
    Render side-by-side dual-panel comparison of Raw GPS Trajectory (meters) vs Raw COLMAP Trajectory (arbitrary units).
    """
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    canvas = np.full((img_height, img_width, 3), 255, dtype=np.uint8)

    # Split canvas into two equal panels
    panel_w = (img_width - 120) // 2
    panel_h = img_height - 180

    p1_x1, p1_y1 = 50, 110
    p2_x1, p2_y1 = 50 + panel_w + 20, 110

    # 1. Backgrounds & Borders
    cv2.rectangle(canvas, (p1_x1, p1_y1), (p1_x1 + panel_w, p1_y1 + panel_h), (244, 246, 248), -1)
    cv2.rectangle(canvas, (p1_x1, p1_y1), (p1_x1 + panel_w, p1_y1 + panel_h), (190, 198, 206), 2)

    cv2.rectangle(canvas, (p2_x1, p2_y1), (p2_x1 + panel_w, p2_y1 + panel_h), (244, 246, 248), -1)
    cv2.rectangle(canvas, (p2_x1, p2_y1), (p2_x1 + panel_w, p2_y1 + panel_h), (190, 198, 206), 2)

    # --- PANEL 1: RAW GPS (Local ENU Meters) ---
    gx, gy = gps_local_pts[:, 0], gps_local_pts[:, 1]
    g_span_x = max(1e-3, np.ptp(gx))
    g_span_y = max(1e-3, np.ptp(gy))
    g_scale = min((panel_w - 80) / g_span_x, (panel_h - 80) / g_span_y) * 0.85
    g_cx, g_cy = np.mean(gx), np.mean(gy)
    p1_cx = p1_x1 + panel_w / 2.0
    p1_cy = p1_y1 + panel_h / 2.0

    def g_to_screen(x: float, y: float) -> Tuple[int, int]:
        return int(p1_cx + (x - g_cx) * g_scale), int(p1_cy - (y - g_cy) * g_scale)

    # Grid for Panel 1
    for i in range(1, 5):
        lx = p1_x1 + int(panel_w * i / 5)
        ly = p1_y1 + int(panel_h * i / 5)
        cv2.line(canvas, (lx, p1_y1), (lx, p1_y1 + panel_h), (225, 230, 236), 1)
        cv2.line(canvas, (p1_x1, ly), (p1_x1 + panel_w, ly), (225, 230, 236), 1)

    g_screen = [g_to_screen(p[0], p[1]) for p in gps_local_pts]
    for i in range(len(g_screen) - 1):
        cv2.line(canvas, g_screen[i], g_screen[i + 1], (180, 70, 20), 2, cv2.LINE_AA)
    for sx, sy in g_screen:
        cv2.circle(canvas, (sx, sy), 2, (180, 70, 20), -1)

    cv2.circle(canvas, g_screen[0], 7, (34, 139, 34), -1, cv2.LINE_AA)
    cv2.circle(canvas, g_screen[-1], 7, (0, 0, 220), -1, cv2.LINE_AA)

    # Panel 1 Titles
    cv2.putText(canvas, "1. Raw GPS Stream (UTM Zone 32N / Local ENU)", (p1_x1 + 15, p1_y1 + 30),
                cv2.FONT_HERSHEY_DUPLEX, 0.65, (30, 35, 45), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"350 Associated Frames | Units: Meters (m)", (p1_x1 + 15, p1_y1 + 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (100, 105, 115), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"Span X: {g_span_x:.2f} m | Span Y: {g_span_y:.2f} m", (p1_x1 + 15, p1_y1 + p_h - 20 if (p_h := panel_h) else 0),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 85, 95), 1, cv2.LINE_AA)

    # --- PANEL 2: RAW COLMAP B0 (Reconstructed C_w Units) ---
    cx, cy = colmap_pts[:, 0], colmap_pts[:, 1]
    c_span_x = max(1e-3, np.ptp(cx))
    c_span_y = max(1e-3, np.ptp(cy))
    c_scale = min((panel_w - 80) / c_span_x, (panel_h - 80) / c_span_y) * 0.85
    c_cx, c_cy = np.mean(cx), np.mean(cy)
    p2_cx = p2_x1 + panel_w / 2.0
    p2_cy = p2_y1 + panel_h / 2.0

    def c_to_screen(x: float, y: float) -> Tuple[int, int]:
        return int(p2_cx + (x - c_cx) * c_scale), int(p2_cy - (y - c_cy) * c_scale)

    # Grid for Panel 2
    for i in range(1, 5):
        lx = p2_x1 + int(panel_w * i / 5)
        ly = p2_y1 + int(panel_h * i / 5)
        cv2.line(canvas, (lx, p2_y1), (lx, p2_y1 + panel_h), (225, 230, 236), 1)
        cv2.line(canvas, (p2_x1, ly), (p2_x1 + panel_w, ly), (225, 230, 236), 1)

    c_screen = [c_to_screen(p[0], p[1]) for p in colmap_pts]
    for i in range(len(c_screen) - 1):
        cv2.line(canvas, c_screen[i], c_screen[i + 1], (30, 140, 255), 2, cv2.LINE_AA)
    for sx, sy in c_screen:
        cv2.circle(canvas, (sx, sy), 2, (30, 140, 255), -1)

    cv2.circle(canvas, c_screen[0], 7, (34, 139, 34), -1, cv2.LINE_AA)
    cv2.circle(canvas, c_screen[-1], 7, (0, 0, 220), -1, cv2.LINE_AA)

    # Panel 2 Titles
    cv2.putText(canvas, "2. Raw COLMAP B0 Reconstruction (C_w)", (p2_x1 + 15, p2_y1 + 30),
                cv2.FONT_HERSHEY_DUPLEX, 0.65, (30, 35, 45), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"350 Registered Cameras | Units: Arbitrary Scale", (p2_x1 + 15, p2_y1 + 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (100, 105, 115), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"Span X: {c_span_x:.2f} units | Span Y: {c_span_y:.2f} units", (p2_x1 + 15, p2_y1 + panel_h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 85, 95), 1, cv2.LINE_AA)

    # Header
    cv2.putText(canvas, "STEP 9A: Raw GPS Stream vs Raw COLMAP B0 Camera Trajectory", (40, 45),
                cv2.FONT_HERSHEY_DUPLEX, 0.9, (25, 30, 40), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Pre-Alignment Comparison (Strict Coordinate Separation - No Ground Truth Used)",
                (40, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (90, 100, 110), 1, cv2.LINE_AA)

    cv2.imwrite(str(output_path), canvas)

def render_gps_trajectory_local(
    gps_local_pts: np.ndarray,
    timestamps: np.ndarray,
    output_path: Union[str, Path],
    img_width: int = 1400,
    img_height: int = 900
) -> None:
    """
    Render 2D top-down local flight path with an altitude profile subpanel.
    """
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    canvas = np.full((img_height, img_width, 3), 255, dtype=np.uint8)

    # Top Panel: 2D East-North Trajectory
    top_x, top_y = 60, 110
    top_w, top_h = img_width - 120, 480

    cv2.rectangle(canvas, (top_x, top_y), (top_x + top_w, top_y + top_h), (244, 246, 248), -1)
    cv2.rectangle(canvas, (top_x, top_y), (top_x + top_w, top_y + top_h), (190, 198, 206), 2)

    gx, gy, gz = gps_local_pts[:, 0], gps_local_pts[:, 1], gps_local_pts[:, 2]
    span_x = max(1e-3, np.ptp(gx))
    span_y = max(1e-3, np.ptp(gy))
    scale = min((top_w - 100) / span_x, (top_h - 100) / span_y) * 0.85
    cx, cy = np.mean(gx), np.mean(gy)
    top_cx = top_x + top_w / 2.0
    top_cy = top_y + top_h / 2.0

    def to_screen(x: float, y: float) -> Tuple[int, int]:
        return int(top_cx + (x - cx) * scale), int(top_cy - (y - cy) * scale)

    # Grid
    for i in range(1, 6):
        lx = top_x + int(top_w * i / 6)
        ly = top_y + int(top_h * i / 6)
        cv2.line(canvas, (lx, top_y), (lx, top_y + top_h), (225, 230, 236), 1)
        cv2.line(canvas, (top_x, ly), (top_x + top_w, ly), (225, 230, 236), 1)

    pts_scr = [to_screen(p[0], p[1]) for p in gps_local_pts]
    for i in range(len(pts_scr) - 1):
        cv2.line(canvas, pts_scr[i], pts_scr[i + 1], (180, 50, 20), 2, cv2.LINE_AA)

    cv2.circle(canvas, pts_scr[0], 8, (34, 139, 34), -1, cv2.LINE_AA)
    cv2.putText(canvas, "START (Frame 1)", (pts_scr[0][0] + 12, pts_scr[0][1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (34, 139, 34), 2)

    cv2.circle(canvas, pts_scr[-1], 8, (0, 0, 220), -1, cv2.LINE_AA)
    cv2.putText(canvas, "END (Frame 350)", (pts_scr[-1][0] + 12, pts_scr[-1][1] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 220), 2)

    cv2.putText(canvas, "Top-Down Flight Path (Local East-North Frame, Meters)", (top_x + 15, top_y + 30),
                cv2.FONT_HERSHEY_DUPLEX, 0.65, (30, 35, 45), 1, cv2.LINE_AA)

    # Bottom Panel: Altitude vs Time Profile
    bot_x, bot_y = 60, 620
    bot_w, bot_h = img_width - 120, 230

    cv2.rectangle(canvas, (bot_x, bot_y), (bot_x + bot_w, bot_y + bot_h), (244, 246, 248), -1)
    cv2.rectangle(canvas, (bot_x, bot_y), (bot_x + bot_w, bot_y + bot_h), (190, 198, 206), 2)

    t_span = max(1e-3, np.ptp(timestamps))
    z_span = max(1e-3, np.ptp(gz))
    z_min = np.min(gz)

    def alt_to_screen(t: float, z: float) -> Tuple[int, int]:
        sx = int(bot_x + 50 + (t - timestamps[0]) / t_span * (bot_w - 100))
        sy = int(bot_y + bot_h - 40 - (z - z_min) / max(1e-3, z_span) * (bot_h - 80))
        return sx, sy

    alt_scr = [alt_to_screen(t, z) for t, z in zip(timestamps, gz)]
    for i in range(len(alt_scr) - 1):
        cv2.line(canvas, alt_scr[i], alt_scr[i + 1], (40, 140, 255), 2, cv2.LINE_AA)

    cv2.putText(canvas, "GPS Altitude Profile (Meters relative to Origin vs Time in Seconds)", (bot_x + 15, bot_y + 25),
                cv2.FONT_HERSHEY_DUPLEX, 0.55, (30, 35, 45), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"Min Alt: {np.min(gz):.2f} m | Max Alt: {np.max(gz):.2f} m | Delta Alt: {z_span:.2f} m",
                (bot_x + 15, bot_y + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 105, 115), 1, cv2.LINE_AA)

    # Main Header
    cv2.putText(canvas, "STEP 9A: Zurich Urban MAV Local GPS Trajectory & Altitude Profile", (40, 45),
                cv2.FONT_HERSHEY_DUPLEX, 0.9, (25, 30, 40), 2, cv2.LINE_AA)
    cv2.putText(canvas, "WGS84 -> UTM Zone 32N / Local ENU Frame (Origin: Frame 1 GPS)",
                (40, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (90, 100, 110), 1, cv2.LINE_AA)

    cv2.imwrite(str(output_path), canvas)
