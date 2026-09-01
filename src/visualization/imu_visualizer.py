import math
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Union, Tuple

def render_imu_acceleration_plot(
    timestamps: np.ndarray,
    ax: np.ndarray,
    ay: np.ndarray,
    az: np.ndarray,
    amag: np.ndarray,
    output_path: Union[str, Path],
    img_width: int = 1400,
    img_height: int = 800
) -> None:
    """
    Render 3-axis linear acceleration and magnitude vs timestamp for the 350-image flight window.
    """
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    canvas = np.full((img_height, img_width, 3), 255, dtype=np.uint8)

    margin_left, margin_right = 90, 60
    margin_top, margin_bottom = 110, 80
    plot_w = img_width - margin_left - margin_right
    plot_h = img_height - margin_top - margin_bottom

    t_min, t_max = float(np.min(timestamps)), float(np.max(timestamps))
    t_span = max(1e-3, t_max - t_min)

    y_min, y_max = -14.0, 14.0
    y_span = y_max - y_min

    def to_screen(t_val: float, y_val: float) -> Tuple[int, int]:
        sx = int(margin_left + ((float(t_val) - t_min) / t_span) * plot_w)
        sy = int(margin_top + (1.0 - (float(y_val) - y_min) / y_span) * plot_h)
        return sx, sy

    # Background and Grid
    cv2.rectangle(canvas, (margin_left, margin_top), (margin_left + plot_w, margin_top + plot_h), (246, 248, 250), -1)
    cv2.rectangle(canvas, (margin_left, margin_top), (margin_left + plot_w, margin_top + plot_h), (190, 198, 206), 2)

    for y_grid in range(-12, 13, 4):
        _, gy = to_screen(t_min, y_grid)
        cv2.line(canvas, (margin_left, gy), (margin_left + plot_w, gy), (225, 230, 236), 1)
        cv2.putText(canvas, f"{y_grid:+.1f}", (margin_left - 65, gy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80, 85, 95), 1)

    # Zero line
    _, zero_y = to_screen(t_min, 0.0)
    cv2.line(canvas, (margin_left, zero_y), (margin_left + plot_w, zero_y), (160, 165, 175), 1, cv2.LINE_AA)

    # Plot lines: ax (Red), ay (Green), az (Blue), amag (Purple)
    curves = [
        (ax, (0, 0, 220), 1),
        (ay, (34, 139, 34), 1),
        (az, (220, 100, 20), 2),
        (amag, (180, 40, 180), 2)
    ]

    for data, color, thick in curves:
        pts = [to_screen(t, v) for t, v in zip(timestamps, data)]
        for i in range(len(pts) - 1):
            cv2.line(canvas, (int(pts[i][0]), int(pts[i][1])), (int(pts[i+1][0]), int(pts[i+1][1])), color, thick, cv2.LINE_AA)

    # Header and Legend
    cv2.putText(canvas, "STEP 11A: Zurich Urban MAV IMU Linear Acceleration (m/s^2)", (40, 45),
                cv2.FONT_HERSHEY_DUPLEX, 0.9, (25, 30, 40), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"Time Range: [{t_min:.3f}s, {t_max:.3f}s] | Mean |a|: {np.mean(amag):.2f} m/s^2 (Gravity inclusive)",
                (40, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (90, 100, 110), 1, cv2.LINE_AA)

    # Legend
    leg_x = margin_left + plot_w - 380
    leg_y = margin_top + 15
    cv2.rectangle(canvas, (leg_x, leg_y), (leg_x + 365, leg_y + 40), (255, 255, 255), -1)
    cv2.rectangle(canvas, (leg_x, leg_y), (leg_x + 365, leg_y + 40), (190, 198, 206), 1)

    items = [("a_x", (0, 0, 220)), ("a_y", (34, 139, 34)), ("a_z", (220, 100, 20)), ("|a|", (180, 40, 180))]
    for i, (label, col) in enumerate(items):
        lx = leg_x + 15 + i * 85
        cv2.line(canvas, (lx, leg_y + 20), (lx + 20, leg_y + 20), col, 2)
        cv2.putText(canvas, label, (lx + 25, leg_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (30, 30, 30), 1)

    cv2.imwrite(str(output_path), canvas)

def render_imu_angular_velocity_plot(
    timestamps: np.ndarray,
    gx: np.ndarray,
    gy: np.ndarray,
    gz: np.ndarray,
    output_path: Union[str, Path],
    img_width: int = 1400,
    img_height: int = 800
) -> None:
    """
    Render 3-axis angular velocity vs timestamp for the 350-image flight window.
    """
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    canvas = np.full((img_height, img_width, 3), 255, dtype=np.uint8)

    margin_left, margin_right = 90, 60
    margin_top, margin_bottom = 110, 80
    plot_w = img_width - margin_left - margin_right
    plot_h = img_height - margin_top - margin_bottom

    t_min, t_max = float(np.min(timestamps)), float(np.max(timestamps))
    t_span = max(1e-3, t_max - t_min)

    # In degrees per second
    gx_deg = np.degrees(gx)
    gy_deg = np.degrees(gy)
    gz_deg = np.degrees(gz)

    y_max = max(20.0, float(np.max(np.abs(np.hstack([gx_deg, gy_deg, gz_deg])))) * 1.2)
    y_min = -y_max
    y_span = y_max - y_min

    def to_screen(t_val: float, y_val: float) -> Tuple[int, int]:
        sx = int(margin_left + ((float(t_val) - t_min) / t_span) * plot_w)
        sy = int(margin_top + (1.0 - (float(y_val) - y_min) / y_span) * plot_h)
        return sx, sy

    # Background and Grid
    cv2.rectangle(canvas, (margin_left, margin_top), (margin_left + plot_w, margin_top + plot_h), (246, 248, 250), -1)
    cv2.rectangle(canvas, (margin_left, margin_top), (margin_left + plot_w, margin_top + plot_h), (190, 198, 206), 2)

    for i in range(5):
        val = y_max - i * (2 * y_max / 4)
        _, gy_scr = to_screen(t_min, val)
        cv2.line(canvas, (margin_left, gy_scr), (margin_left + plot_w, gy_scr), (225, 230, 236), 1)
        cv2.putText(canvas, f"{val:+.1f} deg/s", (margin_left - 80, gy_scr + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (80, 85, 95), 1)

    curves = [
        (gx_deg, (0, 0, 220), "Roll Rate (gx)"),
        (gy_deg, (34, 139, 34), "Pitch Rate (gy)"),
        (gz_deg, (220, 100, 20), "Yaw Rate (gz)")
    ]

    for data, color, _ in curves:
        pts = [to_screen(t, v) for t, v in zip(timestamps, data)]
        for i in range(len(pts) - 1):
            cv2.line(canvas, (int(pts[i][0]), int(pts[i][1])), (int(pts[i+1][0]), int(pts[i+1][1])), color, 2, cv2.LINE_AA)

    # Header
    cv2.putText(canvas, "STEP 11A: Zurich Urban MAV IMU Gyroscope Angular Velocity (deg/s)", (40, 45),
                cv2.FONT_HERSHEY_DUPLEX, 0.9, (25, 30, 40), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"3-Axis Angular Rates | Time Range: [{t_min:.3f}s, {t_max:.3f}s]",
                (40, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (90, 100, 110), 1, cv2.LINE_AA)

    # Legend
    leg_x = margin_left + plot_w - 360
    leg_y = margin_top + 15
    cv2.rectangle(canvas, (leg_x, leg_y), (leg_x + 345, leg_y + 40), (255, 255, 255), -1)
    cv2.rectangle(canvas, (leg_x, leg_y), (leg_x + 345, leg_y + 40), (190, 198, 206), 1)

    for i, (data, col, label) in enumerate(curves):
        lx = leg_x + 15 + i * 110
        cv2.line(canvas, (lx, leg_y + 20), (lx + 20, leg_y + 20), col, 2)
        cv2.putText(canvas, label.split()[0], (lx + 25, leg_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (30, 30, 30), 1)

    cv2.imwrite(str(output_path), canvas)

def render_imu_sampling_interval_plot(
    dt_ms: np.ndarray,
    output_path: Union[str, Path],
    img_width: int = 1400,
    img_height: int = 800
) -> None:
    """
    Render sampling interval histogram and distribution.
    """
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    canvas = np.full((img_height, img_width, 3), 255, dtype=np.uint8)

    margin_left, margin_right = 90, 60
    margin_top, margin_bottom = 110, 80
    plot_w = img_width - margin_left - margin_right
    plot_h = img_height - margin_top - margin_bottom

    # Histogram between 80 ms and 120 ms
    bins = np.linspace(80.0, 120.0, 41)
    counts, bin_edges = np.histogram(dt_ms, bins=bins)
    max_count = max(1, int(np.max(counts)))

    cv2.rectangle(canvas, (margin_left, margin_top), (margin_left + plot_w, margin_top + plot_h), (246, 248, 250), -1)
    cv2.rectangle(canvas, (margin_left, margin_top), (margin_left + plot_w, margin_top + plot_h), (190, 198, 206), 2)

    bar_w = plot_w / len(counts)

    for i in range(len(counts)):
        bx = int(margin_left + i * bar_w)
        bh = int(plot_h * (counts[i] / (max_count * 1.15)))
        by = margin_top + plot_h - bh - 2
        cv2.rectangle(canvas, (bx, by), (int(bx + bar_w - 1), margin_top + plot_h - 2), (220, 120, 30), -1)

    # Grid and Labels
    cv2.putText(canvas, f"Max Bin Count: {max_count}", (margin_left + 15, margin_top + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (80, 85, 95), 1)

    for b_val in [80, 90, 100, 110, 120]:
        sx = int(margin_left + ((b_val - 80.0) / 40.0) * plot_w)
        cv2.line(canvas, (sx, margin_top), (sx, margin_top + plot_h), (210, 215, 225), 1)
        cv2.putText(canvas, f"{b_val} ms", (sx - 18, margin_top + plot_h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (60, 60, 60), 1)

    # Header
    cv2.putText(canvas, "STEP 11A: Zurich Urban MAV IMU Sampling Interval Distribution (ms)", (40, 45),
                cv2.FONT_HERSHEY_DUPLEX, 0.9, (25, 30, 40), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"Nominal Rate: 10.0 Hz (100.0 ms) | Mean: {np.mean(dt_ms):.2f} ms | Std: {np.std(dt_ms):.2f} ms | Samples: {len(dt_ms)}",
                (40, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (90, 100, 110), 1, cv2.LINE_AA)

    cv2.imwrite(str(output_path), canvas)
