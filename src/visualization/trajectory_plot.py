import math
import numpy as np
import cv2
from pathlib import Path
from typing import Union, List, Tuple

from src.pose.trajectory import Trajectory

def plot_topdown_trajectory(
    trajectory: Trajectory,
    output_path: Union[str, Path],
    img_width: int = 1200,
    img_height: int = 1000
) -> None:
    """
    Render a 2D top-down trajectory plot (East on X-axis, North on Y-axis).
    """
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not trajectory.poses:
        img = np.full((img_height, img_width, 3), 255, dtype=np.uint8)
        cv2.putText(img, "Empty Trajectory", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imwrite(str(output_path), img)
        return

    # Canvas setup (clean white background)
    canvas = np.full((img_height, img_width, 3), 255, dtype=np.uint8)

    # Extract X (East) and Y (North)
    xs = np.array([p.position.x for p in trajectory.poses])
    ys = np.array([p.position.y for p in trajectory.poses])

    min_x, max_x = xs.min(), xs.max()
    min_y, max_y = ys.min(), ys.max()

    span_x = max(1.0, max_x - min_x)
    span_y = max(1.0, max_y - min_y)

    margin_left = 120
    margin_right = 60
    margin_top = 100
    margin_bottom = 100

    plot_w = img_width - margin_left - margin_right
    plot_h = img_height - margin_top - margin_bottom

    # Equal aspect ratio scaling
    scale = min(plot_w / span_x, plot_h / span_y) * 0.9
    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0
    canvas_cx = margin_left + plot_w / 2.0
    canvas_cy = margin_top + plot_h / 2.0

    def world_to_screen(wx: float, wy: float) -> Tuple[int, int]:
        sx = int(canvas_cx + (wx - center_x) * scale)
        sy = int(canvas_cy - (wy - center_y) * scale)  # Y inverted on screen
        return sx, sy

    # 1. Draw Grid Lines and Labels
    grid_color = (235, 235, 235)
    border_color = (180, 180, 180)
    text_color = (60, 60, 60)

    # Outer bounding box
    cv2.rectangle(canvas, (margin_left, margin_top), (margin_left + plot_w, margin_top + plot_h), border_color, 1)

    # Determine nice grid step
    raw_step = max(span_x, span_y) / 6.0
    grid_step = 10.0 ** math.floor(math.log10(raw_step))
    if raw_step / grid_step > 5:
        grid_step *= 5
    elif raw_step / grid_step > 2:
        grid_step *= 2

    # Draw vertical grid (East)
    x_start = math.floor(min_x / grid_step) * grid_step
    x_curr = x_start
    while x_curr <= max_x + grid_step:
        gx, _ = world_to_screen(x_curr, center_y)
        if margin_left <= gx <= margin_left + plot_w:
            cv2.line(canvas, (gx, margin_top), (gx, margin_top + plot_h), grid_color, 1)
            lbl = f"{x_curr:.0f}m"
            cv2.putText(canvas, lbl, (gx - 20, margin_top + plot_h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1, cv2.LINE_AA)
        x_curr += grid_step

    # Draw horizontal grid (North)
    y_start = math.floor(min_y / grid_step) * grid_step
    y_curr = y_start
    while y_curr <= max_y + grid_step:
        _, gy = world_to_screen(center_x, y_curr)
        if margin_top <= gy <= margin_top + plot_h:
            cv2.line(canvas, (margin_left, gy), (margin_left + plot_w, gy), grid_color, 1)
            lbl = f"{y_curr:.0f}m"
            cv2.putText(canvas, lbl, (margin_left - 75, gy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1, cv2.LINE_AA)
        y_curr += grid_step

    # 2. Draw Trajectory Polyline
    points = [world_to_screen(x, y) for x, y in zip(xs, ys)]
    for i in range(1, len(points)):
        cv2.line(canvas, points[i - 1], points[i], (210, 100, 30), 2, cv2.LINE_AA)

    # 3. Draw Start and End Markers
    start_pt = points[0]
    end_pt = points[-1]

    # Start: Green circle
    cv2.circle(canvas, start_pt, 7, (40, 180, 40), -1, cv2.LINE_AA)
    cv2.circle(canvas, start_pt, 9, (20, 100, 20), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Start", (start_pt[0] + 12, start_pt[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 140, 20), 2, cv2.LINE_AA)

    # End: Red circle
    cv2.circle(canvas, end_pt, 7, (30, 30, 220), -1, cv2.LINE_AA)
    cv2.circle(canvas, end_pt, 9, (10, 10, 140), 1, cv2.LINE_AA)
    cv2.putText(canvas, "End", (end_pt[0] + 12, end_pt[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 180), 2, cv2.LINE_AA)

    # 4. Title and Information Overlay
    cv2.putText(canvas, "Zurich Urban MAV - 2D Top-Down Trajectory", (margin_left, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (30, 30, 30), 2, cv2.LINE_AA)
    subtitle = f"Frame: {trajectory.frame_id} | Total Length: {trajectory.compute_statistics()['trajectory_length_meters']:.1f} m | Poses: {len(trajectory.poses)}"
    cv2.putText(canvas, subtitle, (margin_left, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1, cv2.LINE_AA)

    # Axis Titles
    cv2.putText(canvas, "East [meters]", (margin_left + plot_w // 2 - 40, img_height - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 2, cv2.LINE_AA)
    
    # Legend box
    leg_x, leg_y = margin_left + plot_w - 200, margin_top + 20
    cv2.rectangle(canvas, (leg_x, leg_y), (leg_x + 190, leg_y + 90), (245, 245, 245), -1)
    cv2.rectangle(canvas, (leg_x, leg_y), (leg_x + 190, leg_y + 90), (200, 200, 200), 1)
    
    cv2.line(canvas, (leg_x + 15, leg_y + 25), (leg_x + 45, leg_y + 25), (210, 100, 30), 2)
    cv2.putText(canvas, "UAV Flight Path", (leg_x + 55, leg_y + 29), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (40, 40, 40), 1, cv2.LINE_AA)
    
    cv2.circle(canvas, (leg_x + 30, leg_y + 50), 5, (40, 180, 40), -1)
    cv2.putText(canvas, "Start Point", (leg_x + 55, leg_y + 54), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (40, 40, 40), 1, cv2.LINE_AA)

    cv2.circle(canvas, (leg_x + 30, leg_y + 75), 5, (30, 30, 220), -1)
    cv2.putText(canvas, "End Point", (leg_x + 55, leg_y + 79), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (40, 40, 40), 1, cv2.LINE_AA)

    cv2.imwrite(str(output_path), canvas)


def plot_3d_trajectory(
    trajectory: Trajectory,
    output_path: Union[str, Path],
    img_width: int = 1200,
    img_height: int = 1000
) -> None:
    """
    Render an isometric 3D trajectory plot.
    """
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not trajectory.poses:
        img = np.full((img_height, img_width, 3), 255, dtype=np.uint8)
        cv2.putText(img, "Empty Trajectory", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imwrite(str(output_path), img)
        return

    canvas = np.full((img_height, img_width, 3), 255, dtype=np.uint8)

    xs = np.array([p.position.x for p in trajectory.poses])
    ys = np.array([p.position.y for p in trajectory.poses])
    zs = np.array([p.position.z for p in trajectory.poses])

    cx = (xs.min() + xs.max()) / 2.0
    cy = (ys.min() + ys.max()) / 2.0
    cz = (zs.min() + zs.max()) / 2.0

    # Center coordinates
    x_c = xs - cx
    y_c = ys - cy
    z_c = zs - cz

    # Isometric projection angles
    azimuth = math.radians(-35)
    elevation = math.radians(25)

    cos_az, sin_az = math.cos(azimuth), math.sin(azimuth)
    cos_el, sin_el = math.cos(elevation), math.sin(elevation)

    # 3D projection matrix: R_z(azimuth) followed by R_x(elevation)
    proj_x = x_c * cos_az - y_c * sin_az
    proj_y = (x_c * sin_az + y_c * cos_az) * sin_el + z_c * cos_el

    span_px = max(1.0, proj_x.max() - proj_x.min())
    span_py = max(1.0, proj_y.max() - proj_y.min())

    plot_w = img_width - 200
    plot_h = img_height - 200
    scale = min(plot_w / span_px, plot_h / span_py) * 0.85

    canvas_cx = img_width / 2.0
    canvas_cy = img_height / 2.0 + 30

    def proj_to_screen(px: float, py: float) -> Tuple[int, int]:
        sx = int(canvas_cx + px * scale)
        sy = int(canvas_cy - py * scale)
        return sx, sy

    # Draw Ground Plane Shadow
    shadow_proj_x = x_c * cos_az - y_c * sin_az
    shadow_proj_y = (x_c * sin_az + y_c * cos_az) * sin_el + (zs.min() - cz) * cos_el
    shadow_pts = [proj_to_screen(px, py) for px, py in zip(shadow_proj_x, shadow_proj_y)]
    for i in range(1, len(shadow_pts)):
        cv2.line(canvas, shadow_pts[i - 1], shadow_pts[i], (230, 230, 230), 1, cv2.LINE_AA)

    # Draw 3D Trajectory with Altitude Color Gradient
    traj_pts = [proj_to_screen(px, py) for px, py in zip(proj_x, proj_y)]
    z_norm = (zs - zs.min()) / max(1e-3, (zs.max() - zs.min()))

    for i in range(1, len(traj_pts)):
        # Color transition from Dark Blue (low) to Cyan/Orange (high)
        val = z_norm[i]
        b = int(220 * (1 - val) + 30 * val)
        g = int(80 * (1 - val) + 160 * val)
        r = int(30 * (1 - val) + 230 * val)
        cv2.line(canvas, traj_pts[i - 1], traj_pts[i], (b, g, r), 2, cv2.LINE_AA)

    # Start and End points
    start_pt = traj_pts[0]
    end_pt = traj_pts[-1]

    cv2.circle(canvas, start_pt, 7, (40, 180, 40), -1, cv2.LINE_AA)
    cv2.putText(canvas, "Start", (start_pt[0] + 10, start_pt[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 140, 20), 2, cv2.LINE_AA)

    cv2.circle(canvas, end_pt, 7, (30, 30, 220), -1, cv2.LINE_AA)
    cv2.putText(canvas, "End", (end_pt[0] + 10, end_pt[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 180), 2, cv2.LINE_AA)

    # Title and overlay
    cv2.putText(canvas, "Zurich Urban MAV - 3D Trajectory Visualization", (60, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (30, 30, 30), 2, cv2.LINE_AA)
    subtitle = f"Frame: {trajectory.frame_id} | Length: {trajectory.compute_statistics()['trajectory_length_meters']:.1f} m | Altitude Range: {zs.min():.1f}m - {zs.max():.1f}m"
    cv2.putText(canvas, subtitle, (60, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1, cv2.LINE_AA)

    # Coordinate Axis Indicator (Bottom Left)
    origin_screen = (120, img_height - 100)
    axis_len = 50
    # X (East)
    ax_x = (int(origin_screen[0] + axis_len * cos_az), int(origin_screen[1] - axis_len * sin_az * sin_el))
    cv2.arrowedLine(canvas, origin_screen, ax_x, (0, 0, 200), 2, cv2.LINE_AA, tipLength=0.25)
    cv2.putText(canvas, "+X (East)", (ax_x[0] + 5, ax_x[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 180), 1, cv2.LINE_AA)

    # Y (North)
    ax_y = (int(origin_screen[0] - axis_len * sin_az), int(origin_screen[1] - axis_len * cos_az * sin_el))
    cv2.arrowedLine(canvas, origin_screen, ax_y, (0, 180, 0), 2, cv2.LINE_AA, tipLength=0.25)
    cv2.putText(canvas, "+Y (North)", (ax_y[0] + 5, ax_y[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 140, 0), 1, cv2.LINE_AA)

    # Z (Up)
    ax_z = (origin_screen[0], origin_screen[1] - int(axis_len * cos_el))
    cv2.arrowedLine(canvas, origin_screen, ax_z, (200, 100, 0), 2, cv2.LINE_AA, tipLength=0.25)
    cv2.putText(canvas, "+Z (Up)", (ax_z[0] + 5, ax_z[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 80, 0), 1, cv2.LINE_AA)

    cv2.imwrite(str(output_path), canvas)
