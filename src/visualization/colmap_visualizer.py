import math
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Union, Tuple

def render_colmap_trajectory_plot(
    registered_images: Dict[int, Dict[str, Any]],
    output_path: Union[str, Path],
    img_width: int = 1200,
    img_height: int = 1000
) -> None:
    """
    Render 2D top-down / isometric trajectory plot of COLMAP registered camera poses.
    """
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    canvas = np.full((img_height, img_width, 3), 255, dtype=np.uint8)

    if not registered_images:
        cv2.putText(canvas, "No Registered Cameras in COLMAP Model", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.imwrite(str(output_path), canvas)
        return

    # Sort cameras by imgid or image_id
    sorted_cams = sorted(registered_images.values(), key=lambda c: c["imgid"])
    xs = np.array([c["x_world"] for c in sorted_cams])
    ys = np.array([c["y_world"] for c in sorted_cams])
    zs = np.array([c["z_world"] for c in sorted_cams])

    min_x, max_x = xs.min(), xs.max()
    min_y, max_y = ys.min(), ys.max()

    span_x = max(1e-3, max_x - min_x)
    span_y = max(1e-3, max_y - min_y)

    margin_left = 120
    margin_right = 60
    margin_top = 100
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

    # 1. Grid
    cv2.rectangle(canvas, (margin_left, margin_top), (margin_left + plot_w, margin_top + plot_h), (230, 230, 230), -1)
    cv2.rectangle(canvas, (margin_left, margin_top), (margin_left + plot_w, margin_top + plot_h), (180, 180, 180), 2)

    # 2. Draw flight path polyline
    pts_screen = [world_to_screen(c["x_world"], c["y_world"]) for c in sorted_cams]
    for i in range(len(pts_screen) - 1):
        # Color gradient by progress (Blue to Orange)
        t = i / max(1, len(pts_screen) - 1)
        b = int(220 * (1 - t) + 40 * t)
        g = int(120 * (1 - t) + 140 * t)
        r = int(40 * (1 - t) + 240 * t)
        cv2.line(canvas, pts_screen[i], pts_screen[i + 1], (b, g, r), 3, cv2.LINE_AA)

    # 3. Draw camera frustum / position markers
    for idx, (sx, sy) in enumerate(pts_screen):
        if idx == 0:
            # Start marker (Green)
            cv2.circle(canvas, (sx, sy), 10, (34, 139, 34), -1, cv2.LINE_AA)
            cv2.circle(canvas, (sx, sy), 12, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(canvas, "START (Frame 1)", (sx + 15, sy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (34, 139, 34), 2)
        elif idx == len(pts_screen) - 1:
            # End marker (Red)
            cv2.circle(canvas, (sx, sy), 10, (0, 0, 220), -1, cv2.LINE_AA)
            cv2.circle(canvas, (sx, sy), 12, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(canvas, f"END (Frame {sorted_cams[-1]['imgid']})", (sx + 15, sy + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 220), 2)
        else:
            cv2.circle(canvas, (sx, sy), 4, (180, 80, 20), -1, cv2.LINE_AA)

    # 4. Header and Telemetry Overlay
    cv2.putText(canvas, "COLMAP Baseline (b0) - Registered Camera Trajectory", (40, 50),
                cv2.FONT_HERSHEY_DUPLEX, 0.9, (30, 30, 30), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"Registered Images: {len(sorted_cams)} | Coordinate Frame: COLMAP World (Optical Center C_w)",
                (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 100, 100), 1, cv2.LINE_AA)

    info_box_x = margin_left + 15
    info_box_y = margin_top + 20
    cv2.rectangle(canvas, (info_box_x, info_box_y), (info_box_x + 280, info_box_y + 90), (255, 255, 255), -1)
    cv2.rectangle(canvas, (info_box_x, info_box_y), (info_box_x + 280, info_box_y + 90), (180, 180, 180), 1)

    cv2.putText(canvas, f"Camera Count: {len(sorted_cams)}", (info_box_x + 10, info_box_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1)
    cv2.putText(canvas, f"X Span: {span_x:.2f} (arb. scale)", (info_box_x + 10, info_box_y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1)
    cv2.putText(canvas, f"Y Span: {span_y:.2f} (arb. scale)", (info_box_x + 10, info_box_y + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1)

    cv2.imwrite(str(output_path), canvas)

def render_colmap_sparse_pointcloud_plot(
    sparse_points: Dict[int, Dict[str, Any]],
    registered_images: Dict[int, Dict[str, Any]],
    output_path: Union[str, Path],
    img_width: int = 1200,
    img_height: int = 1000
) -> None:
    """
    Render 3D isometric / orthographic projection of the COLMAP sparse point cloud and camera stations.
    """
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    canvas = np.full((img_height, img_width, 3), 255, dtype=np.uint8)

    if not sparse_points:
        cv2.putText(canvas, "No Sparse 3D Points in COLMAP Model", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.imwrite(str(output_path), canvas)
        return

    # Extract points and filter outliers (quantile 1% - 99%)
    pts = list(sparse_points.values())
    xs = np.array([p["x"] for p in pts])
    ys = np.array([p["y"] for p in pts])
    zs = np.array([p["z"] for p in pts])

    q_low, q_high = 0.02, 0.98
    x_min_q, x_max_q = np.quantile(xs, q_low), np.quantile(xs, q_high)
    y_min_q, y_max_q = np.quantile(ys, q_low), np.quantile(ys, q_high)
    z_min_q, z_max_q = np.quantile(zs, q_low), np.quantile(zs, q_high)

    valid_mask = (xs >= x_min_q) & (xs <= x_max_q) & (ys >= y_min_q) & (ys <= y_max_q) & (zs >= z_min_q) & (zs <= z_max_q)
    filtered_pts = [p for i, p in enumerate(pts) if valid_mask[i]]

    f_xs = np.array([p["x"] for p in filtered_pts])
    f_ys = np.array([p["y"] for p in filtered_pts])
    f_zs = np.array([p["z"] for p in filtered_pts])

    min_x, max_x = f_xs.min(), f_xs.max()
    min_y, max_y = f_ys.min(), f_ys.max()
    min_z, max_z = f_zs.min(), f_zs.max()

    span_x = max(1e-3, max_x - min_x)
    span_y = max(1e-3, max_y - min_y)

    margin = 100
    plot_w = img_width - 2 * margin
    plot_h = img_height - 2 * margin

    # Isometric projection angles (Azimuth 45 deg, Elevation 30 deg)
    azimuth = math.radians(40)
    elevation = math.radians(25)

    cos_az, sin_az = math.cos(azimuth), math.sin(azimuth)
    cos_el, sin_el = math.cos(elevation), math.sin(elevation)

    # Center of pointcloud
    cx = (min_x + max_x) / 2.0
    cy = (min_y + max_y) / 2.0
    cz = (min_z + max_z) / 2.0

    # Project 3D to 2D
    def project_3d(x: float, y: float, z: float) -> Tuple[float, float, float]:
        dx = x - cx
        dy = y - cy
        dz = z - cz
        # Rotate around Z (azimuth)
        rx = dx * cos_az - dy * sin_az
        ry = dx * sin_az + dy * cos_az
        # Rotate around X (elevation)
        px = rx
        py = ry * cos_el - dz * sin_el
        depth = ry * sin_el + dz * cos_el
        return px, py, depth

    proj_pts = [project_3d(p["x"], p["y"], p["z"]) for p in filtered_pts]
    pxs = [p[0] for p in proj_pts]
    pys = [p[1] for p in proj_pts]

    p_span_x = max(1e-3, max(pxs) - min(pxs))
    p_span_y = max(1e-3, max(pys) - min(pys))

    scale = min(plot_w / p_span_x, plot_h / p_span_y) * 0.85
    canvas_cx = img_width / 2.0
    canvas_cy = img_height / 2.0

    def screen_coord(px: float, py: float) -> Tuple[int, int]:
        sx = int(canvas_cx + px * scale)
        sy = int(canvas_cy - py * scale)
        return sx, sy

    # Dark background for pointcloud contrast
    canvas[:, :] = (20, 24, 28)

    # Render points sorted by depth
    depth_order = sorted(range(len(filtered_pts)), key=lambda i: proj_pts[i][2])
    for i in depth_order:
        p = filtered_pts[i]
        px, py, _ = proj_pts[i]
        sx, sy = screen_coord(px, py)
        if 0 <= sx < img_width and 0 <= sy < img_height:
            # Point color from image RGB if valid, else height gradient
            r, g, b = p["r"], p["g"], p["b"]
            if r == 0 and g == 0 and b == 0:
                z_norm = (p["z"] - min_z) / max(1e-3, max_z - min_z)
                b = int(255 * (1 - z_norm))
                g = int(200 * z_norm)
                r = int(255 * z_norm)
            cv2.circle(canvas, (sx, sy), 1, (b, g, r), -1)

    # Render registered camera frustums / stations
    for cam in registered_images.values():
        c_px, c_py, _ = project_3d(cam["x_world"], cam["y_world"], cam["z_world"])
        c_sx, c_sy = screen_coord(c_px, c_py)
        if 0 <= c_sx < img_width and 0 <= c_sy < img_height:
            cv2.circle(canvas, (c_sx, c_sy), 3, (0, 165, 255), -1, cv2.LINE_AA)

    # Header and Telemetry Overlay
    cv2.putText(canvas, "COLMAP Baseline (b0) - Sparse 3D Point Cloud", (40, 50),
                cv2.FONT_HERSHEY_DUPLEX, 0.9, (240, 240, 240), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"Points: {len(sparse_points):,} | Cameras: {len(registered_images)} | Projection: Isometric",
                (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1, cv2.LINE_AA)

    cv2.imwrite(str(output_path), canvas)
