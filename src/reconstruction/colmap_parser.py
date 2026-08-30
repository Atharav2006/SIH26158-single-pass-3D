import csv
import json
import statistics
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

from src.reconstruction.colmap_wrapper import invert_colmap_pose

def parse_colmap_cameras_txt(cameras_file: Path) -> Dict[int, Dict[str, Any]]:
    """Parse COLMAP cameras.txt file."""
    cameras = {}
    if not cameras_file.exists():
        return cameras

    with open(cameras_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            cam_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = [float(p) for p in parts[4:]]
            cameras[cam_id] = {
                "camera_id": cam_id,
                "model": model,
                "width": width,
                "height": height,
                "params": params
            }
    return cameras

def parse_colmap_images_txt(images_file: Path) -> Dict[int, Dict[str, Any]]:
    """
    Parse COLMAP images.txt file.
    Every image has 2 lines:
      Line 1: IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
      Line 2: POINTS2D[] as (X, Y, POINT3D_ID) triples
    """
    images = {}
    if not images_file.exists():
        return images

    with open(images_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    idx = 0
    while idx < len(lines):
        line1 = lines[idx]
        parts = line1.split()
        if len(parts) >= 10:
            colmap_image_id = int(parts[0])
            qw = float(parts[1])
            qx = float(parts[2])
            qy = float(parts[3])
            qz = float(parts[4])
            tx = float(parts[5])
            ty = float(parts[6])
            tz = float(parts[7])
            camera_id = int(parts[8])
            name = parts[9]

            # Invert COLMAP pose (T_cw -> T_wc)
            (cw_x, cw_y, cw_z), (qwc_x, qwc_y, qwc_z, qwc_w) = invert_colmap_pose(qw, qx, qy, qz, tx, ty, tz)

            # Read line 2 (2D point observations)
            num_points2d = 0
            num_points3d = 0
            if idx + 1 < len(lines):
                p2d_parts = lines[idx + 1].split()
                num_points2d = len(p2d_parts) // 3
                # Count points with valid 3D point ID != -1
                for p_idx in range(2, len(p2d_parts), 3):
                    if p2d_parts[p_idx] != "-1":
                        num_points3d += 1

            # Extract numeric imgid from name (e.g. '00001.jpg' -> 1)
            import re
            m = re.search(r'(\d+)$', Path(name).stem)
            native_imgid = int(m.group(1)) if m else colmap_image_id

            images[colmap_image_id] = {
                "colmap_image_id": colmap_image_id,
                "imgid": native_imgid,
                "filename": name,
                "camera_id": camera_id,
                "q_colmap_wxyz": (qw, qx, qy, qz),
                "t_colmap_xyz": (tx, ty, tz),
                "x_world": cw_x,
                "y_world": cw_y,
                "z_world": cw_z,
                "qx_world": qwc_x,
                "qy_world": qwc_y,
                "qz_world": qwc_z,
                "qw_world": qwc_w,
                "num_points2D": num_points2d,
                "num_points3D": num_points3d
            }
            idx += 2
        else:
            idx += 1

    return images

def parse_colmap_points3D_txt(points_file: Path) -> Dict[int, Dict[str, Any]]:
    """
    Parse COLMAP points3D.txt file.
    Format: POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX) pairs
    """
    points = {}
    if not points_file.exists():
        return points

    with open(points_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 8:
                point_id = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                z = float(parts[3])
                r = int(parts[4])
                g = int(parts[5])
                b = int(parts[6])
                error = float(parts[7])
                track_elements = parts[8:]
                track_length = len(track_elements) // 2

                points[point_id] = {
                    "point3D_id": point_id,
                    "x": x,
                    "y": y,
                    "z": z,
                    "r": r,
                    "g": g,
                    "b": b,
                    "error": error,
                    "track_length": track_length
                }
    return points

def compute_colmap_metrics(
    total_images_count: int,
    registered_images: Dict[int, Dict[str, Any]],
    sparse_points: Dict[int, Dict[str, Any]]
) -> Dict[str, Any]:
    """Compute standard photogrammetry baseline metrics."""
    registered_count = len(registered_images)
    reg_rate = round(registered_count / total_images_count, 4) if total_images_count > 0 else 0.0

    errors = [p["error"] for p in sparse_points.values()]
    tracks = [p["track_length"] for p in sparse_points.values()]

    mean_err = round(statistics.mean(errors), 4) if errors else 0.0
    median_err = round(statistics.median(errors), 4) if errors else 0.0
    max_err = round(max(errors), 4) if errors else 0.0

    mean_track = round(statistics.mean(tracks), 2) if tracks else 0.0
    total_observations = sum(tracks)

    return {
        "total_images": total_images_count,
        "registered_images": registered_count,
        "unregistered_images": total_images_count - registered_count,
        "registration_rate": reg_rate,
        "registration_percentage": round(reg_rate * 100, 2),
        "sparse_3d_point_count": len(sparse_points),
        "total_observations": total_observations,
        "mean_track_length": mean_track,
        "reprojection_error": {
            "mean_px": mean_err,
            "median_px": median_err,
            "max_px": max_err
        }
    }
