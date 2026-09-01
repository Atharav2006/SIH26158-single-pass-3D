import os
import subprocess
import json
import sqlite3
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple
from src.metrics.alignment import quaternion_to_rotation_matrix, rotation_matrix_to_quaternion

def b2_pose_to_colmap_pose(q_wc: np.ndarray, c_w: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert B2 Camera-to-World pose (q_wc, c_w) to COLMAP World-to-Camera pose (q_cw, t_cw).
    X_w = R_wc * X_c + c_w
    X_c = R_cw * X_w + t_cw
    Therefore:
      R_cw = R_wc.T
      t_cw = -R_cw * c_w
    """
    R_wc = quaternion_to_rotation_matrix(q_wc)
    R_cw = R_wc.T
    
    # Check orthogonality
    assert np.allclose(R_cw @ R_cw.T, np.eye(3), atol=1e-5), "Rotation matrix is not orthogonal"
    assert np.isclose(np.linalg.det(R_cw), 1.0, atol=1e-5), "Rotation matrix determinant is not 1"
    
    q_cw = rotation_matrix_to_quaternion(R_cw)
    t_cw = -R_cw @ c_w
    return q_cw, t_cw

def colmap_pose_to_b2_pose(q_cw: np.ndarray, t_cw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert COLMAP World-to-Camera pose (q_cw, t_cw) back to Camera-to-World (q_wc, c_w).
    c_w = -R_cw.T * t_cw
    """
    R_cw = quaternion_to_rotation_matrix(q_cw)
    R_wc = R_cw.T
    
    assert np.allclose(R_wc @ R_wc.T, np.eye(3), atol=1e-5)
    assert np.isclose(np.linalg.det(R_wc), 1.0, atol=1e-5)
    
    q_wc = rotation_matrix_to_quaternion(R_wc)
    c_w = -R_wc @ t_cw
    return q_wc, c_w

class ColmapWorkspace:
    def __init__(self, workspace_dir: Path):
        self.workspace_dir = workspace_dir
        self.images_dir = self.workspace_dir / "images"
        self.sparse_dir = self.workspace_dir / "sparse"
        self.dense_dir = self.workspace_dir / "dense"
        
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.sparse_dir.mkdir(parents=True, exist_ok=True)
        self.dense_dir.mkdir(parents=True, exist_ok=True)

    def write_cameras_txt(self, camera_model: str, width: int, height: int, params: List[float]):
        cameras_txt = self.sparse_dir / "cameras.txt"
        with open(cameras_txt, "w") as f:
            f.write("# Camera list with one line of data per camera:\n")
            f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
            f.write(f"1 {camera_model} {width} {height} {' '.join(map(str, params))}\n")

    def write_images_txt(self, b2_poses: List[Dict[str, Any]]):
        """
        b2_poses: list of dicts with 'imgid', 'filename', 'q_wc', 'c_w'
        """
        images_txt = self.sparse_dir / "images.txt"
        with open(images_txt, "w") as f:
            f.write("# Image list with two lines of data per image:\n")
            f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
            f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
            
            for pose in b2_poses:
                img_id = pose["imgid"]
                filename = pose["filename"]
                q_wc = pose["q_wc"]
                c_w = pose["c_w"]
                
                q_cw, t_cw = b2_pose_to_colmap_pose(q_wc, c_w)
                
                # COLMAP format: qw, qx, qy, qz (which is w, x, y, z)
                # But our q array is [qx, qy, qz, qw]
                colmap_q = [q_cw[3], q_cw[0], q_cw[1], q_cw[2]]
                
                line1 = f"{img_id} {colmap_q[0]:.6f} {colmap_q[1]:.6f} {colmap_q[2]:.6f} {colmap_q[3]:.6f} {t_cw[0]:.6f} {t_cw[1]:.6f} {t_cw[2]:.6f} 1 {filename}\n"
                f.write(line1)
                f.write("\n") # No points2D

    def write_points3D_txt(self):
        # Empty file for dense reconstruction
        points3d_txt = self.sparse_dir / "points3D.txt"
        with open(points3d_txt, "w") as f:
            f.write("# 3D point list with one line of data per point:\n")
            f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")

    def read_images_txt(self) -> List[Dict[str, Any]]:
        images_txt = self.sparse_dir / "images.txt"
        poses = []
        with open(images_txt, "r") as f:
            lines = f.readlines()
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line or line.startswith("#"):
                i += 1
                continue
            
            parts = line.split()
            img_id = int(parts[0])
            qw, qx, qy, qz = map(float, parts[1:5])
            tx, ty, tz = map(float, parts[5:8])
            camera_id = int(parts[8])
            filename = parts[9]
            
            q_cw = np.array([qx, qy, qz, qw])
            t_cw = np.array([tx, ty, tz])
            
            q_wc, c_w = colmap_pose_to_b2_pose(q_cw, t_cw)
            poses.append({
                "imgid": img_id,
                "filename": filename,
                "q_wc": q_wc,
                "c_w": c_w
            })
            
            # skip points2d line
            i += 2
            
        return poses

def run_colmap_image_undistorter(workspace: Path, image_dir: Path):
    cmd = [
        "colmap", "image_undistorter",
        "--image_path", str(image_dir),
        "--input_path", str(workspace / "sparse"),
        "--output_path", str(workspace / "dense")
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    return res

def run_colmap_patch_match_stereo(workspace: Path, gpu_index: str = "0", max_image_size: int = 1000):
    cmd = [
        "colmap", "patch_match_stereo",
        "--workspace_path", str(workspace / "dense"),
        "--workspace_format", "COLMAP",
        "--PatchMatchStereo.gpu_index", gpu_index,
        "--PatchMatchStereo.max_image_size", str(max_image_size),
        "--PatchMatchStereo.depth_min", "1.0",
        "--PatchMatchStereo.depth_max", "150.0",
        "--PatchMatchStereo.min_triangulation_angle", "0.1",
        "--PatchMatchStereo.filter_min_triangulation_angle", "0.1"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    return res

def run_colmap_stereo_fusion(workspace: Path, output_ply: Path):
    cmd = [
        "colmap", "stereo_fusion",
        "--workspace_path", str(workspace / "dense"),
        "--workspace_format", "COLMAP",
        "--input_type", "geometric",
        "--output_path", str(output_ply),
        "--StereoFusion.min_num_pixels", "2",
        "--StereoFusion.max_depth_error", "0.5",
        "--StereoFusion.max_reproj_error", "10",
        "--StereoFusion.max_normal_error", "30"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    return res
