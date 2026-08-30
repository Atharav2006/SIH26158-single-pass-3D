import os
import shutil
import subprocess
import time
import re
import math
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

DEFAULT_COLMAP_SEARCH_PATHS = [
    Path(r"D:\SIH26158\tools\colmap\colmap.exe"),
    Path(r"D:\SIH26158\tools\colmap\bin\colmap.exe"),
]

def find_colmap_executable() -> Path:
    """Locate the colmap.exe binary on the system."""
    # 1. Check known explicit paths
    for p in DEFAULT_COLMAP_SEARCH_PATHS:
        if p.is_file():
            return p

    # 2. Check PATH
    which_colmap = shutil.which("colmap")
    if which_colmap:
        return Path(which_colmap).resolve()

    which_colmap_exe = shutil.which("colmap.exe")
    if which_colmap_exe:
        return Path(which_colmap_exe).resolve()

    raise FileNotFoundError("COLMAP executable not found in known locations or PATH.")

def colmap_quat_to_rotmat(qw: float, qx: float, qy: float, qz: float) -> List[List[float]]:
    """Convert COLMAP quaternion (qw, qx, qy, qz) to 3x3 rotation matrix R_cw."""
    norm = math.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
    if norm < 1e-10:
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    qw, qx, qy, qz = qw/norm, qx/norm, qy/norm, qz/norm

    return [
        [1.0 - 2.0*(qy*qy + qz*qz), 2.0*(qx*qy - qz*qw),       2.0*(qx*qz + qy*qw)],
        [2.0*(qx*qy + qz*qw),       1.0 - 2.0*(qx*qx + qz*qz), 2.0*(qy*qz - qx*qw)],
        [2.0*(qx*qz - qy*qw),       2.0*(qy*qz + qx*qw),       1.0 - 2.0*(qx*qx + qy*qy)]
    ]

def invert_colmap_pose(qw: float, qx: float, qy: float, qz: float, tx: float, ty: float, tz: float) -> Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]:
    """
    Convert COLMAP World-to-Camera (T_cw) pose to Camera-in-World (T_wc) position and attitude.
    
    COLMAP convention:
      X_c = R_cw * X_w + t_cw
      
    World coordinates:
      C_w = - R_cw^T * t_cw
      R_wc = R_cw^T
      q_wc (Hamilton scalar-last [qx, qy, qz, qw]) = [-qx, -qy, -qz, qw]
    """
    R_cw = colmap_quat_to_rotmat(qw, qx, qy, qz)
    
    # R_wc = R_cw^T
    # C_w = - R_wc * t_cw
    cx = -(R_cw[0][0]*tx + R_cw[1][0]*ty + R_cw[2][0]*tz)
    cy = -(R_cw[0][1]*tx + R_cw[1][1]*ty + R_cw[2][1]*tz)
    cz = -(R_cw[0][2]*tx + R_cw[1][2]*ty + R_cw[2][2]*tz)

    norm = math.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
    if norm > 1e-10:
        qw_n, qx_n, qy_n, qz_n = qw/norm, qx/norm, qy/norm, qz/norm
        # Invert unit quaternion: q^-1 = [-qx, -qy, -qz, qw]
        q_wc = (-qx_n, -qy_n, -qz_n, qw_n)
    else:
        q_wc = (0.0, 0.0, 0.0, 1.0)

    return (cx, cy, cz), q_wc

class COLMAPRunner:
    """Wrapper for executing COLMAP CLI commands with detailed logging and metrics."""
    def __init__(self, colmap_bin: Optional[Union[str, Path]] = None, workspace_dir: Optional[Union[str, Path]] = None):
        self.colmap_bin = Path(colmap_bin).resolve() if colmap_bin else find_colmap_executable()
        self.workspace_dir = Path(workspace_dir).resolve() if workspace_dir else Path(r"D:\SIH26158\colmap_workspace\zurich_mav_b0").resolve()
        
        # Ensure workspace subdirectories exist
        (self.workspace_dir / "sparse").mkdir(parents=True, exist_ok=True)
        (self.workspace_dir / "logs").mkdir(parents=True, exist_ok=True)
        (self.workspace_dir / "metadata").mkdir(parents=True, exist_ok=True)

    def _run_command(self, cmd_args: List[str], log_path: Path) -> Tuple[int, float, str]:
        """Execute a COLMAP command and record runtime and stdout/stderr to a log file."""
        log_path.parent.mkdir(parents=True, exist_ok=True)
        full_cmd = [str(self.colmap_bin)] + cmd_args
        
        start_time = time.perf_counter()
        with open(log_path, "w", encoding="utf-8") as f_log:
            f_log.write(f"COMMAND: {' '.join(full_cmd)}\n\n")
            f_log.flush()
            proc = subprocess.run(
                full_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace"
            )
            f_log.write(proc.stdout)

        elapsed = time.perf_counter() - start_time
        return proc.returncode, elapsed, proc.stdout

    def extract_features(
        self,
        image_path: Union[str, Path],
        database_path: Union[str, Path],
        camera_model: str = "OPENCV",
        camera_params: Optional[str] = None,
        single_camera: bool = True,
        max_num_features: int = 8192,
        use_gpu: bool = True
    ) -> Tuple[int, float, Path]:
        """Run COLMAP feature_extractor."""
        log_path = self.workspace_dir / "logs" / "feature_extractor.log"
        cmd = [
            "feature_extractor",
            "--database_path", str(database_path),
            "--image_path", str(image_path),
            "--ImageReader.camera_model", str(camera_model),
            "--ImageReader.single_camera", "1" if single_camera else "0",
            "--FeatureExtraction.use_gpu", "1" if use_gpu else "0",
            "--SiftExtraction.max_num_features", str(max_num_features)
        ]
        if camera_params:
            cmd.extend(["--ImageReader.camera_params", str(camera_params)])

        code, elapsed, _ = self._run_command(cmd, log_path)
        return code, elapsed, log_path

    def match_exhaustive(
        self,
        database_path: Union[str, Path],
        use_gpu: bool = True
    ) -> Tuple[int, float, Path]:
        """Run COLMAP exhaustive_matcher."""
        log_path = self.workspace_dir / "logs" / "exhaustive_matcher.log"
        cmd = [
            "exhaustive_matcher",
            "--database_path", str(database_path),
            "--FeatureMatching.use_gpu", "1" if use_gpu else "0"
        ]
        code, elapsed, _ = self._run_command(cmd, log_path)
        return code, elapsed, log_path

    def run_mapper(
        self,
        image_path: Union[str, Path],
        database_path: Union[str, Path],
        output_path: Union[str, Path],
        min_num_matches: int = 15
    ) -> Tuple[int, float, Path]:
        """Run COLMAP incremental mapper."""
        log_path = self.workspace_dir / "logs" / "mapper.log"
        cmd = [
            "mapper",
            "--database_path", str(database_path),
            "--image_path", str(image_path),
            "--output_path", str(output_path),
            "--Mapper.min_num_matches", str(min_num_matches)
        ]
        code, elapsed, _ = self._run_command(cmd, log_path)
        return code, elapsed, log_path

    def convert_model(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        output_type: str = "TXT"
    ) -> Tuple[int, float, Path]:
        """Convert binary model files (.bin) to readable text format (.txt)."""
        log_path = self.workspace_dir / "logs" / "model_converter.log"
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        cmd = [
            "model_converter",
            "--input_path", str(input_path),
            "--output_path", str(output_path),
            "--output_type", str(output_type)
        ]
        code, elapsed, _ = self._run_command(cmd, log_path)
        return code, elapsed, log_path
