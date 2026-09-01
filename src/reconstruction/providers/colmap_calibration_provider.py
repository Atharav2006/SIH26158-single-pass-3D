import json
from pathlib import Path
from typing import Dict, Any, Optional

from src.reconstruction.session import ReconstructionSession
from src.reconstruction.calibration_provider import CalibrationSource
from src.reconstruction.colmap_parser import parse_colmap_cameras_txt

class ColmapCalibrationProvider:
    """Provides camera calibration either from supplied input or COLMAP estimation."""
    
    def __init__(self, session: ReconstructionSession):
        self.session = session
        
    def estimate_calibration(self) -> Dict[str, Any]:
        """
        Parses the estimated calibration from the COLMAP sparse reconstruction.
        Should be called AFTER colmap_pose_provider completes if auto-estimating.
        """
        text_out = self.session.base_dir / "colmap" / "sparse" / "0_text"
        cameras_file = text_out / "cameras.txt"
        
        if not cameras_file.exists():
            return {"status": "CALIBRATION_ESTIMATION_FAILED", "reason": "No COLMAP reconstruction found"}
            
        cameras = parse_colmap_cameras_txt(cameras_file)
        if not cameras:
            return {"status": "CALIBRATION_ESTIMATION_FAILED", "reason": "No cameras found in reconstruction"}
            
        # Assume single camera model for the sequence
        cam_data = list(cameras.values())[0]
        params = cam_data["params"]
        model = cam_data["model"]
        
        # Parse based on model. OPENCV has fx, fy, cx, cy, k1, k2, p1, p2
        # PINHOLE has fx, fy, cx, cy
        if model == "PINHOLE":
            fx, fy, cx, cy = params[0], params[1], params[2], params[3]
        elif model in ["OPENCV", "OPENCV_FISHEYE"]:
            fx, fy, cx, cy = params[0], params[1], params[2], params[3]
        elif model == "SIMPLE_RADIAL":
            fx = fy = params[0]
            cx, cy = params[1], params[2]
        else:
            # Fallback
            fx = fy = params[0]
            cx, cy = params[1], params[2] if len(params) > 2 else params[0]
            
        # Quality check plausibility
        width, height = cam_data["width"], cam_data["height"]
        uncertain = False
        
        # Plausibility: cx, cy should be roughly near center
        if not (0.2 * width < cx < 0.8 * width) or not (0.2 * height < cy < 0.8 * height):
            uncertain = True
            
        # Plausibility: focal length shouldn't be extreme
        if fx < 0.1 * width or fx > 10 * width:
            uncertain = True
            
        calib_dict = {
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "width": width,
            "height": height,
            "model": model,
            "params": params
        }
        
        calib_path = self.session.calibration_dir / "estimated_calibration.json"
        with open(calib_path, 'w') as f:
            json.dump(calib_dict, f, indent=4)
            
        return {
            "status": "CALIBRATION_UNCERTAIN" if uncertain else "CALIBRATION_READY",
            "source": CalibrationSource.COLMAP_ESTIMATED.value,
            "calibration_path": str(calib_path),
            "data": calib_dict
        }
