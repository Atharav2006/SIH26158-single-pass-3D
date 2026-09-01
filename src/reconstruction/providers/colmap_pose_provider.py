import json
from pathlib import Path
from typing import Dict, Any, Optional

from src.reconstruction.session import ReconstructionSession
from src.reconstruction.pose_provider import PoseSource
from src.reconstruction.colmap_wrapper import COLMAPRunner
from src.reconstruction.colmap_parser import parse_colmap_images_txt, parse_colmap_points3D_txt, compute_colmap_metrics

class ColmapPoseProvider:
    """Automatic pose estimation using COLMAP."""
    
    def __init__(self, session: ReconstructionSession):
        self.session = session
        self.colmap_dir = self.session.base_dir / "colmap"
        self.database_path = self.colmap_dir / "database.db"
        self.sparse_dir = self.colmap_dir / "sparse"
        self.images_dir = self.session.frames_dir
        
        self.runner = COLMAPRunner(workspace_dir=self.colmap_dir)
        
    def estimate_poses(self, camera_model: str = "OPENCV", camera_params: Optional[str] = None) -> Dict[str, Any]:
        """Runs the COLMAP SfM pipeline and exports Camera-to-World poses."""
        self.colmap_dir.mkdir(parents=True, exist_ok=True)
        self.sparse_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Feature Extraction
        code, ext_time, _ = self.runner.extract_features(
            image_path=self.images_dir,
            database_path=self.database_path,
            camera_model=camera_model,
            camera_params=camera_params,
            single_camera=True
        )
        if code != 0:
            return {"status": "POSE_ESTIMATION_FAILED", "reason": "Feature extraction failed"}
            
        # 2. Matching
        code, match_time, _ = self.runner.match_exhaustive(database_path=self.database_path)
        if code != 0:
            return {"status": "POSE_ESTIMATION_FAILED", "reason": "Matching failed"}
            
        # 3. Mapper
        mapper_out = self.sparse_dir / "0"
        mapper_out.mkdir(parents=True, exist_ok=True)
        code, map_time, _ = self.runner.run_mapper(
            image_path=self.images_dir,
            database_path=self.database_path,
            output_path=mapper_out
        )
        if code != 0 or not (mapper_out / "cameras.bin").exists():
            return {"status": "POSE_ESTIMATION_FAILED", "reason": "INSUFFICIENT_FEATURES or INSUFFICIENT_MOTION"}
            
        # 4. Convert model to text
        text_out = self.sparse_dir / "0_text"
        self.runner.convert_model(mapper_out, text_out)
        
        # 5. Parse and Quality Validation
        images = parse_colmap_images_txt(text_out / "images.txt")
        points = parse_colmap_points3D_txt(text_out / "points3D.txt")
        total_images = len(list(self.images_dir.glob("*.jpg"))) + len(list(self.images_dir.glob("*.png")))
        
        metrics = compute_colmap_metrics(total_images, images, points)
        
        # Validation rules
        if metrics["registration_rate"] < 0.2:
            return {"status": "POSE_QUALITY_LOW", "reason": "DISCONNECTED_RECONSTRUCTION", "metrics": metrics}
            
        # 6. Save poses to a standard CSV trajectory format matching the engine's expectation
        poses_csv = self.session.poses_dir / "colmap_fused_trajectory.csv"
        with open(poses_csv, 'w') as f:
            f.write("imgid,filename,x_w,y_w,z_w,qx_w,qy_w,qz_w,qw_w\n")
            # Sort by image ID
            for cam_id in sorted(images.keys()):
                img = images[cam_id]
                f.write(f"{img['imgid']},{img['filename']},{img['x_world']},{img['y_world']},{img['z_world']},"
                        f"{img['qx_world']},{img['qy_world']},{img['qz_world']},{img['qw_world']}\n")
        
        return {
            "status": "POSE_ESTIMATION_READY",
            "source": PoseSource.COLMAP_SfM.value,
            "metrics": metrics,
            "poses_path": str(poses_csv),
            "performance": {
                "extraction_time": ext_time,
                "match_time": match_time,
                "map_time": map_time,
                "total_time": ext_time + match_time + map_time
            }
        }
