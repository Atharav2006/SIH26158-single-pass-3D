from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any

class PoseSource(str, Enum):
    PRECOMPUTED_B2 = "PRECOMPUTED_B2"
    COLMAP_SfM = "COLMAP_SfM"
    PROVIDED_TRAJECTORY = "PROVIDED_TRAJECTORY"
    NOT_AVAILABLE = "NOT_AVAILABLE"

class PoseProvider:
    @staticmethod
    def identify(poses_path: Optional[Path]) -> PoseSource:
        if poses_path is None:
            return PoseSource.NOT_AVAILABLE
        name = poses_path.name.lower()
        if "b2" in name or "zurich" in name:
            return PoseSource.PRECOMPUTED_B2
        elif "colmap" in name:
            return PoseSource.COLMAP_SfM
        else:
            return PoseSource.PROVIDED_TRAJECTORY
