from dataclasses import dataclass
from typing import Optional
from pathlib import Path

@dataclass
class VideoInputSpec:
    """Generalized input contract for a new reconstruction session."""
    video_path: Path
    
    # Optional sensor and metadata inputs
    gps_path: Optional[Path] = None
    imu_path: Optional[Path] = None
    calibration_path: Optional[Path] = None
    poses_path: Optional[Path] = None
    rtk_path: Optional[Path] = None
    
    def __post_init__(self):
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {self.video_path}")
