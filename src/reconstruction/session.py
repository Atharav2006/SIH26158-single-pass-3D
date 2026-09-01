from pathlib import Path
from typing import Optional

class ReconstructionSession:
    """
    General session abstraction to ensure completely independent and isolated
    reconstruction contexts for different datasets/videos.
    """
    def __init__(self, session_id: str, base_workspace_dir: str):
        self.session_id = session_id
        self.base_dir = Path(base_workspace_dir) / self.session_id
        
        # Isolated subdirectories for session data
        self.inputs_dir = self.base_dir / "inputs"
        self.frames_dir = self.base_dir / "frames"
        self.poses_dir = self.base_dir / "poses"
        self.depth_dir = self.base_dir / "depth"
        self.geometry_dir = self.base_dir / "geometry"
        self.diagnostics_dir = self.base_dir / "diagnostics"
        self.calibration_dir = self.base_dir / "calibration"
        self.metadata_dir = self.base_dir / "metadata"
        self.exports_dir = self.base_dir / "exports"
        
        self._initialize_directories()
        
    def _initialize_directories(self):
        """Creates the isolated directory structure for this session."""
        for directory in [
            self.inputs_dir, self.frames_dir, self.poses_dir, 
            self.depth_dir, self.geometry_dir, self.diagnostics_dir,
            self.calibration_dir, self.metadata_dir, self.exports_dir
        ]:
            directory.mkdir(parents=True, exist_ok=True)
            
    def get_path(self, relative_path: str) -> Path:
        """Returns a path safely resolved inside the session's base directory."""
        return self.base_dir / relative_path
