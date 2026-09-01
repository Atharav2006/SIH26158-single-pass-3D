from abc import ABC, abstractmethod
from pathlib import Path
import json

from .session import ReconstructionSession
from .mode_selector import ModeSelection

class ReconstructionBackend(ABC):
    @abstractmethod
    def prepare(self, session: ReconstructionSession, mode: ModeSelection):
        pass
        
    @abstractmethod
    def run(self, session: ReconstructionSession) -> str:
        """Runs the reconstruction and returns the path to the geometry."""
        pass

class RelativeDepthBackend(ReconstructionBackend):
    """
    Adapts the B5 relative depth fusion engine to the generalized session interface.
    """
    def prepare(self, session: ReconstructionSession, mode: ModeSelection):
        self.mode = mode
        
    def run(self, session: ReconstructionSession) -> str:
        # In a real implementation, this would invoke the B5 pointcloud_fusion module
        # parsing frames and poses from session.frames_dir and session.poses_dir
        
        # For the architectural contract, we mock the final execution 
        # to prove isolation and pipeline flow without requiring full ML inference.
        geom_path = session.get_path("geometry/pointcloud.ply")
        geom_path.parent.mkdir(parents=True, exist_ok=True)
        geom_path.write_text("ply\nformat ascii 1.0\nend_header\n")
        
        return str(geom_path)
