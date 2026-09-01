from dataclasses import dataclass
from typing import List, Optional
from .input_spec import VideoInputSpec
from .reconstruction_result import MetricAnchorCategory
from .pose_provider import PoseProvider, PoseSource
from .calibration_provider import CalibrationProvider, CalibrationSource

@dataclass
class ModeSelection:
    status: str
    selected_mode: Optional[str] = None
    missing_requirements: List[str] = None
    recommended_action: Optional[str] = None
    anchor_source: Optional[MetricAnchorCategory] = None
    
    def __post_init__(self):
        if self.missing_requirements is None:
            self.missing_requirements = []

class ModeSelector:
    @staticmethod
    def evaluate(spec: VideoInputSpec) -> ModeSelection:
        """
        Determines reconstruction feasibility and mode based on available inputs.
        Fails closed for metric reconstruction.
        """
        pose_source = PoseProvider.identify(spec.poses_path)
        calib_source = CalibrationProvider.identify(spec.calibration_path)
        
        missing = []
        if pose_source == PoseSource.NOT_AVAILABLE:
            missing.append("poses")
        if calib_source == CalibrationSource.NOT_AVAILABLE:
            missing.append("calibration")
            
        if missing:
            return ModeSelection(
                status="RECONSTRUCTION_BLOCKED",
                selected_mode="RELATIVE_RECONSTRUCTION",
                missing_requirements=missing,
                recommended_action="POSE_AND_CALIBRATION_REQUIRED: Provide a pose source or enable a supported pose estimator, and provide camera intrinsics/calibration."
            )
            
        # If we have basic requirements (pose + calib), we can do relative.
        # Determine if we have a metric anchor
        if spec.rtk_path is not None:
            # We treat RTK as a valid metric anchor for this example contract
            return ModeSelection(
                status="METRIC_RECONSTRUCTION_READY",
                selected_mode="METRIC_RECONSTRUCTION",
                missing_requirements=[],
                anchor_source=MetricAnchorCategory.RTK_PPK_GEOMETRY
            )
            
        # Default fallback
        return ModeSelection(
            status="RELATIVE_RECONSTRUCTION_READY",
            selected_mode="RELATIVE_RECONSTRUCTION",
            missing_requirements=[]
        )
