from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum

class MetricAnchorCategory(str, Enum):
    CALIBRATED_STEREO = "calibrated_stereo"
    RGB_D = "rgb_d"
    LIDAR = "lidar"
    RTK_PPK_GEOMETRY = "rtk_ppk_supported_geometry"
    SURVEYED_CONTROL_POINTS = "surveyed_control_points"
    EXTERNALLY_MEASURED_DEPTH = "externally_measured_depth"
    EXPLICIT_REGISTERED_REFERENCE = "explicit_registered_reference"

@dataclass
class ReconstructionResult:
    geometry_path: str
    metric: bool
    scale_type: str
    coordinate_frame: str
    status: str
    confidence_available: bool = False
    uncertainty_available: bool = False
    camera_trajectory_path: Optional[str] = None
    anchor_source: Optional[MetricAnchorCategory] = None
    provenance: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Enforces the strict reconstruction mode contract."""
        if self.metric:
            if self.scale_type != "metric":
                raise ValueError("Metric reconstruction must specify scale_type='metric'.")
            if self.anchor_source is None:
                raise ValueError("Metric mode MUST fail closed if anchor_source is missing.")
            if not isinstance(self.anchor_source, MetricAnchorCategory):
                raise ValueError(f"anchor_source must be a legitimate MetricAnchorCategory. Found: {self.anchor_source}")
            if not self.provenance:
                raise ValueError("Metric mode MUST fail closed if provenance is missing.")
        else:
            if self.scale_type != "relative":
                raise ValueError("Relative reconstruction must specify scale_type='relative'.")
            if self.anchor_source is not None:
                raise ValueError("Relative reconstruction cannot declare a metric anchor source.")
