"""
SIH26158 Depth Fusion - Metric Anchor Interface and Output Contract

This module defines the typed anchor structures, source provenance,
and metric depth output contracts for monocular depth calibration.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import numpy as np

class AnchorSource(Enum):
    B0_SPARSE_REPROJECTION = "B0_SPARSE_REPROJECTION"
    EXTERNAL_DEPTH_SENSOR = "EXTERNAL_DEPTH_SENSOR"
    USER_DEFINED = "USER_DEFINED"
    GROUND_TRUTH_EVALUATION_ONLY = "GROUND_TRUTH_EVALUATION_ONLY"

class CalibrationStatus(Enum):
    METRIC_ALIGNMENT_VALID = "METRIC_ALIGNMENT_VALID"
    METRIC_ALIGNMENT_UNSTABLE = "METRIC_ALIGNMENT_UNSTABLE"
    METRIC_SCALE_NOT_IDENTIFIABLE = "METRIC_SCALE_NOT_IDENTIFIABLE"

@dataclass
class MetricAnchor:
    """
    Explicit typed anchor point tying image pixel coordinates to a candidate metric depth.
    """
    pixel_u: float
    pixel_v: float
    frame_id: int
    metric_depth_m: float
    inv_depth_predicted: float
    confidence: float = 1.0
    source: AnchorSource = AnchorSource.B0_SPARSE_REPROJECTION
    provenance: str = "B0_COLMAP_SfM_Transformed_B1_B2"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.metric_depth_m <= 0:
            raise ValueError(f"Metric depth must be strictly positive, got {self.metric_depth_m}")
        if self.inv_depth_predicted <= 0:
            raise ValueError(f"Predicted inverse depth must be strictly positive, got {self.inv_depth_predicted}")

@dataclass
class MetricDepthOutput:
    """
    Typed output contract for calibrated or relative dense depth maps.
    """
    depth: np.ndarray
    confidence: np.ndarray
    metric: bool
    scale_a: Optional[float] = None
    shift_b: Optional[float] = None
    source: str = "MiDaS_small"
    calibration_method: str = "Affine_Inverse_Depth"
    calibration_status: CalibrationStatus = CalibrationStatus.METRIC_SCALE_NOT_IDENTIFIABLE
    uncertainty: Optional[np.ndarray] = None
    provenance: str = "SIH26158_B5_Engine"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_calibrated(self) -> bool:
        return self.metric and (self.calibration_status == CalibrationStatus.METRIC_ALIGNMENT_VALID)
