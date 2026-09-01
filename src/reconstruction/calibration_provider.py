from enum import Enum
from pathlib import Path
from typing import Optional

class CalibrationSource(str, Enum):
    FULL_OPENCV = "FULL_OPENCV"
    SUPPLIED_INTRINSICS = "SUPPLIED_INTRINSICS"
    COLMAP_ESTIMATED = "COLMAP_ESTIMATED"
    NOT_AVAILABLE = "NOT_AVAILABLE"

class CalibrationProvider:
    @staticmethod
    def identify(calibration_path: Optional[Path]) -> CalibrationSource:
        if calibration_path is None:
            return CalibrationSource.NOT_AVAILABLE
        return CalibrationSource.SUPPLIED_INTRINSICS
