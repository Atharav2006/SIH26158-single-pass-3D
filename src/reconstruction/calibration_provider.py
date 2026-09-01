from enum import Enum

class CalibrationSource(str, Enum):
    FULL_OPENCV = "FULL_OPENCV"
    SUPPLIED_INTRINSICS = "SUPPLIED_INTRINSICS"
    NOT_AVAILABLE = "NOT_AVAILABLE"

class CalibrationProvider:
    @staticmethod
    def identify(calibration_path) -> CalibrationSource:
        if calibration_path is None:
            return CalibrationSource.NOT_AVAILABLE
        # Simplified abstraction
        return CalibrationSource.FULL_OPENCV
