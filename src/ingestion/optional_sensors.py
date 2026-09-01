from enum import Enum
from pathlib import Path
from typing import Optional

class SensorStatus(str, Enum):
    AVAILABLE = "AVAILABLE"
    NOT_AVAILABLE = "NOT_AVAILABLE"
    INVALID = "INVALID"
    UNSUPPORTED = "UNSUPPORTED"

class SensorDetector:
    @staticmethod
    def detect(path: Optional[Path]) -> SensorStatus:
        if path is None:
            return SensorStatus.NOT_AVAILABLE
        if not path.exists():
            return SensorStatus.INVALID
        if path.suffix not in ['.csv', '.json', '.txt', '.yaml']:
            return SensorStatus.UNSUPPORTED
        return SensorStatus.AVAILABLE
