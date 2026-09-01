import cv2
from pathlib import Path
from dataclasses import dataclass

@dataclass
class VideoMetadata:
    valid: bool
    width: int
    height: int
    fps: float
    frame_count: int
    duration_sec: float

class VideoValidator:
    @staticmethod
    def validate(video_path: Path) -> VideoMetadata:
        """Validates video and extracts basic metadata."""
        if not video_path.exists():
            return VideoMetadata(False, 0, 0, 0.0, 0, 0.0)
            
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return VideoMetadata(False, 0, 0, 0.0, 0, 0.0)
            
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        cap.release()
        
        if fps <= 0 or frame_count <= 0:
            return VideoMetadata(False, width, height, fps, frame_count, 0.0)
            
        duration = frame_count / fps
        return VideoMetadata(True, width, height, fps, frame_count, duration)
