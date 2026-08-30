import os
import json
from pathlib import Path
from typing import Any, Dict

# Default configuration dictionary fallback
DEFAULT_CONFIG = {
    "project_name": "SIH26158-single-pass-3D",
    "version": "0.1.0",
    "logging": {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "log_to_file": True,
        "log_file_path": "outputs/reports/pipeline.log"
    },
    "ingestion": {
        "allowed_formats": [".mp4", ".avi", ".mov", ".mkv"],
        "extraction_fps": None,
        "output_image_format": "jpg",
        "output_image_quality": 2,
        "output_width": None,
        "output_height": None
    },
    "frame_selection": {
        "method": "laplacian_variance",
        "sharpness_threshold": 100.0,
        "keyframe_interval": 10
    },
    "pose": {
        "sfm_method": "colmap",
        "max_num_features": 8192
    },
    "depth": {
        "depth_estimator": "none",
        "min_depth": 0.1,
        "max_depth": 100.0
    },
    "reconstruction": {
        "voxel_size": 0.05,
        "max_correspondence_distance": 0.07
    }
}

class Configuration:
    """Simple configuration management class."""
    def __init__(self, config_dict: Dict[str, Any] = None):
        self._config = config_dict or DEFAULT_CONFIG

    def get(self, key_path: str, default: Any = None) -> Any:
        """Retrieve a configuration value using dot-notation path, e.g., 'logging.level'."""
        keys = key_path.split(".")
        current = self._config
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        return current

    @property
    def raw(self) -> Dict[str, Any]:
        """Return the raw configuration dictionary."""
        return self._config


def load_config(config_path: str = None) -> Configuration:
    """Load configuration from a JSON file, falling back to default configuration if not found."""
    if config_path is None:
        # Find project root relative to this file
        current_dir = Path(__file__).resolve().parent
        config_path = current_dir.parent / "configs" / "default_config.json"
    
    path = Path(config_path)
    if path.exists():
        try:
            with open(path, "r") as f:
                data = json.load(f)
            return Configuration(data)
        except Exception:
            # Fallback on parsing error
            pass
            
    return Configuration(DEFAULT_CONFIG)
