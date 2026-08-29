import os
import sys
from pathlib import Path

# Ensure 'project_root' is in the python path so 'src' can be imported as a package
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

def test_directories_exist():
    """Verify that all required project directories exist."""
    required_dirs = [
        "configs",
        "data/raw",
        "data/processed",
        "data/samples",
        "data/ground_truth",
        "src/ingestion",
        "src/preprocessing",
        "src/frame_selection",
        "src/pose",
        "src/depth",
        "src/reconstruction",
        "src/sensor_fusion",
        "src/dynamic_objects",
        "src/occlusion",
        "src/confidence",
        "src/metrics",
        "src/visualization",
        "pipelines/baseline",
        "pipelines/experiments",
        "pipelines/production",
        "tests/unit",
        "tests/integration",
        "tests/regression",
        "scripts",
        "notebooks",
        "outputs/pointclouds",
        "outputs/meshes",
        "outputs/renders",
        "outputs/reports"
    ]
    for d in required_dirs:
        dir_path = project_root / d
        assert dir_path.is_dir(), f"Required directory {d} does not exist"

def test_imports():
    """Verify that all Python submodules in src are importable under the 'src' package."""
    modules = [
        "src.ingestion",
        "src.preprocessing",
        "src.frame_selection",
        "src.pose",
        "src.depth",
        "src.reconstruction",
        "src.sensor_fusion",
        "src.dynamic_objects",
        "src.occlusion",
        "src.confidence",
        "src.metrics",
        "src.visualization",
        "src.config",
        "src.logger",
        "src.version"
    ]
    for mod in modules:
        try:
            # Dynamically import the module
            imported = __import__(mod, fromlist=["*"])
            assert imported is not None, f"Failed to import module {mod}"
        except ImportError as e:
            assert False, f"Could not import {mod}: {e}"

def test_project_version():
    """Verify project version matches expectation."""
    import src.version
    assert hasattr(src.version, "__version__"), "version module has no __version__ attribute"
    assert src.version.__version__ == "0.1.0"

    import src
    assert hasattr(src, "__version__"), "src package has no __version__ attribute"
    assert src.__version__ == "0.1.0"

def test_config_system():
    """Verify the configuration system loads and retrieves config properties."""
    from src.config import load_config
    cfg = load_config()
    assert cfg is not None
    assert cfg.get("project_name") == "SIH26158-single-pass-3D"
    assert cfg.get("logging.level") == "INFO"
    assert cfg.get("nonexistent.key", "fallback") == "fallback"

def test_logging_system(tmp_path):
    """Verify the logging system sets up loggers successfully."""
    from src.logger import setup_logging
    import logging
    
    logger = setup_logging(name="test_logger", log_level="DEBUG")
    assert isinstance(logger, logging.Logger)
    assert logger.level == logging.DEBUG
    
    # Try a simple log statement
    logger.debug("Testing basic logging system setup")
