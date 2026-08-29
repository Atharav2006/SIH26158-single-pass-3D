import logging
import os
from pathlib import Path
from typing import Optional
from src.config import load_config

def setup_logging(name: str = "sih_pipeline", log_level: Optional[str] = None) -> logging.Logger:
    """Sets up and returns a configured logger using parameters from the config system."""
    config = load_config()
    
    # Retrieve configuration values with fallbacks
    fmt = config.get("logging.format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    level_str = log_level or config.get("logging.level", "INFO")
    log_to_file = config.get("logging.log_to_file", True)
    log_file_path = config.get("logging.log_file_path", "outputs/reports/pipeline.log")
    
    # Convert string log level to numeric logging value
    numeric_level = getattr(logging, level_str.upper(), logging.INFO)
    
    logger = logging.getLogger(name)
    logger.setLevel(numeric_level)
    
    # Avoid duplicate handlers if setup is called multiple times
    if logger.handlers:
        return logger
        
    formatter = logging.Formatter(fmt)
    
    # Stream Handler (console)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File Handler
    if log_to_file:
        try:
            # Resolve log file path relative to project root
            project_root = Path(__file__).resolve().parent.parent
            full_log_path = project_root / log_file_path
            
            # Ensure the target folder exists
            full_log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(str(full_log_path), encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            # Fallback output to stderr if file handler creation fails
            sys_logger = logging.getLogger("logger_setup")
            sys_logger.warning(f"Could not setup file logging handler: {e}")
            
    return logger
