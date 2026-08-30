import os
import json
import pytest
from src.ingestion.video_metadata import get_video_metadata

def test_video_metadata_extraction():
    # Setup paths
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    video_path = os.path.join(project_root, "data", "samples", "controlled_test", "test_video.mp4")
    gt_path = os.path.join(project_root, "data", "samples", "controlled_test", "ground_truth.json")
    
    # Assert files exist
    assert os.path.exists(video_path), f"Test video not found at: {video_path}"
    assert os.path.exists(gt_path), f"Ground truth JSON not found at: {gt_path}"
    
    # Load ground truth
    with open(gt_path, "r") as f:
        gt = json.load(f)
        
    # Extract metadata
    metadata = get_video_metadata(video_path)
    
    # Validate fields
    assert metadata["width"] == gt["expected_width"]
    assert metadata["height"] == gt["expected_height"]
    
    # Check duration with small tolerance (0.05 seconds)
    assert abs(metadata["duration"] - gt["expected_duration"]) < 0.05
    
    # Check average frame rate (FPS) with tolerance (0.1 FPS)
    assert abs(metadata["average_frame_rate"] - gt["expected_fps"]) < 0.1
    
    # Check frame count if available
    if metadata["frame_count"] is not None:
        assert metadata["frame_count"] == gt["expected_frame_count"]
