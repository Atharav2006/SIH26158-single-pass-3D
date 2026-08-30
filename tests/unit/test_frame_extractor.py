import os
import csv
import json
import pytest
from pathlib import Path

from src.ingestion.frame_extractor import FrameExtractor, extract_frames

@pytest.fixture
def test_video_path():
    project_root = Path(__file__).resolve().parent.parent.parent
    video_path = project_root / "data" / "samples" / "controlled_test" / "test_video.mp4"
    assert video_path.exists(), f"Test video not found at: {video_path}"
    return video_path

@pytest.fixture
def ground_truth(test_video_path):
    gt_path = test_video_path.parent / "ground_truth.json"
    assert gt_path.exists(), f"Ground truth JSON not found at: {gt_path}"
    with open(gt_path, "r", encoding="utf-8") as f:
        return json.load(f)

def test_frame_extractor_full_fps(tmp_path, test_video_path, ground_truth):
    output_dir = tmp_path / "extracted_full"
    extractor = FrameExtractor()
    meta = extractor.extract(test_video_path, output_dir)

    # 1. Correct frame count
    expected_frames = ground_truth["expected_frame_count"]
    assert meta["extracted_frame_count"] == expected_frames
    
    # 8 & 9. Image verification on disk
    frames_dir = output_dir / "frames"
    images = sorted(list(frames_dir.glob("frame_*.jpg")))
    assert len(images) == expected_frames

    # 2. Correct filename sequence & no extra images
    for i, img_path in enumerate(images, start=1):
        expected_filename = f"frame_{i:06d}.jpg"
        assert img_path.name == expected_filename

    # 3. Required CSV columns & 4. Sequential frame IDs & 5. Timestamp monotonicity
    csv_path = output_dir / "frame_index.csv"
    assert csv_path.exists()
    
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        assert "frame_id" in fieldnames
        assert "filename" in fieldnames
        assert "timestamp_seconds" in fieldnames

        rows = list(reader)
        assert len(rows) == expected_frames

        prev_timestamp = -1.0
        for i, row in enumerate(rows, start=1):
            assert int(row["frame_id"]) == i
            assert row["filename"] == f"frame_{i:06d}.jpg"
            ts = float(row["timestamp_seconds"])
            
            # Monotonicity
            assert ts > prev_timestamp
            prev_timestamp = ts

            # Check corresponding image exists
            assert (frames_dir / row["filename"]).exists()

        # 6. First timestamp is approx 0
        first_ts = float(rows[0]["timestamp_seconds"])
        assert abs(first_ts - 0.0) < 1e-4

        # 7. Last timestamp is within reasonable tolerance of duration
        last_ts = float(rows[-1]["timestamp_seconds"])
        expected_duration = ground_truth["expected_duration"]
        expected_last_ts = (expected_frames - 1) / ground_truth["expected_fps"]
        assert abs(last_ts - expected_last_ts) < 0.05
        assert abs(last_ts - expected_duration) < 0.1

    # 10. Extraction metadata agrees with source metadata
    json_path = output_dir / "extraction_metadata.json"
    assert json_path.exists()
    with open(json_path, mode="r", encoding="utf-8") as f:
        loaded_meta = json.load(f)

    assert loaded_meta["source_width"] == ground_truth["expected_width"]
    assert loaded_meta["source_height"] == ground_truth["expected_height"]
    assert abs(loaded_meta["source_fps"] - ground_truth["expected_fps"]) < 0.01
    assert abs(loaded_meta["source_duration_seconds"] - ground_truth["expected_duration"]) < 0.05
    assert loaded_meta["extracted_frame_count"] == expected_frames
    assert loaded_meta["output_width"] == ground_truth["expected_width"]
    assert loaded_meta["output_height"] == ground_truth["expected_height"]

def test_frame_extractor_custom_fps_and_resize(tmp_path, test_video_path):
    output_dir = tmp_path / "extracted_custom"
    extractor = FrameExtractor()
    meta = extractor.extract(
        test_video_path,
        output_dir,
        extraction_fps=5.0,
        output_width=640,
        output_height=360
    )

    assert meta["extraction_fps"] == 5.0
    assert meta["output_width"] == 640
    assert meta["output_height"] == 360
    # 10 seconds at 5 fps -> ~50 frames
    assert 48 <= meta["extracted_frame_count"] <= 52

    frames_dir = output_dir / "frames"
    images = list(frames_dir.glob("frame_*.jpg"))
    assert len(images) == meta["extracted_frame_count"]

def test_frame_extractor_invalid_inputs(tmp_path, test_video_path):
    extractor = FrameExtractor()

    # Non-existent video file
    with pytest.raises(FileNotFoundError):
        extractor.extract("non_existent_video.mp4", tmp_path / "out")

    # Invalid extension
    dummy_txt = tmp_path / "dummy.txt"
    dummy_txt.write_text("not a video")
    with pytest.raises(ValueError, match="Unsupported video format"):
        extractor.extract(dummy_txt, tmp_path / "out")

    # Invalid extraction FPS
    with pytest.raises(ValueError, match="Invalid extraction FPS"):
        extractor.extract(test_video_path, tmp_path / "out", extraction_fps=-5.0)
