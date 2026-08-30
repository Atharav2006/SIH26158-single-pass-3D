import pytest
import json
import csv
from pathlib import Path

OUT_DIR = Path("outputs/reports/zurich_mav/b1")
JSON_PATH = OUT_DIR / "gps_quality.json"
OUTLIERS_PATH = OUT_DIR / "gps_outliers.csv"
RAW_PNG = OUT_DIR / "gps_vs_colmap_raw.png"
LOCAL_PNG = OUT_DIR / "gps_trajectory_local.png"

def test_gps_quality_json_deliverable():
    """Test that gps_quality.json exists and contains complete statistics."""
    assert JSON_PATH.is_file(), f"Missing {JSON_PATH}"
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["gps_statistics"]["b0_associated_gps_records"] == 350
    assert data["gps_statistics"]["total_gps_records_in_stream"] > 80000
    assert data["image_gps_association"]["associated_count"] == 350

    comp = data["colmap_comparison"]
    assert comp["total_correspondence_pairs"] == 350
    assert comp["gps_total_path_length_m"] > 0.0
    assert comp["colmap_total_path_length_units"] > 0.0

def test_gps_outliers_csv_deliverable():
    """Test that gps_outliers.csv exists."""
    assert OUTLIERS_PATH.is_file(), f"Missing {OUTLIERS_PATH}"

def test_gps_visualizations_exist_and_non_empty():
    """Test that both GPS visualization PNG artifacts exist on disk."""
    for p in [RAW_PNG, LOCAL_PNG]:
        assert p.is_file(), f"Missing artifact {p}"
        assert p.stat().st_size > 5000, f"Artifact file too small: {p} ({p.stat().st_size} bytes)"
