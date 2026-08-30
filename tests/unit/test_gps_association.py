import pytest
import csv
from pathlib import Path

def test_image_gps_association_exact_imgids():
    """Test that all 350 images have corresponding GPS records with identical timestamps."""
    gps_path = Path("outputs/reports/zurich_mav/gps.csv")
    imgs_path = Path("outputs/reports/zurich_mav/images.csv")

    with open(imgs_path, "r", encoding="utf-8") as f:
        imgs = list(csv.DictReader(f))

    with open(gps_path, "r", encoding="utf-8") as f:
        gps = list(csv.DictReader(f))

    assert len(imgs) == 350
    assert len(gps) >= 350

    for img in imgs:
        imgid = int(img["imgid"])
        g = gps[imgid - 1]
        img_ts = float(img["timestamp_seconds"])
        gps_ts = float(g["timestamp_seconds"])
        assert abs(img_ts - gps_ts) < 1e-5, f"Timestamp mismatch for imgid {imgid}: {img_ts} vs {gps_ts}"

def test_gps_timestamps_strictly_monotonic():
    """Test that GPS stream timestamps are strictly monotonically increasing."""
    gps_path = Path("outputs/reports/zurich_mav/gps.csv")
    with open(gps_path, "r", encoding="utf-8") as f:
        gps = list(csv.DictReader(f))[:350]

    ts_list = [float(r["timestamp_seconds"]) for r in gps]
    for i in range(len(ts_list) - 1):
        assert ts_list[i + 1] > ts_list[i], f"Non-monotonic timestamp at index {i}: {ts_list[i]} -> {ts_list[i+1]}"
