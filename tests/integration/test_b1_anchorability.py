import pytest
import json
import csv
from pathlib import Path

OUT_DIR = Path("outputs/reports/zurich_mav/b1")
CORR_CSV = OUT_DIR / "gps_colmap_correspondences.csv"
JSON_PATH = OUT_DIR / "gps_anchorability.json"
CORR_PNG = OUT_DIR / "gps_colmap_correspondence.png"
COND_PNG = OUT_DIR / "gps_conditioning.png"
SENS_PNG = OUT_DIR / "sim3_noise_sensitivity.png"

def test_correspondence_csv_completeness():
    """Test that gps_colmap_correspondences.csv contains 350 non-empty correspondence pairs."""
    assert CORR_CSV.is_file(), f"Missing {CORR_CSV}"
    with open(CORR_CSV, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 350, f"Expected 350 correspondences, found {len(rows)}"
    for r in rows:
        assert r["imgid"] and r["filename"]
        assert float(r["gps_east_local_m"]) is not None
        assert float(r["colmap_x"]) is not None

def test_anchorability_json_schema_and_classification():
    """Test that gps_anchorability.json contains valid conditioning and readiness decision."""
    assert JSON_PATH.is_file(), f"Missing {JSON_PATH}"
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["correspondence_count"] == 350
    gps_cond = data["geometric_conditioning"]["gps_local_enu"]
    assert gps_cond["degeneracy_flags"]["rank"] == 3
    assert gps_cond["condition_number_svd"] < 10.0

    loo = data["leave_one_out_conditioning"]
    assert loo["total_loo_iterations"] == 350
    assert not loo["dominating_point_detected"]

    decision = data["b1_readiness_decision"]["status"]
    assert decision in ["B1_READY", "B1_CONDITIONALLY_READY", "B1_NOT_READY"]
    assert decision == "B1_CONDITIONALLY_READY"

def test_visualizations_exist_and_non_empty():
    """Test that all three anchorability visualization PNGs exist on disk."""
    for p in [CORR_PNG, COND_PNG, SENS_PNG]:
        assert p.is_file(), f"Missing visualization artifact: {p}"
        assert p.stat().st_size > 5000, f"Artifact file too small: {p} ({p.stat().st_size} bytes)"
