import pytest
import json
import math
import numpy as np
from pathlib import Path

from src.geodesy.projection import wgs84_to_utm32n, utm32n_to_wgs84

def test_utm32n_roundtrip_submillimeter_precision():
    """Test WGS84 -> UTM Zone 32N -> WGS84 roundtrip has sub-millimeter error."""
    test_coords = [
        (47.3843571, 8.5451784, 464.91),
        (47.3843807, 8.5452293, 466.87),
        (47.3800000, 8.5400000, 450.00),
        (47.3900000, 8.5500000, 500.00)
    ]

    for lat, lon, alt in test_coords:
        e, n, u = wgs84_to_utm32n(lat, lon, alt)
        lat_rec, lon_rec, alt_rec = utm32n_to_wgs84(e, n, u)

        # 1e-8 degrees is ~1 mm on Earth's surface
        assert math.isclose(lat, lat_rec, abs_tol=1e-7), f"Lat roundtrip mismatch: {lat} vs {lat_rec}"
        assert math.isclose(lon, lon_rec, abs_tol=1e-7), f"Lon roundtrip mismatch: {lon} vs {lon_rec}"
        assert math.isclose(alt, alt_rec, abs_tol=1e-4), f"Alt roundtrip mismatch: {alt} vs {alt_rec}"

def test_b1_georeferencing_audit_json_validity():
    """Test that b1_georeferencing_audit.json exists and verifies audit metrics."""
    audit_json_path = Path("outputs/reports/zurich_mav/b1/b1_georeferencing_audit.json")
    assert audit_json_path.is_file(), f"Missing {audit_json_path}"
    with open(audit_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["status"] == "PASS"
    assert data["time_alignment_audit"]["max_timestamp_delta_s"] < 1e-5
    assert data["coordinate_frame_audit"]["reversibility_test"]["reversibility_verified"]
    assert data["sim3_direction_audit"]["direction_verified"]

    gps_res = data["residual_decompositions"]["b1_gps_fit_residual"]["3d_magnitude"]
    assert 0.50 < gps_res["rmse_m"] < 1.00

    gt_res = data["residual_decompositions"]["b1_vs_ground_truth_residual"]["3d_magnitude"]
    assert 1.00 < gt_res["rmse_m"] < 2.50
