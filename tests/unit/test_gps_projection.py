import pytest
import math
import numpy as np

from src.geodesy.projection import wgs84_to_utm32n, utm32n_to_local_enu

def test_wgs84_to_utm32n_zurich_reference():
    """Test UTM Zone 32N conversion on Zurich Urban MAV coordinates."""
    lat = 47.3843571
    lon = 8.5451784
    alt = 464.91

    e, n, u = wgs84_to_utm32n(lat, lon, alt)

    # In UTM Zone 32N (central meridian 9 deg E):
    # Easting should be ~465,670 m, Northing ~5,247,978 m
    assert 465600.0 < e < 465800.0, f"Unexpected Easting: {e}"
    assert 5247800.0 < n < 5248100.0, f"Unexpected Northing: {n}"
    assert u == alt

def test_utm32n_to_local_enu_origin():
    """Test that origin point maps exactly to (0, 0, 0) in local ENU."""
    e0, n0, u0 = 465670.7068, 5247978.0338, 464.91
    de, dn, du = utm32n_to_local_enu(e0, n0, u0, e0, n0, u0)
    assert de == 0.0
    assert dn == 0.0
    assert du == 0.0

def test_local_enu_relative_offsets():
    """Test relative metric offsets in local ENU frame."""
    e0, n0, u0 = 500000.0, 5000000.0, 400.0
    e1, n1, u1 = 500015.5, 5000020.2, 405.8

    de, dn, du = utm32n_to_local_enu(e1, n1, u1, e0, n0, u0)
    assert math.isclose(de, 15.5, abs_tol=1e-5)
    assert math.isclose(dn, 20.2, abs_tol=1e-5)
    assert math.isclose(du, 5.8, abs_tol=1e-5)
