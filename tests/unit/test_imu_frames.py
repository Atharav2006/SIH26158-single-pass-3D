import pytest
import numpy as np
import math

from src.pose.imu_frames import frd_to_flu, flu_to_frd, raw_sensor_to_flu

def test_frd_to_flu_conversion():
    """Test Forward-Right-Down to Forward-Left-Up transformation."""
    a_frd = np.array([1.5, 2.0, -9.81])
    w_frd = np.array([0.1, -0.2, 0.3])

    a_flu, w_flu = frd_to_flu(a_frd, w_frd)

    assert a_flu[0] == 1.5
    assert a_flu[1] == -2.0
    assert a_flu[2] == +9.81

    assert w_flu[0] == 0.1
    assert w_flu[1] == 0.2
    assert w_flu[2] == -0.3

def test_frame_transformation_involutory_roundtrip():
    """Test that applying FRD->FLU and then FLU->FRD is exact identity."""
    np.random.seed(42)
    a_orig = np.random.randn(3) * 10.0
    w_orig = np.random.randn(3) * 2.0

    a_flu, w_flu = frd_to_flu(a_orig, w_orig)
    a_rec, w_rec = flu_to_frd(a_flu, w_flu)

    assert np.allclose(a_orig, a_rec, atol=1e-12)
    assert np.allclose(w_orig, w_rec, atol=1e-12)

def test_raw_stationary_gravity_in_flu():
    """Test that stationary gravity specific force in raw FRD maps to positive vertical in FLU."""
    a_raw_stat = np.array([-0.1638, -0.1654, -9.1785])
    w_raw_stat = np.array([0.0113, -0.0397, -0.0245])

    a_flu, w_flu = raw_sensor_to_flu(a_raw_stat, w_raw_stat)

    assert a_flu[2] > 9.0, f"FLU vertical acceleration should be positive under gravity: {a_flu[2]}"
    assert math.isclose(np.linalg.norm(a_raw_stat), np.linalg.norm(a_flu), rel_tol=1e-10)
