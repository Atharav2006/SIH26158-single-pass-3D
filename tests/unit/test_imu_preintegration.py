import pytest
import numpy as np
import math

from src.sensor_fusion.imu_types import IMUMeasurement, PreintegratedNavState
from src.sensor_fusion.imu_preintegration import (
    preintegrate_imu_measurements,
    predict_nav_state,
    so3_exp
)

def test_preintegration_zero_rotation_and_acceleration():
    """Test zero angular velocity and zero acceleration yields identity rotation and zero displacement."""
    meas = [
        IMUMeasurement(0.0, np.zeros(3), np.zeros(3)),
        IMUMeasurement(0.1, np.zeros(3), np.zeros(3)),
        IMUMeasurement(0.2, np.zeros(3), np.zeros(3)),
    ]
    res = preintegrate_imu_measurements(meas)

    assert math.isclose(res.integration_time_s, 0.2, rel_tol=1e-7)
    assert res.sample_count == 3
    assert np.allclose(res.delta_R, np.eye(3), atol=1e-9)
    assert np.allclose(res.delta_v, np.zeros(3), atol=1e-9)
    assert np.allclose(res.delta_p, np.zeros(3), atol=1e-9)

def test_preintegration_constant_pure_rotation():
    """Test constant angular velocity around Z-axis yields exact expected 90-degree yaw rotation."""
    w_z = math.pi / 2.0  # 90 deg/s
    meas = [
        IMUMeasurement(0.0, np.zeros(3), np.array([0.0, 0.0, w_z])),
        IMUMeasurement(0.5, np.zeros(3), np.array([0.0, 0.0, w_z])),
        IMUMeasurement(1.0, np.zeros(3), np.array([0.0, 0.0, w_z])),
    ]
    res = preintegrate_imu_measurements(meas)

    assert math.isclose(res.integration_time_s, 1.0, rel_tol=1e-7)
    # Expected 90 degree rotation matrix about Z
    R_expected = np.array([
        [0.0, -1.0, 0.0],
        [1.0,  0.0, 0.0],
        [0.0,  0.0, 1.0]
    ])
    assert np.allclose(res.delta_R, R_expected, atol=1e-7)

def test_preintegration_constant_linear_acceleration():
    """Test constant linear acceleration yields exact kinematic velocity v=at and position p=0.5*a*t^2."""
    a_const = np.array([2.0, -1.0, 0.5])
    t_total = 2.0
    N = 21
    ts = np.linspace(0.0, t_total, N)
    meas = [IMUMeasurement(t, a_const, np.zeros(3)) for t in ts]

    res = preintegrate_imu_measurements(meas)

    assert math.isclose(res.integration_time_s, t_total, rel_tol=1e-7)
    assert np.allclose(res.delta_R, np.eye(3), atol=1e-9)

    v_expected = a_const * t_total
    p_expected = 0.5 * a_const * (t_total ** 2)

    assert np.allclose(res.delta_v, v_expected, atol=1e-7)
    assert np.allclose(res.delta_p, p_expected, atol=1e-7)

def test_preintegration_irregular_timestamps():
    """Test preintegration with irregular time intervals (e.g. 80ms, 120ms, 95ms)."""
    ts = [0.0, 0.08, 0.20, 0.295, 0.40]
    a_const = np.array([1.0, 0.0, 0.0])
    meas = [IMUMeasurement(t, a_const, np.zeros(3)) for t in ts]

    res = preintegrate_imu_measurements(meas)
    assert math.isclose(res.integration_time_s, 0.40, rel_tol=1e-7)
    assert np.allclose(res.delta_v, a_const * 0.40, atol=1e-7)
    assert np.allclose(res.delta_p, 0.5 * a_const * (0.40 ** 2), atol=1e-7)

def test_preintegration_zero_duration_edge_cases():
    """Test N=0 and N=1 edge cases gracefully return identity."""
    res0 = preintegrate_imu_measurements([])
    assert res0.sample_count == 0
    assert res0.integration_time_s == 0.0
    assert np.allclose(res0.delta_R, np.eye(3))

    res1 = preintegrate_imu_measurements([IMUMeasurement(1.0, np.ones(3), np.ones(3))])
    assert res1.sample_count == 1
    assert res1.integration_time_s == 0.0
    assert np.allclose(res1.delta_R, np.eye(3))

def test_preintegration_bias_compensation():
    """Test that applying gyro and accel bias parameters correctly cancels sensor bias."""
    b_g = np.array([0.05, -0.02, 0.01])
    b_a = np.array([0.2, -0.1, 0.05])

    # Measurement with bias added
    meas = [
        IMUMeasurement(0.0, b_a, b_g),
        IMUMeasurement(0.5, b_a, b_g),
        IMUMeasurement(1.0, b_a, b_g),
    ]

    res = preintegrate_imu_measurements(meas, accel_bias=b_a, gyro_bias=b_g)

    assert np.allclose(res.delta_R, np.eye(3), atol=1e-9)
    assert np.allclose(res.delta_v, np.zeros(3), atol=1e-9)
    assert np.allclose(res.delta_p, np.zeros(3), atol=1e-9)

def test_predict_nav_state_stationary_gravity_compensation():
    """Test that a stationary vehicle resting on ground stays stationary after gravity compensation."""
    # Under gravity g_world = [0, 0, -9.80665], body measures upward specific force [0, 0, +9.80665] in FLU
    g_world = np.array([0.0, 0.0, -9.80665])
    a_body_measured = np.array([0.0, 0.0, 9.80665])

    ts = np.linspace(0.0, 1.0, 11)
    meas = [IMUMeasurement(t, a_body_measured, np.zeros(3)) for t in ts]

    preint = preintegrate_imu_measurements(meas)

    R_i = np.eye(3)
    p_i = np.array([10.0, 20.0, 5.0])
    v_i = np.zeros(3)

    R_j, p_j, v_j = predict_nav_state(R_i, p_i, v_i, preint, gravity_world=g_world)

    assert np.allclose(R_j, R_i, atol=1e-7)
    assert np.allclose(v_j, v_i, atol=1e-7)
    assert np.allclose(p_j, p_i, atol=1e-7)
