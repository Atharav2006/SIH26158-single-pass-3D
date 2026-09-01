import pytest
import numpy as np
import math
from src.sensor_fusion.sensor_factors import VisualRelativeFactor, GPSFactor, IMUFactor
from src.sensor_fusion.imu_types import PreintegratedNavState

def test_visual_relative_factor_zero_residual():
    R_ij = np.eye(3)
    t_ij = np.array([1.0, 0.0, 0.0])
    factor = VisualRelativeFactor(0, 1, R_ij, t_ij, sigma_rot=1.0, sigma_trans=1.0)
    
    R_i = np.eye(3)
    p_i = np.zeros(3)
    R_j = np.eye(3)
    p_j = np.array([1.0, 0.0, 0.0])
    
    res = factor.compute_residual(R_i, p_i, R_j, p_j)
    assert np.allclose(res, 0.0)

def test_gps_factor_zero_residual():
    p_gps = np.array([10.0, 20.0, 30.0])
    factor = GPSFactor(0, p_gps, sigma_gps=1.0)
    
    res = factor.compute_residual(p_gps)
    assert np.allclose(res, 0.0)

def test_imu_factor_zero_residual():
    preint = PreintegratedNavState(
        delta_R=np.eye(3),
        delta_v=np.array([0.0, 0.0, 0.0]),
        delta_p=np.array([0.0, 0.0, 0.0]),
        integration_time_s=1.0,
        sample_count=10
    )
    # Gravity is [0, 0, -9.8] in world, so freefall gives acceleration of +9.8 upwards
    factor = IMUFactor(0, 1, preint, gravity_world=np.array([0.0, 0.0, -9.8]), sigma_rot=1.0, sigma_vel=1.0, sigma_pos=1.0)
    
    R_i = np.eye(3)
    p_i = np.zeros(3)
    v_i = np.zeros(3)
    
    R_j = np.eye(3)
    v_j = np.array([0.0, 0.0, -9.8])
    p_j = np.array([0.0, 0.0, -4.9])
    
    res = factor.compute_residual(R_i, p_i, v_i, R_j, p_j, v_j)
    assert np.allclose(res, 0.0)
