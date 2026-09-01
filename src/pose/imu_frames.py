import numpy as np
from typing import Tuple, Union

def frd_to_flu(
    accel_frd: np.ndarray,
    gyro_frd: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transform linear acceleration and angular velocity vectors from 
    Forward-Right-Down (FRD / NED body frame) to Forward-Left-Up (FLU robotic body frame).
    
    Transformation matrix:
      R_flu_frd = diag(1, -1, -1)
      
    v_flu = [v_x, -v_y, -v_z]^T
    """
    accel_flu = np.array([accel_frd[0], -accel_frd[1], -accel_frd[2]], dtype=np.float64)
    gyro_flu = np.array([gyro_frd[0], -gyro_frd[1], -gyro_frd[2]], dtype=np.float64)
    return accel_flu, gyro_flu

def flu_to_frd(
    accel_flu: np.ndarray,
    gyro_flu: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transform linear acceleration and angular velocity vectors from 
    Forward-Left-Up (FLU) back to Forward-Right-Down (FRD).
    
    Since R = diag(1, -1, -1) is involutory (R = R^-1 = R^T), the inverse transformation is identical.
    """
    accel_frd = np.array([accel_flu[0], -accel_flu[1], -accel_flu[2]], dtype=np.float64)
    gyro_frd = np.array([gyro_flu[0], -gyro_flu[1], -gyro_flu[2]], dtype=np.float64)
    return accel_frd, gyro_frd

def raw_sensor_to_flu(
    accel_raw: np.ndarray,
    gyro_raw: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transform raw sensor measurements to internal FLU frame.
    In Zurich Urban MAV dataset, raw IMU channels are logged in native FRD:
      ax = forward, ay = right, az = down (reaction force against gravity is -9.18 m/s^2 along az).
    In FLU:
      ax = forward, ay = left, az = up (reaction force against gravity is +9.18 m/s^2 along az).
    """
    return frd_to_flu(accel_raw, gyro_raw)
