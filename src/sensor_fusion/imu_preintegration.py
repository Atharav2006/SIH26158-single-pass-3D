import math
import numpy as np
from typing import List, Optional, Tuple

from src.sensor_fusion.imu_types import IMUMeasurement, PreintegratedNavState

def skew_symmetric(v: np.ndarray) -> np.ndarray:
    """Compute 3x3 skew-symmetric matrix [v]x from 3D vector."""
    return np.array([
        [0.0,    -v[2],   v[1]],
        [v[2],    0.0,   -v[0]],
        [-v[1],   v[0],   0.0]
    ], dtype=np.float64)

def so3_exp(phi: np.ndarray) -> np.ndarray:
    """
    SO(3) Exponential map via Rodrigues' formula.
    Maps Lie algebra vector phi in so(3) (axis-angle) to Lie group SO(3) rotation matrix.
    """
    theta = float(np.linalg.norm(phi))
    if theta < 1e-10:
        return np.eye(3, dtype=np.float64)

    K = skew_symmetric(phi / theta)
    return np.eye(3, dtype=np.float64) + math.sin(theta) * K + (1.0 - math.cos(theta)) * (K @ K)

def preintegrate_imu_measurements(
    measurements: List[IMUMeasurement],
    accel_bias: Optional[np.ndarray] = None,
    gyro_bias: Optional[np.ndarray] = None
) -> PreintegratedNavState:
    """
    Preintegrate a sequence of IMU measurements between two keyframe timestamps 
    using actual irregular intervals dt_i = t_{i+1} - t_i (double precision).
    
    Equations:
      Delta_R_{i+1} = Delta_R_i * Exp((omega_i - b_g) * dt_i)
      Delta_v_{i+1} = Delta_v_i + Delta_R_i * (a_i - b_a) * dt_i
      Delta_p_{i+1} = Delta_p_i + Delta_v_i * dt_i + 0.5 * Delta_R_i * (a_i - b_a) * dt_i^2
    """
    if accel_bias is None:
        accel_bias = np.zeros(3, dtype=np.float64)
    else:
        accel_bias = np.asarray(accel_bias, dtype=np.float64)

    if gyro_bias is None:
        gyro_bias = np.zeros(3, dtype=np.float64)
    else:
        gyro_bias = np.asarray(gyro_bias, dtype=np.float64)

    n = len(measurements)
    if n == 0:
        return PreintegratedNavState(
            delta_R=np.eye(3, dtype=np.float64),
            delta_v=np.zeros(3, dtype=np.float64),
            delta_p=np.zeros(3, dtype=np.float64),
            integration_time_s=0.0,
            sample_count=0
        )

    if n == 1:
        return PreintegratedNavState(
            delta_R=np.eye(3, dtype=np.float64),
            delta_v=np.zeros(3, dtype=np.float64),
            delta_p=np.zeros(3, dtype=np.float64),
            integration_time_s=0.0,
            sample_count=1
        )

    delta_R = np.eye(3, dtype=np.float64)
    delta_v = np.zeros(3, dtype=np.float64)
    delta_p = np.zeros(3, dtype=np.float64)
    total_time = 0.0

    for i in range(n - 1):
        m_curr = measurements[i]
        m_next = measurements[i + 1]
        dt = float(m_next.timestamp_seconds - m_curr.timestamp_seconds)

        if dt < 0.0:
            raise ValueError(f"Non-monotonic timestamp detected: {m_curr.timestamp_seconds} -> {m_next.timestamp_seconds}")
        if dt == 0.0:
            continue

        # Corrected measurements
        omega_unbiased = np.asarray(m_curr.gyro, dtype=np.float64) - gyro_bias
        accel_unbiased = np.asarray(m_curr.accel, dtype=np.float64) - accel_bias

        # Rotation step
        dR_step = so3_exp(omega_unbiased * dt)

        # Position step
        delta_p = delta_p + delta_v * dt + 0.5 * (delta_R @ accel_unbiased) * (dt ** 2)

        # Velocity step
        delta_v = delta_v + (delta_R @ accel_unbiased) * dt

        # Update rotation
        delta_R = delta_R @ dR_step

        total_time += dt

    return PreintegratedNavState(
        delta_R=delta_R,
        delta_v=delta_v,
        delta_p=delta_p,
        integration_time_s=float(total_time),
        sample_count=n
    )

def predict_nav_state(
    R_i: np.ndarray,
    p_i: np.ndarray,
    v_i: np.ndarray,
    preintegrated: PreintegratedNavState,
    gravity_world: np.ndarray = np.array([0.0, 0.0, -9.80665], dtype=np.float64)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Project world navigation state at timestamp j from initial state at timestamp i 
    and preintegrated delta measurements:
    
      R_j = R_i * Delta_R
      v_j = v_i + g_world * Delta_T + R_i * Delta_v
      p_j = p_i + v_i * Delta_T + 0.5 * g_world * Delta_T^2 + R_i * Delta_p
    """
    dT = preintegrated.integration_time_s
    R_j = R_i @ preintegrated.delta_R
    v_j = v_i + gravity_world * dT + R_i @ preintegrated.delta_v
    p_j = p_i + v_i * dT + 0.5 * gravity_world * (dT ** 2) + R_i @ preintegrated.delta_p
    return R_j, p_j, v_j
