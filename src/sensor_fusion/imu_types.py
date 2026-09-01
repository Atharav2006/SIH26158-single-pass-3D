from dataclasses import dataclass
import numpy as np
from typing import Optional, List

@dataclass
class IMUMeasurement:
    timestamp_seconds: float
    accel: np.ndarray  # (3,) in m/s^2 (specific force)
    gyro: np.ndarray   # (3,) in rad/s (angular rates)

@dataclass
class PreintegratedNavState:
    delta_R: np.ndarray       # (3, 3) relative rotation matrix
    delta_v: np.ndarray       # (3,) relative velocity in initial body frame (m/s)
    delta_p: np.ndarray       # (3,) relative position displacement in initial body frame (m)
    integration_time_s: float # total duration sum(dt_i)
    sample_count: int         # total IMU measurements integrated
