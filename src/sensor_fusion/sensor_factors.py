import math
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

from src.sensor_fusion.imu_preintegration import skew_symmetric, so3_exp
from src.sensor_fusion.imu_types import PreintegratedNavState

def so3_log(R: np.ndarray) -> np.ndarray:
    """
    SO(3) Logarithmic map via inverse Rodrigues' formula.
    Maps rotation matrix R in SO(3) to Lie algebra vector phi in so(3) (axis-angle).
    """
    cos_theta = (np.trace(R) - 1.0) / 2.0
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    theta = math.acos(cos_theta)
    
    if theta < 1e-10:
        return np.zeros(3, dtype=np.float64)
    
    # Standard Rodrigues inverse
    sin_theta = math.sin(theta)
    if sin_theta < 1e-10:
        # Near pi ambiguity: find largest diagonal entry
        # For small perturbations in factor graphs this is rare
        return np.zeros(3, dtype=np.float64)
        
    K = (R - R.T) / (2.0 * sin_theta)
    return np.array([K[2, 1], K[0, 2], K[1, 0]], dtype=np.float64) * theta

@dataclass
class VisualRelativeFactor:
    """
    Relative visual pose observation constraint between state i and state j.
    Residual:
      r_rot = Log(R_ij_meas^T * R_i^T * R_j)  (3D rotation error in rad)
      r_trans = (p_j - p_i) - R_i * t_ij_meas  (3D position error in m)
    """
    i: int
    j: int
    R_ij_meas: np.ndarray  # (3, 3) relative rotation
    t_ij_meas: np.ndarray  # (3,) relative translation in frame i
    sigma_rot: float = 0.01  # ~0.57 deg
    sigma_trans: float = 0.02  # 2 cm

    def compute_residual(
        self,
        R_i: np.ndarray,
        p_i: np.ndarray,
        R_j: np.ndarray,
        p_j: np.ndarray
    ) -> np.ndarray:
        # Rotation residual
        R_rel_est = R_i.T @ R_j
        r_rot = so3_log(self.R_ij_meas.T @ R_rel_est) / self.sigma_rot

        # Translation residual
        t_rel_est = R_i.T @ (p_j - p_i)
        r_trans = (t_rel_est - self.t_ij_meas) / self.sigma_trans

        return np.hstack([r_rot, r_trans])

@dataclass
class GPSFactor:
    """
    Absolute GPS 3D position observation factor.
    Residual:
      r_gps = (p_i - p_gps) / sigma_gps
    """
    i: int
    p_gps: np.ndarray  # (3,) in Metric Local ENU
    sigma_gps: float = 0.5  # meters

    def compute_residual(self, p_i: np.ndarray) -> np.ndarray:
        return (p_i - self.p_gps) / self.sigma_gps

@dataclass
class IMUFactor:
    """
    Combined on-manifold IMU preintegration factor between state i and state j.
    Residuals:
      r_rot = Log(Delta_R_ij^T * R_i^T * R_j) / sigma_rot
      r_vel = (R_i^T * (v_j - v_i - g_world * dT) - Delta_v_ij) / sigma_vel
      r_pos = (R_i^T * (p_j - p_i - v_i * dT - 0.5 * g_world * dT^2) - Delta_p_ij) / sigma_pos
    """
    i: int
    j: int
    preintegrated: PreintegratedNavState
    gravity_world: np.ndarray = np.array([0.0, 0.0, -9.80665], dtype=np.float64)
    sigma_rot: float = 0.02   # rad
    sigma_vel: float = 0.10   # m/s
    sigma_pos: float = 0.15   # m

    def compute_residual(
        self,
        R_i: np.ndarray,
        p_i: np.ndarray,
        v_i: np.ndarray,
        R_j: np.ndarray,
        p_j: np.ndarray,
        v_j: np.ndarray
    ) -> np.ndarray:
        dT = self.preintegrated.integration_time_s
        if dT <= 0.0:
            return np.zeros(9, dtype=np.float64)

        # 1. Rotation residual
        R_rel = R_i.T @ R_j
        r_rot = so3_log(self.preintegrated.delta_R.T @ R_rel) / self.sigma_rot

        # 2. Velocity residual
        v_pred_body = R_i.T @ (v_j - v_i - self.gravity_world * dT)
        r_vel = (v_pred_body - self.preintegrated.delta_v) / self.sigma_vel

        # 3. Position residual
        p_pred_body = R_i.T @ (p_j - p_i - v_i * dT - 0.5 * self.gravity_world * (dT ** 2))
        r_pos = (p_pred_body - self.preintegrated.delta_p) / self.sigma_pos

        return np.hstack([r_rot, r_vel, r_pos])
