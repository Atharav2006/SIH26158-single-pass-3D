import math
import time
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from scipy.optimize import least_squares

from src.sensor_fusion.imu_types import IMUMeasurement, PreintegratedNavState
from src.sensor_fusion.imu_preintegration import so3_exp
from src.sensor_fusion.sensor_factors import (
    VisualRelativeFactor,
    GPSFactor,
    IMUFactor,
    so3_log
)

class B2TrajectoryOptimizer:
    """
    Classical Nonlinear Least-Squares Multimodal Trajectory Fusion Optimizer (B2).
    Jointly optimizes keyframe orientations, positions, and velocities against:
      - Visual relative pose constraints
      - Absolute GPS position observations
      - Preintegrated IMU inertial constraints
      - Prior / smoothing regularization
    """
    def __init__(
        self,
        num_states: int,
        initial_rotations: List[np.ndarray],
        initial_positions: np.ndarray,
        initial_velocities: np.ndarray,
        gyro_bias: np.ndarray = np.zeros(3),
        accel_bias: np.ndarray = np.zeros(3),
        loss_type: str = "soft_l1",
        loss_scale: float = 1.0
    ):
        self.num_states = num_states
        self.R0_list = [np.array(R, dtype=np.float64) for R in initial_rotations]
        self.p0 = np.array(initial_positions, dtype=np.float64)
        self.v0 = np.array(initial_velocities, dtype=np.float64)
        self.gyro_bias = np.array(gyro_bias, dtype=np.float64)
        self.accel_bias = np.array(accel_bias, dtype=np.float64)
        self.loss_type = loss_type
        self.loss_scale = loss_scale

        self.visual_factors: List[VisualRelativeFactor] = []
        self.gps_factors: List[GPSFactor] = []
        self.imu_factors: List[IMUFactor] = []

    def add_visual_factor(self, factor: VisualRelativeFactor) -> None:
        self.visual_factors.append(factor)

    def add_gps_factor(self, factor: GPSFactor) -> None:
        self.gps_factors.append(factor)

    def add_imu_factor(self, factor: IMUFactor) -> None:
        self.imu_factors.append(factor)

    def _unpack_state_vector(
        self,
        x: np.ndarray
    ) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
        """
        Unpack optimization vector x of length (9 * N) into:
          - R_list: list of N (3, 3) rotation matrices: R_k = R0_k * Exp(delta_theta_k)
          - p: (N, 3) positions: p_k = p0_k + delta_p_k
          - v: (N, 3) velocities: v_k = v0_k + delta_v_k
        """
        R_list = []
        p = np.zeros((self.num_states, 3), dtype=np.float64)
        v = np.zeros((self.num_states, 3), dtype=np.float64)

        for k in range(self.num_states):
            offset = k * 9
            delta_theta = x[offset : offset + 3]
            delta_p = x[offset + 3 : offset + 6]
            delta_v = x[offset + 6 : offset + 9]

            R_k = self.R0_list[k] @ so3_exp(delta_theta)
            R_list.append(R_k)
            p[k] = self.p0[k] + delta_p
            v[k] = self.v0[k] + delta_v

        return R_list, p, v

    def _cost_function(
        self,
        x: np.ndarray,
        lambda_vis: float = 1.0,
        lambda_gps: float = 1.0,
        lambda_imu: float = 1.0
    ) -> np.ndarray:
        """
        Evaluate full residual vector across all active sensor factors.
        """
        R_list, p, v = self._unpack_state_vector(x)
        residuals = []

        # 1. Visual Factors
        if lambda_vis > 0.0 and len(self.visual_factors) > 0:
            w_vis = math.sqrt(lambda_vis)
            for vf in self.visual_factors:
                r_vis = vf.compute_residual(R_list[vf.i], p[vf.i], R_list[vf.j], p[vf.j])
                residuals.append(w_vis * r_vis)

        # 2. GPS Factors
        if lambda_gps > 0.0 and len(self.gps_factors) > 0:
            w_gps = math.sqrt(lambda_gps)
            for gf in self.gps_factors:
                r_gps = gf.compute_residual(p[gf.i])
                residuals.append(w_gps * r_gps)

        # 3. IMU Factors
        if lambda_imu > 0.0 and len(self.imu_factors) > 0:
            w_imu = math.sqrt(lambda_imu)
            for imuf in self.imu_factors:
                r_imu = imuf.compute_residual(
                    R_list[imuf.i], p[imuf.i], v[imuf.i],
                    R_list[imuf.j], p[imuf.j], v[imuf.j]
                )
                residuals.append(w_imu * r_imu)

        if len(residuals) == 0:
            return np.zeros(1, dtype=np.float64)

        return np.concatenate(residuals)

    def _build_sparsity(self) -> np.ndarray:
        """
        Build the Jacobian sparsity pattern matrix (lil_matrix or dense boolean array).
        Rows = residuals, Cols = state variables.
        """
        from scipy.sparse import lil_matrix
        
        num_residuals = len(self.visual_factors) * 6 + len(self.gps_factors) * 3 + len(self.imu_factors) * 9
        sparsity = lil_matrix((num_residuals, self.num_states * 9), dtype=int)
        
        row_idx = 0
        
        # 1. Visual Factors (6 rows each)
        for vf in self.visual_factors:
            idx_i = vf.i * 9
            idx_j = vf.j * 9
            # Visual depends on rotation (0:3) and position (3:6) of i and j
            sparsity[row_idx:row_idx+6, idx_i:idx_i+6] = 1
            sparsity[row_idx:row_idx+6, idx_j:idx_j+6] = 1
            row_idx += 6
            
        # 2. GPS Factors (3 rows each)
        for gf in self.gps_factors:
            idx_i = gf.i * 9
            # GPS depends only on position (3:6) of i
            sparsity[row_idx:row_idx+3, idx_i+3:idx_i+6] = 1
            row_idx += 3
            
        # 3. IMU Factors (9 rows each)
        for imuf in self.imu_factors:
            idx_i = imuf.i * 9
            idx_j = imuf.j * 9
            # IMU depends on all 9 states of i and j
            sparsity[row_idx:row_idx+9, idx_i:idx_i+9] = 1
            sparsity[row_idx:row_idx+9, idx_j:idx_j+9] = 1
            row_idx += 9
            
        return sparsity

    def optimize(
        self,
        max_nfev: int = 50,
        lambda_vis: float = 1.0,
        lambda_gps: float = 1.0,
        lambda_imu: float = 1.0,
        verbose: int = 0
    ) -> Dict[str, Any]:
        """
        Run nonlinear least-squares optimization using scipy.optimize.least_squares.
        Uses sparse numerical Jacobians for massive performance gains.
        """
        x0 = np.zeros(self.num_states * 9, dtype=np.float64)
        t_start = time.perf_counter()

        init_res = self._cost_function(x0, lambda_vis, lambda_gps, lambda_imu)
        init_cost = float(0.5 * np.sum(init_res ** 2))
        
        jac_sparsity = self._build_sparsity()

        opt_res = least_squares(
            fun=self._cost_function,
            x0=x0,
            args=(lambda_vis, lambda_gps, lambda_imu),
            method="trf",
            jac_sparsity=jac_sparsity,
            loss=self.loss_type,
            f_scale=self.loss_scale,
            max_nfev=max_nfev,
            ftol=1e-4,
            xtol=1e-4,
            gtol=1e-4,
            verbose=verbose
        )

        t_end = time.perf_counter()
        runtime_s = t_end - t_start

        # Unpack optimized states
        R_opt, p_opt, v_opt = self._unpack_state_vector(opt_res.x)
        final_cost = float(0.5 * np.sum(opt_res.fun ** 2))

        return {
            "success": bool(opt_res.success),
            "status": int(opt_res.status),
            "message": str(opt_res.message),
            "iterations": int(opt_res.nfev),
            "initial_cost": init_cost,
            "final_cost": final_cost,
            "cost_reduction_percent": round(float((init_cost - final_cost) / max(1e-6, init_cost) * 100.0), 2),
            "runtime_seconds": round(runtime_s, 4),
            "optimized_rotations": R_opt,
            "optimized_positions": p_opt,
            "optimized_velocities": v_opt,
            "gyro_bias": self.gyro_bias,
            "accel_bias": self.accel_bias
        }
