import sys
import os
import csv
import time
import numpy as np
from pathlib import Path
from scipy.sparse import lil_matrix

sys.path.insert(0, str(Path("D:/SIH26158-single-pass-3D").resolve()))
from src.geodesy.projection import wgs84_to_utm32n, utm32n_to_local_enu
from src.pose.imu_frames import frd_to_flu
from src.metrics.alignment import quaternion_to_rotation_matrix
from src.sensor_fusion.imu_types import IMUMeasurement
from src.sensor_fusion.imu_preintegration import preintegrate_imu_measurements
from src.sensor_fusion.sensor_factors import VisualRelativeFactor, GPSFactor, IMUFactor
from src.sensor_fusion.b2_optimizer import B2TrajectoryOptimizer

def debug_b2(N_test=20):
    print(f"--- Running Debug B2 with N={N_test} ---")
    b0_poses_path = Path("outputs/reports/zurich_mav/b0/camera_poses_colmap.csv")
    gps_path = Path("outputs/reports/zurich_mav/gps.csv")
    imu_path = Path("outputs/reports/zurich_mav/imu.csv")
    images_path = Path("outputs/reports/zurich_mav/images.csv")
    
    images = []
    with open(images_path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            images.append({"imgid": int(r["imgid"]), "ts": float(r["timestamp_seconds"])})
    images = images[:N_test]
    
    colmap_poses = {}
    with open(b0_poses_path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["registered"].lower() == "true":
                colmap_poses[int(r["imgid"])] = {
                    "c_w": np.array([float(r["camera_center_x"]), float(r["camera_center_y"]), float(r["camera_center_z"])]),
                    "q_wc": np.array([float(r["q_wc_x"]), float(r["q_wc_y"]), float(r["q_wc_z"]), float(r["q_wc_w"])])
                }
                
    all_gps = []
    with open(gps_path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            all_gps.append({
                "ts": float(r["timestamp_seconds"]),
                "lat": float(r["latitude"]),
                "lon": float(r["longitude"]),
                "alt": float(r["altitude_if_available"]) if r.get("altitude_if_available") else 465.0
            })
    
    gps_ts = np.array([g["ts"] for g in all_gps])
    gps_utm = [wgs84_to_utm32n(g["lat"], g["lon"], g["alt"]) for g in all_gps]
    e0, n0, u0 = gps_utm[0]
    gps_enu = np.array([utm32n_to_local_enu(e, n, u, e0, n0, u0) for e, n, u in gps_utm])
    
    imu_meas_list = []
    with open(imu_path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            a_flu, w_flu = frd_to_flu(
                np.array([float(r["accel_x"]), float(r["accel_y"]), float(r["accel_z"])]),
                np.array([float(r["gyro_x"]), float(r["gyro_y"]), float(r["gyro_z"])])
            )
            imu_meas_list.append(IMUMeasurement(float(r["timestamp_seconds"]), a_flu, w_flu))
            
    imu_ts = np.array([m.timestamp_seconds for m in imu_meas_list])
    imu_a = np.array([m.accel for m in imu_meas_list])
    imu_w = np.array([m.gyro for m in imu_meas_list])
    
    def get_imu_at(t: float) -> IMUMeasurement:
        a = np.array([np.interp(t, imu_ts, imu_a[:, 0]), np.interp(t, imu_ts, imu_a[:, 1]), np.interp(t, imu_ts, imu_a[:, 2])])
        w = np.array([np.interp(t, imu_ts, imu_w[:, 0]), np.interp(t, imu_ts, imu_w[:, 1]), np.interp(t, imu_ts, imu_w[:, 2])])
        return IMUMeasurement(t, a, w)

    timestamps = np.array([img["ts"] for img in images])
    init_rotations = [quaternion_to_rotation_matrix(colmap_poses[img["imgid"]]["q_wc"]) for img in images]
    
    # Interpolate GPS safely
    init_positions = np.zeros((N_test, 3))
    for k in range(N_test):
        t_img = timestamps[k]
        init_positions[k, 0] = np.interp(t_img, gps_ts, gps_enu[:, 0])
        init_positions[k, 1] = np.interp(t_img, gps_ts, gps_enu[:, 1])
        init_positions[k, 2] = np.interp(t_img, gps_ts, gps_enu[:, 2])
        
    init_velocities = np.zeros((N_test, 3))
    for k in range(N_test - 1):
        dt = timestamps[k + 1] - timestamps[k]
        init_velocities[k] = (init_positions[k + 1] - init_positions[k]) / max(1e-4, dt)
    init_velocities[-1] = init_velocities[-2]
    
    gyro_bias = np.array([0.0113, -(-0.0397), -(-0.0245)]) # FLU
    accel_bias = np.array([-0.1638, -(-0.1654), -(-0.6153)]) # FLU
    
    opt = B2TrajectoryOptimizer(N_test, init_rotations, init_positions, init_velocities, gyro_bias, accel_bias)
    
    s_b1 = 0.14082986 # Hardcoded for debug
    
    for k in range(N_test - 1):
        id_i, id_j = images[k]["imgid"], images[k+1]["imgid"]
        Ri, Rj = init_rotations[k], init_rotations[k+1]
        R_ij_meas = Ri.T @ Rj
        ci, cj = colmap_poses[id_i]["c_w"], colmap_poses[id_j]["c_w"]
        t_ij_metric = s_b1 * (Ri.T @ (cj - ci))
        opt.add_visual_factor(VisualRelativeFactor(k, k+1, R_ij_meas, t_ij_metric))
        
    for k in range(N_test):
        opt.add_gps_factor(GPSFactor(k, init_positions[k], sigma_gps=0.5))
        
    for k in range(N_test - 1):
        t_i, t_j = timestamps[k], timestamps[k+1]
        sub_imu = [get_imu_at(t_i)]
        sub_imu.extend([m for m in imu_meas_list if t_i < m.timestamp_seconds < t_j])
        sub_imu.append(get_imu_at(t_j))
        preint = preintegrate_imu_measurements(sub_imu, accel_bias=accel_bias, gyro_bias=gyro_bias)
        opt.add_imu_factor(IMUFactor(k, k+1, preint))

    print(f"Factors: {len(opt.visual_factors)} Vis, {len(opt.gps_factors)} GPS, {len(opt.imu_factors)} IMU")
    
    start_time = time.time()
    res = opt.optimize(max_nfev=20, lambda_vis=1.0, lambda_gps=1.0, lambda_imu=1.0)
    end_time = time.time()
    
    print(f"Debug Opt N={N_test} finished in {end_time - start_time:.2f}s")
    print(f"Status: {res['status']}, Iterations: {res['iterations']}")
    print(f"Cost: {res['initial_cost']:.2f} -> {res['final_cost']:.2f}")

if __name__ == '__main__':
    debug_b2(20)
    debug_b2(50)
    debug_b2(100)
