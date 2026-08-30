import os
import csv
import math
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from src.ingestion.datasets.base import BaseDatasetAdapter

def euler_to_quaternion(roll_deg: float, pitch_deg: float, yaw_deg: float):
    """Convert Euler angles in degrees (roll/omega, pitch/phi, yaw/kappa) to unit quaternion (qx, qy, qz, qw)."""
    r = math.radians(roll_deg)
    p = math.radians(pitch_deg)
    y = math.radians(yaw_deg)

    cy = math.cos(y * 0.5)
    sy = math.sin(y * 0.5)
    cp = math.cos(p * 0.5)
    sp = math.sin(p * 0.5)
    cr = math.cos(r * 0.5)
    sr = math.sin(r * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    return qx, qy, qz, qw

class ZurichMAVAdapter(BaseDatasetAdapter):
    """
    Adapter for the Zurich Urban Micro Aerial Vehicle (Air-Ground Zurich AGZ) dataset.
    Parses onboard GPS, raw IMU, onboard/ground-truth pose, camera calibration, and image sequences.
    """
    def __init__(self, dataset_root: Union[str, Path]):
        super().__init__(dataset_root)
        self.actual_root = self.dataset_root
        self._find_actual_root()

        self.dataset_info = {
            "name": "Zurich Urban MAV Dataset (AGZ)",
            "type": "UAV Urban Aerial Flight",
            "provider": "Robotics and Perception Group (RPG), University of Zurich",
            "url": "http://rpg.ifi.uzh.ch/zurichmavdataset.html",
            "license": "Open / Unrestricted Research & Commercial"
        }

    def _find_actual_root(self):
        """Locate directory containing 'Log Files' and 'calibration_data.npz'."""
        if (self.dataset_root / "Log Files").is_dir() and (self.dataset_root / "calibration_data.npz").is_file():
            self.actual_root = self.dataset_root
        elif (self.dataset_root / "AGZ_subset" / "Log Files").is_dir():
            self.actual_root = self.dataset_root / "AGZ_subset"

    def validate_root(self) -> bool:
        """Validate presence of critical files."""
        self._find_actual_root()
        log_dir = self.actual_root / "Log Files"
        calib_file = self.actual_root / "calibration_data.npz"

        if not log_dir.is_dir():
            raise FileNotFoundError(f"Missing 'Log Files' directory in {self.actual_root}")
        if not calib_file.is_file():
            raise FileNotFoundError(f"Missing 'calibration_data.npz' in {self.actual_root}")

        return True

    def parse(self) -> None:
        """Parse all components of the Zurich MAV dataset."""
        self.validate_root()
        self.parse_camera_calibration()
        self.parse_gps()
        self.parse_imu()
        self.parse_pose()
        self.parse_images()

    def parse_camera_calibration(self) -> None:
        """Parse intrinsic matrix and distortion coefficients from calibration_data.npz."""
        calib_path = self.actual_root / "calibration_data.npz"
        calib = np.load(calib_path)
        
        K = calib["intrinsic_matrix"]
        dist = calib["distCoeff"]

        self.camera = {
            "model": "pinhole_radial_tangential",
            "fx": float(K[0, 0]),
            "fy": float(K[1, 1]),
            "cx": float(K[0, 2]),
            "cy": float(K[1, 2]),
            "distortion_parameters_if_available": [float(v) for v in dist.flatten()],
            "image_width": 1920,
            "image_height": 1080
        }

    def parse_gps(self) -> None:
        """Parse OnboardGPS.csv."""
        gps_file = self.actual_root / "Log Files" / "OnboardGPS.csv"
        if not gps_file.is_file():
            return

        self.gps = []
        with open(gps_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = [h.strip() for h in next(reader)]
            # Locate column indices
            ts_idx = header.index("Timpstemp")
            lat_idx = header.index("lat")
            lon_idx = header.index("lon")
            alt_idx = header.index("alt") if "alt" in header else -1

            for row in reader:
                if not row or not row[0].strip():
                    continue
                try:
                    ts_us = float(row[ts_idx].strip())
                    lat = float(row[lat_idx].strip())
                    lon = float(row[lon_idx].strip())
                    alt = float(row[alt_idx].strip()) if alt_idx != -1 and row[alt_idx].strip() else None

                    self.gps.append({
                        "timestamp_seconds": round(ts_us / 1e6, 6),
                        "latitude": lat,
                        "longitude": lon,
                        "altitude_if_available": alt
                    })
                except (ValueError, IndexError) as e:
                    raise ValueError(f"Malformed row in {gps_file.name}: {row} (error: {e})")

    def parse_imu(self) -> None:
        """Parse IMU measurements from RawAccel.csv and RawGyro.csv."""
        accel_file = self.actual_root / "Log Files" / "RawAccel.csv"
        gyro_file = self.actual_root / "Log Files" / "RawGyro.csv"

        if not accel_file.is_file() or not gyro_file.is_file():
            return

        # Load Gyro dictionary keyed by timestamp_us
        gyro_dict = {}
        with open(gyro_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = [h.strip() for h in next(reader)]
            ts_idx = header.index("Timpstemp")
            gx_idx = header.index("x")
            gy_idx = header.index("y")
            gz_idx = header.index("z")

            for row in reader:
                if not row or not row[0].strip():
                    continue
                try:
                    ts_us = int(float(row[ts_idx].strip()))
                    gyro_dict[ts_us] = (
                        float(row[gx_idx].strip()),
                        float(row[gy_idx].strip()),
                        float(row[gz_idx].strip())
                    )
                except (ValueError, IndexError) as e:
                    raise ValueError(f"Malformed row in {gyro_file.name}: {row} (error: {e})")

        self.imu = []
        with open(accel_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = [h.strip() for h in next(reader)]
            ts_idx = header.index("Timpstemp")
            ax_idx = header.index("x")
            ay_idx = header.index("y")
            az_idx = header.index("z")

            for row in reader:
                if not row or not row[0].strip():
                    continue
                try:
                    ts_us = int(float(row[ts_idx].strip()))
                    ax = float(row[ax_idx].strip())
                    ay = float(row[ay_idx].strip())
                    az = float(row[az_idx].strip())

                    gx, gy, gz = gyro_dict.get(ts_us, (0.0, 0.0, 0.0))

                    self.imu.append({
                        "timestamp_seconds": round(ts_us / 1e6, 6),
                        "accel_x": ax,
                        "accel_y": ay,
                        "accel_z": az,
                        "gyro_x": gx,
                        "gyro_y": gy,
                        "gyro_z": gz
                    })
                except (ValueError, IndexError) as e:
                    raise ValueError(f"Malformed row in {accel_file.name}: {row} (error: {e})")

    def parse_pose(self) -> None:
        """Parse GroundTruthAGL.csv (Ground Truth 6DoF) or OnboardPose.csv."""
        gt_file = self.actual_root / "Log Files" / "GroundTruthAGL.csv"
        onboard_pose_file = self.actual_root / "Log Files" / "OnboardPose.csv"

        self.pose = []

        # First priority: GroundTruthAGL.csv
        if gt_file.is_file():
            # Build imgid to timestamp map from GPS
            imgid_ts_map = {}
            gps_file = self.actual_root / "Log Files" / "OnboardGPS.csv"
            if gps_file.is_file():
                with open(gps_file, "r", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    header = [h.strip() for h in next(reader)]
                    ts_idx = header.index("Timpstemp")
                    imgid_idx = header.index("imgid")
                    for row in reader:
                        if row and row[0].strip():
                            try:
                                imgid = int(float(row[imgid_idx].strip()))
                                ts_s = round(float(row[ts_idx].strip()) / 1e6, 6)
                                imgid_ts_map[imgid] = ts_s
                            except (ValueError, IndexError):
                                pass

            with open(gt_file, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = [h.strip() for h in next(reader)]
                imgid_idx = header.index("imgid")
                x_idx = header.index("x_gt")
                y_idx = header.index("y_gt")
                z_idx = header.index("z_gt")
                om_idx = header.index("omega_gt")
                phi_idx = header.index("phi_gt")
                kap_idx = header.index("kappa_gt")

                for row in reader:
                    if not row or not row[0].strip():
                        continue
                    try:
                        imgid = int(float(row[imgid_idx].strip()))
                        ts = imgid_ts_map.get(imgid, round(imgid / 30.0, 6))
                        tx = float(row[x_idx].strip())
                        ty = float(row[y_idx].strip())
                        tz = float(row[z_idx].strip())
                        om = float(row[om_idx].strip())
                        phi = float(row[phi_idx].strip())
                        kap = float(row[kap_idx].strip())

                        qx, qy, qz, qw = euler_to_quaternion(om, phi, kap)

                        self.pose.append({
                            "imgid": imgid,
                            "timestamp_seconds": ts,
                            "tx": tx,
                            "ty": ty,
                            "tz": tz,
                            "qx": qx,
                            "qy": qy,
                            "qz": qz,
                            "qw": qw
                        })
                    except (ValueError, IndexError) as e:
                        raise ValueError(f"Malformed row in {gt_file.name}: {row} (error: {e})")

        elif onboard_pose_file.is_file():
            with open(onboard_pose_file, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = [h.strip() for h in next(reader)]
                ts_idx = header.index("Timpstemp")
                qw_idx = header.index("Attitude_w")
                qx_idx = header.index("Attitude_x")
                qy_idx = header.index("Attitude_y")
                qz_idx = header.index("Attitude_z")
                alt_idx = header.index("Altitude")

                for row in reader:
                    if not row or not row[0].strip():
                        continue
                    try:
                        ts_us = float(row[ts_idx].strip())
                        qw = float(row[qw_idx].strip())
                        qx = float(row[qx_idx].strip())
                        qy = float(row[qy_idx].strip())
                        qz = float(row[qz_idx].strip())
                        alt = float(row[alt_idx].strip())

                        self.pose.append({
                            "imgid": None,
                            "timestamp_seconds": round(ts_us / 1e6, 6),
                            "tx": 0.0,
                            "ty": 0.0,
                            "tz": alt,
                            "qx": qx,
                            "qy": qy,
                            "qz": qz,
                            "qw": qw
                        })
                    except (ValueError, IndexError) as e:
                        raise ValueError(f"Malformed row in {onboard_pose_file.name}: {row} (error: {e})")

    def parse_images(self) -> None:
        """Discover image files from MAV Images or MAV Images Calib directories."""
        import re
        self.images = []
        image_dirs = [self.actual_root / "MAV Images", self.actual_root / "MAV Images Calib"]
        
        discovered_files = []
        for d in image_dirs:
            if d.is_dir():
                found = sorted(list(d.glob("*.png")) + list(d.glob("*.jpg")) + list(d.glob("*.jpeg")))
                if found:
                    discovered_files = found
                    break

        # Map GPS timestamps by imgid if available
        imgid_ts_map = {}
        gps_file = self.actual_root / "Log Files" / "OnboardGPS.csv"
        if gps_file.is_file():
            with open(gps_file, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = [h.strip() for h in next(reader)]
                ts_idx = header.index("Timpstemp")
                imgid_idx = header.index("imgid")
                for row in reader:
                    if row and row[0].strip():
                        try:
                            imgid = int(float(row[imgid_idx].strip()))
                            ts_s = round(float(row[ts_idx].strip()) / 1e6, 6)
                            imgid_ts_map[imgid] = ts_s
                        except (ValueError, IndexError):
                            pass

        for idx, img_path in enumerate(discovered_files, start=1):
            # Extract dataset-native imgid from filename (e.g. '00001.jpg' -> 1)
            match = re.search(r'(\d+)$', img_path.stem)
            native_imgid = int(match.group(1)) if match else idx

            ts = imgid_ts_map.get(native_imgid, round((native_imgid - 1) / 30.0, 6))
            self.images.append({
                "image_id": idx,
                "imgid": native_imgid,
                "filename": img_path.name,
                "timestamp_seconds": ts,
                "width": 1920,
                "height": 1080
            })
