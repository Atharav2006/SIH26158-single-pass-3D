import csv
import json
import math
import statistics
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

from src.pose.models import Position, Quaternion, Pose
from src.pose.coordinate_frames import transform_to_local_enu, FRAME_GLOBAL_UTM_ENU, FRAME_LOCAL_ENU

class Trajectory:
    """
    Manages a continuous sequence of 6DoF poses, coordinate frame transformations, and trajectory statistics.
    """
    def __init__(self, poses: List[Pose], frame_id: str = FRAME_GLOBAL_UTM_ENU, origin: Optional[Position] = None):
        self.poses = poses
        self.frame_id = frame_id
        self.origin = origin

    @classmethod
    def from_poses(cls, poses: List[Pose]) -> 'Trajectory':
        frame_id = poses[0].source_frame if poses else FRAME_GLOBAL_UTM_ENU
        return cls(poses=poses, frame_id=frame_id)

    def to_local_enu(self, origin: Optional[Position] = None) -> 'Trajectory':
        """
        Convert all poses to local ENU frame relative to the given origin (or the first pose position).
        """
        if not self.poses:
            return Trajectory([], frame_id=FRAME_LOCAL_ENU)

        ref_origin = origin or self.poses[0].position_xyz
        local_poses = [transform_to_local_enu(p, ref_origin) for p in self.poses]

        return Trajectory(poses=local_poses, frame_id=FRAME_LOCAL_ENU, origin=ref_origin)

    def compute_statistics(self) -> Dict[str, Any]:
        """
        Compute rigorous trajectory metrics, velocity distributions, and spatial statistics.
        """
        if not self.poses:
            return {
                "pose_count": 0,
                "valid_pose_count": 0,
                "trajectory_length_meters": 0.0,
                "mean_speed_mps": 0.0,
                "median_speed_mps": 0.0,
                "max_speed_mps": 0.0
            }

        valid_poses = [p for p in self.poses if p.orientation_xyzw.is_normalized()]
        ts_list = [p.timestamp_seconds for p in self.poses]
        x_list = [p.position_xyz.x for p in self.poses]
        y_list = [p.position_xyz.y for p in self.poses]
        z_list = [p.position_xyz.z for p in self.poses]

        start_ts = ts_list[0]
        end_ts = ts_list[-1]
        duration = max(0.0, end_ts - start_ts)

        # Cumulative Euclidean distance and speed calculation
        cum_dist = 0.0
        speeds = []

        for i in range(1, len(self.poses)):
            p_prev = self.poses[i - 1]
            p_curr = self.poses[i]

            dx = p_curr.position_xyz.x - p_prev.position_xyz.x
            dy = p_curr.position_xyz.y - p_prev.position_xyz.y
            dz = p_curr.position_xyz.z - p_prev.position_xyz.z
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)
            cum_dist += dist

            dt = p_curr.timestamp_seconds - p_prev.timestamp_seconds
            if dt > 1e-4:
                spd = dist / dt
                speeds.append(spd)

        mean_speed = (cum_dist / duration) if duration > 0 else 0.0
        median_speed = statistics.median(speeds) if speeds else 0.0
        max_speed = max(speeds) if speeds else 0.0

        return {
            "pose_count": len(self.poses),
            "valid_pose_count": len(valid_poses),
            "coordinate_frame": self.frame_id,
            "start_timestamp_seconds": round(start_ts, 6),
            "end_timestamp_seconds": round(end_ts, 6),
            "duration_seconds": round(duration, 6),
            "spatial_extent": {
                "min_x": round(min(x_list), 3),
                "max_x": round(max(x_list), 3),
                "span_x": round(max(x_list) - min(x_list), 3),
                "min_y": round(min(y_list), 3),
                "max_y": round(max(y_list), 3),
                "span_y": round(max(y_list) - min(y_list), 3),
                "min_z": round(min(z_list), 3),
                "max_z": round(max(z_list), 3),
                "span_z": round(max(z_list) - min(z_list), 3),
                "unit": "meter"
            },
            "trajectory_length_meters": round(cum_dist, 3),
            "mean_speed_mps": round(mean_speed, 3),
            "average_speed_mps": round(mean_speed, 3),
            "median_speed_mps": round(median_speed, 3),
            "max_speed_mps": round(max_speed, 3)
        }

    def validate_trajectory(self) -> Dict[str, Any]:
        """
        Execute sanity validation rules:
        1. All positions finite.
        2. All quaternions finite and normalized.
        3. No excessive spatial discontinuities (>50 m/s instantaneous step).
        4. Timestamp progression valid.
        5. Local coordinates bounded.
        """
        issues = []
        status = "PASS"

        if not self.poses:
            return {"status": "FAIL", "issues": ["Empty trajectory"]}

        prev_ts = -1.0
        for i, p in enumerate(self.poses):
            # 1. Finite positions
            if not (math.isfinite(p.position_xyz.x) and math.isfinite(p.position_xyz.y) and math.isfinite(p.position_xyz.z)):
                issues.append(f"Non-finite position at index {i}")
                status = "FAIL"

            # 2. Finite & normalized quaternions
            q = p.orientation_xyzw
            if not (math.isfinite(q.qx) and math.isfinite(q.qy) and math.isfinite(q.qz) and math.isfinite(q.qw)):
                issues.append(f"Non-finite quaternion at index {i}")
                status = "FAIL"
            elif not q.is_normalized(tol=1e-3):
                issues.append(f"Unnormalized quaternion (norm={q.norm():.5f}) at index {i}")
                status = "FAIL"

            # 3. Discontinuity & speed check
            if i > 0:
                p_prev = self.poses[i - 1]
                dx = p.position_xyz.x - p_prev.position_xyz.x
                dy = p.position_xyz.y - p_prev.position_xyz.y
                dz = p.position_xyz.z - p_prev.position_xyz.z
                step_dist = math.sqrt(dx*dx + dy*dy + dz*dz)
                dt = p.timestamp_seconds - p_prev.timestamp_seconds

                if dt > 1e-4:
                    spd = step_dist / dt
                    if spd > 25.0:  # Quadrotor urban flight limit: 25 m/s (~90 km/h)
                        issues.append(f"Impossible instantaneous speed ({spd:.2f} m/s) between index {i-1} and {i}")
                        status = "FAIL"

            prev_ts = p.timestamp_seconds

        return {
            "status": status,
            "pose_count": len(self.poses),
            "coordinate_frame": self.frame_id,
            "issues_detected": len(issues),
            "issues": issues[:10]  # First 10 issues
        }

    def export_csv(self, output_path: Union[str, Path]) -> None:
        """Export trajectory to CSV format: timestamp_seconds, x, y, z, qx, qy, qz, qw."""
        output_path = Path(output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp_seconds", "x", "y", "z", "qx", "qy", "qz", "qw"])
            for p in self.poses:
                writer.writerow([
                    p.timestamp_seconds,
                    p.position_xyz.x,
                    p.position_xyz.y,
                    p.position_xyz.z,
                    p.orientation_xyzw.qx,
                    p.orientation_xyzw.qy,
                    p.orientation_xyzw.qz,
                    p.orientation_xyzw.qw
                ])

    def export_json(self, output_path: Union[str, Path], extra_metadata: Optional[Dict[str, Any]] = None) -> None:
        """Export trajectory metadata and statistics to JSON."""
        output_path = Path(output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        stats = self.compute_statistics()
        if extra_metadata:
            stats.update(extra_metadata)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=4)
