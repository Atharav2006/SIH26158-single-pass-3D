import os
import json
import re
import cv2
from pathlib import Path
from typing import Dict, Any, List
from src.ingestion.datasets.base import BaseDatasetAdapter
from src.ingestion.synchronization import TemporalSynchronizer
from src.pose.association import associate_groundtruth_by_imgid, AssociationMethod
from src.pose.models import Pose, Position, Quaternion

class DatasetValidator:
    """
    Validates integrity, schema correctness, readability, monotonicity, multi-sensor synchronization,
    and image identity / ground-truth association for dataset adapters.
    """
    def __init__(self, adapter: BaseDatasetAdapter):
        self.adapter = adapter

    def validate(self) -> Dict[str, Any]:
        """
        Run full validation suite and return structured validation report.
        """
        report: Dict[str, Any] = {
            "dataset_name": self.adapter.dataset_info.get("name", "Unknown"),
            "status": "PASS",
            "checks": {},
            "issues": []
        }

        # 1. Images Check & Identity Check
        images = self.adapter.images
        img_check = {
            "total_images": len(images),
            "readable_images": 0,
            "missing_files": 0,
            "corrupted_files": 0,
            "duplicate_timestamps": 0,
            "duplicate_imgids": 0,
            "non_integer_imgids": 0,
            "filename_id_mismatches": 0,
            "monotonic": True
        }

        ts_set = set()
        imgid_set = set()
        prev_ts = -1.0
        for img in images:
            ts = img.get("timestamp_seconds")
            if ts in ts_set:
                img_check["duplicate_timestamps"] += 1
            ts_set.add(ts)

            if ts is not None:
                if ts < prev_ts:
                    img_check["monotonic"] = False
                prev_ts = ts

            # imgid validation
            native_id = img.get("imgid")
            if native_id is None or not isinstance(native_id, int):
                img_check["non_integer_imgids"] += 1
            elif native_id in imgid_set:
                img_check["duplicate_imgids"] += 1
            if native_id is not None:
                imgid_set.add(native_id)

            filename = img.get("filename", "")
            match = re.search(r'(\d+)$', Path(filename).stem)
            if match:
                derived_id = int(match.group(1))
                if native_id is not None and derived_id != native_id:
                    img_check["filename_id_mismatches"] += 1

            # Image existence & readability
            img_path = self.adapter.actual_root / "MAV Images" / filename
            if not img_path.exists():
                img_path = self.adapter.actual_root / "MAV Images Calib" / filename

            if not img_path.exists():
                img_check["missing_files"] += 1
            else:
                mat = cv2.imread(str(img_path))
                if mat is None:
                    img_check["corrupted_files"] += 1
                else:
                    img_check["readable_images"] += 1

        report["checks"]["images"] = img_check

        # 2. GPS Check
        gps = self.adapter.gps
        gps_check = {
            "total_records": len(gps),
            "valid_coordinates": 0,
            "invalid_coordinates": 0,
            "monotonic": True
        }
        prev_ts = -1.0
        for row in gps:
            ts = row.get("timestamp_seconds", 0.0)
            if ts < prev_ts:
                gps_check["monotonic"] = False
            prev_ts = ts

            lat = row.get("latitude")
            lon = row.get("longitude")
            if lat is not None and lon is not None and -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0:
                gps_check["valid_coordinates"] += 1
            else:
                gps_check["invalid_coordinates"] += 1

        report["checks"]["gps"] = gps_check

        # 3. IMU Check
        imu = self.adapter.imu
        imu_check = {
            "total_records": len(imu),
            "valid_records": 0,
            "monotonic": True
        }
        prev_ts = -1.0
        for row in imu:
            ts = row.get("timestamp_seconds", 0.0)
            if ts < prev_ts:
                imu_check["monotonic"] = False
            prev_ts = ts
            imu_check["valid_records"] += 1
        report["checks"]["imu"] = imu_check

        # 4. Pose Check & Ground Truth Identity Check
        pose = self.adapter.pose
        pose_check = {
            "total_records": len(pose),
            "valid_quaternions": 0,
            "duplicate_imgids": 0,
            "monotonic": True
        }
        prev_ts = -1.0
        pose_imgid_set = set()
        for row in pose:
            ts = row.get("timestamp_seconds", 0.0)
            if ts < prev_ts:
                pose_check["monotonic"] = False
            prev_ts = ts

            p_imgid = row.get("imgid")
            if p_imgid is not None:
                if p_imgid in pose_imgid_set:
                    pose_check["duplicate_imgids"] += 1
                pose_imgid_set.add(p_imgid)

            qx = row.get("qx", 0.0)
            qy = row.get("qy", 0.0)
            qz = row.get("qz", 0.0)
            qw = row.get("qw", 1.0)
            norm = (qx*qx + qy*qy + qz*qz + qw*qw)**0.5
            if abs(norm - 1.0) < 0.05:
                pose_check["valid_quaternions"] += 1

        report["checks"]["pose"] = pose_check

        # 5. Camera Calibration Check
        cam = self.adapter.camera or {}
        cam_check = {
            "has_intrinsics": bool(cam.get("fx") and cam.get("fy") and cam.get("cx") and cam.get("cy")),
            "has_distortion": bool(cam.get("distortion_parameters_if_available")),
            "has_resolution": bool(cam.get("image_width") and cam.get("image_height"))
        }
        report["checks"]["camera_calibration"] = cam_check

        # 6. Authoritative Ground-Truth Association & Telemetry Synchronization
        sync_stats = {}
        if images and gps:
            synced_gps = TemporalSynchronizer.synchronize(images, gps, max_tolerance=0.05)
            dt_list = [item["time_difference"] for item in synced_gps]
            sync_stats["image_to_gps"] = {
                "matched_count": len(synced_gps),
                "match_rate": round(len(synced_gps) / len(images), 4) if images else 0,
                "mean_time_diff_sec": round(sum(dt_list) / len(dt_list), 6) if dt_list else 0.0,
                "max_time_diff_sec": round(max(dt_list), 6) if dt_list else 0.0
            }

        if images and imu:
            synced_imu = TemporalSynchronizer.synchronize(images, imu, max_tolerance=0.1)
            dt_list = [item["time_difference"] for item in synced_imu]
            sync_stats["image_to_imu"] = {
                "matched_count": len(synced_imu),
                "match_rate": round(len(synced_imu) / len(images), 4) if images else 0,
                "mean_time_diff_sec": round(sum(dt_list) / len(dt_list), 6) if dt_list else 0.0,
                "max_time_diff_sec": round(max(dt_list), 6) if dt_list else 0.0
            }

        if images and pose:
            pose_objs = [
                Pose(
                    timestamp_seconds=p.get("timestamp_seconds", 0.0),
                    position_xyz=Position(p.get("tx", 0.0), p.get("ty", 0.0), p.get("tz", 0.0)),
                    orientation_xyzw=Quaternion(p.get("qx", 0.0), p.get("qy", 0.0), p.get("qz", 0.0), p.get("qw", 1.0)),
                    imgid=p.get("imgid")
                )
                for p in pose
            ]
            exact_assocs = associate_groundtruth_by_imgid(images, pose_objs)
            matched_exact = [a for a in exact_assocs if a.matched]
            sync_stats["exact_ground_truth_association"] = {
                "total_images": len(images),
                "exact_keyframe_matches": len(matched_exact),
                "unmatched_intermediate_frames": len(images) - len(matched_exact),
                "exact_keyframe_match_rate": round(len(matched_exact) / len(images), 4) if images else 0.0,
                "association_method": AssociationMethod.EXACT_ID.value
            }

        report["checks"]["synchronization"] = sync_stats

        # Overall Status
        if (img_check["corrupted_files"] > 0 or 
            img_check["duplicate_imgids"] > 0 or 
            img_check["filename_id_mismatches"] > 0 or 
            pose_check["duplicate_imgids"] > 0 or 
            gps_check["invalid_coordinates"] > 0 or 
            not cam_check["has_intrinsics"]):
            report["status"] = "FAIL"

        return report
