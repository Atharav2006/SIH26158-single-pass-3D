import csv
import math
from pathlib import Path
from typing import List, Dict, Any, Union, Optional

from src.pose.models import Position, Quaternion, Pose
from src.pose.coordinate_frames import FRAME_GLOBAL_UTM_ENU, FRAME_CAMERA_RDF
from src.ingestion.synchronization import TemporalSynchronizer
from src.pose.association import (
    AssociationMethod,
    GroundTruthAssociation,
    associate_groundtruth_by_imgid,
    export_image_groundtruth_associations_csv
)

def load_poses_from_csv(
    csv_path: Union[str, Path],
    source_frame: str = FRAME_GLOBAL_UTM_ENU,
    target_frame: str = FRAME_CAMERA_RDF,
    pose_semantics: str = "camera_optical_center_in_world"
) -> List[Pose]:
    """
    Load 6DoF poses from a standardized pose.csv file.
    
    Expected CSV columns:
      timestamp_seconds, tx, ty, tz, qx, qy, qz, qw (and optionally imgid)
    """
    csv_path = Path(csv_path).resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"Pose CSV file not found: {csv_path}")

    poses: List[Pose] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader, start=1):
            try:
                ts = float(row["timestamp_seconds"])
                tx = float(row["tx"])
                ty = float(row["ty"])
                tz = float(row["tz"])
                qx = float(row["qx"])
                qy = float(row["qy"])
                qz = float(row["qz"])
                qw = float(row["qw"])

                imgid_val = None
                if "imgid" in row and row["imgid"] and row["imgid"].strip():
                    imgid_val = int(float(row["imgid"].strip()))
                elif "img_id" in row and row["img_id"] and row["img_id"].strip():
                    imgid_val = int(float(row["img_id"].strip()))

                # Sanity validation: Finite numbers
                if not (math.isfinite(tx) and math.isfinite(ty) and math.isfinite(tz)):
                    raise ValueError(f"Non-finite position values at row {row_idx}: {tx}, {ty}, {tz}")
                if not (math.isfinite(qx) and math.isfinite(qy) and math.isfinite(qz) and math.isfinite(qw)):
                    raise ValueError(f"Non-finite quaternion values at row {row_idx}: {qx}, {qy}, {qz}, {qw}")

                pos = Position(x=tx, y=ty, z=tz, unit="meter")
                raw_quat = Quaternion(qx=qx, qy=qy, qz=qz, qw=qw, convention="Hamilton")
                
                # Check and enforce quaternion normalization
                norm = raw_quat.norm()
                if abs(norm - 1.0) > 1e-2:
                    raise ValueError(f"Invalid quaternion norm ({norm}) at row {row_idx}")
                
                norm_quat = raw_quat.normalized()

                pose = Pose(
                    timestamp_seconds=ts,
                    position_xyz=pos,
                    orientation_xyzw=norm_quat,
                    source_frame=source_frame,
                    target_frame=target_frame,
                    pose_semantics=pose_semantics,
                    imgid=imgid_val
                )
                poses.append(pose)
            except (KeyError, ValueError) as e:
                raise ValueError(f"Malformed row in {csv_path.name} line {row_idx}: {row} (error: {e})")

    return poses

def load_image_metadata(images_csv_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Load image records from an images.csv file.
    Preserves image_id, imgid, filename, timestamp_seconds, width, height.
    """
    images_csv_path = Path(images_csv_path).resolve()
    if not images_csv_path.exists():
        raise FileNotFoundError(f"Images CSV file not found: {images_csv_path}")

    images: List[Dict[str, Any]] = []
    with open(images_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                img_id = int(row["image_id"])
                native_imgid = int(row["imgid"]) if "imgid" in row and row["imgid"].strip() else img_id
                images.append({
                    "image_id": img_id,
                    "imgid": native_imgid,
                    "filename": str(row["filename"]),
                    "timestamp_seconds": float(row["timestamp_seconds"]),
                    "width": int(row["width"]),
                    "height": int(row["height"])
                })
            except (KeyError, ValueError) as e:
                raise ValueError(f"Malformed row in {images_csv_path.name}: {row} (error: {e})")

    return images

def associate_poses_to_images(
    poses: List[Pose],
    images: List[Dict[str, Any]],
    max_tolerance: float = 0.05
) -> List[Dict[str, Any]]:
    """
    Associate each image timestamp with the nearest telemetry/pose timestamp.
    Used for asynchronous telemetry streams (GPS, IMU, etc.).
    """
    pose_records = [
        {
            "timestamp_seconds": p.timestamp_seconds,
            "pose_index": idx,
            "pose": p
        }
        for idx, p in enumerate(poses)
    ]

    associations: List[Dict[str, Any]] = []
    for img in images:
        target_ts = img["timestamp_seconds"]
        match_info = TemporalSynchronizer.find_nearest(
            target_ts=target_ts,
            stream=pose_records,
            timestamp_key="timestamp_seconds",
            max_tolerance=max_tolerance
        )

        if match_info is not None:
            associations.append({
                "image_id": img["image_id"],
                "imgid": img.get("imgid", img["image_id"]),
                "filename": img["filename"],
                "image_timestamp": target_ts,
                "pose_timestamp": match_info["matched_timestamp"],
                "matched_pose_timestamp": match_info["matched_timestamp"],
                "absolute_delta_seconds": match_info["time_difference"],
                "time_difference": match_info["time_difference"],
                "association_method": AssociationMethod.TIMESTAMP_NEAREST.value,
                "matched": True,
                "pose_index": match_info["record"]["pose_index"],
                "pose": match_info["record"]["pose"]
            })
        else:
            # Record unmatched association explicitly
            associations.append({
                "image_id": img["image_id"],
                "imgid": img.get("imgid", img["image_id"]),
                "filename": img["filename"],
                "image_timestamp": target_ts,
                "pose_timestamp": None,
                "matched_pose_timestamp": None,
                "absolute_delta_seconds": None,
                "time_difference": None,
                "association_method": AssociationMethod.UNMATCHED.value,
                "matched": False,
                "pose_index": None,
                "pose": None
            })

    return associations

def export_image_pose_associations_csv(
    associations: List[Dict[str, Any]],
    output_path: Union[str, Path]
) -> None:
    """
    Export image-pose association table to CSV.
    """
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "image_timestamp", "pose_timestamp", "absolute_delta_seconds", "pose_index"])
        for assoc in associations:
            writer.writerow([
                assoc["image_id"],
                assoc["image_timestamp"],
                assoc["pose_timestamp"] if assoc["pose_timestamp"] is not None else "",
                assoc["absolute_delta_seconds"] if assoc["absolute_delta_seconds"] is not None else "",
                assoc["pose_index"] if assoc["pose_index"] is not None else ""
            ])
