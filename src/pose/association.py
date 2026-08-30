import csv
from enum import Enum
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass

from src.pose.models import Pose

class AssociationMethod(str, Enum):
    """Explicit taxonomy for image-to-telemetry/pose association."""
    EXACT_ID = "EXACT_ID"
    TIMESTAMP_NEAREST = "TIMESTAMP_NEAREST"
    UNMATCHED = "UNMATCHED"

@dataclass
class GroundTruthAssociation:
    """Explicit record of image-to-ground-truth-pose relationship."""
    image_id: int
    imgid: int
    filename: str
    image_timestamp_seconds: float
    ground_truth_imgid: Optional[int]
    ground_truth_pose_timestamp_seconds: Optional[float]
    association_method: str
    matched: bool
    delta_seconds: Optional[float]
    pose: Optional[Pose] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_id": self.image_id,
            "imgid": self.imgid,
            "filename": self.filename,
            "image_timestamp_seconds": self.image_timestamp_seconds,
            "ground_truth_imgid": self.ground_truth_imgid,
            "ground_truth_pose_timestamp_seconds": self.ground_truth_pose_timestamp_seconds,
            "association_method": self.association_method,
            "matched": self.matched,
            "delta_seconds": self.delta_seconds
        }

def associate_groundtruth_by_imgid(
    images: List[Dict[str, Any]],
    poses: List[Pose]
) -> List[GroundTruthAssociation]:
    """
    Authoritative Ground-Truth Association for Zurich Urban MAV:
    Maps each image's native imgid directly to the photogrammetric ground-truth imgid.
    
    Distinguishes:
      - EXACT_ID: Keyframes present in GroundTruthAGL.csv (e.g. imgid 1, 31, 61...).
      - UNMATCHED: Intermediate video frames lacking discrete 1 Hz bundle adjustment ground truth.
    """
    # Build dictionary of ground-truth poses indexed by imgid
    gt_map: Dict[int, Pose] = {}
    for p in poses:
        if p.imgid is not None:
            gt_map[int(p.imgid)] = p

    associations: List[GroundTruthAssociation] = []

    for img in images:
        image_id = int(img["image_id"])
        native_imgid = int(img.get("imgid", image_id))
        filename = str(img["filename"])
        img_ts = float(img["timestamp_seconds"])

        if native_imgid in gt_map:
            matched_pose = gt_map[native_imgid]
            gt_ts = matched_pose.timestamp_seconds
            dt = round(abs(img_ts - gt_ts), 6)

            assoc = GroundTruthAssociation(
                image_id=image_id,
                imgid=native_imgid,
                filename=filename,
                image_timestamp_seconds=img_ts,
                ground_truth_imgid=native_imgid,
                ground_truth_pose_timestamp_seconds=gt_ts,
                association_method=AssociationMethod.EXACT_ID.value,
                matched=True,
                delta_seconds=dt,
                pose=matched_pose
            )
        else:
            assoc = GroundTruthAssociation(
                image_id=image_id,
                imgid=native_imgid,
                filename=filename,
                image_timestamp_seconds=img_ts,
                ground_truth_imgid=None,
                ground_truth_pose_timestamp_seconds=None,
                association_method=AssociationMethod.UNMATCHED.value,
                matched=False,
                delta_seconds=None,
                pose=None
            )
        associations.append(assoc)

    return associations

def export_image_groundtruth_associations_csv(
    associations: List[Union[GroundTruthAssociation, Dict[str, Any]]],
    output_path: Union[str, Path]
) -> None:
    """
    Export image-to-ground-truth association table to CSV.
    
    Columns:
      image_id, imgid, filename, image_timestamp_seconds, ground_truth_imgid,
      ground_truth_pose_timestamp_seconds, association_method, matched, delta_seconds
    """
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "image_id",
            "imgid",
            "filename",
            "image_timestamp_seconds",
            "ground_truth_imgid",
            "ground_truth_pose_timestamp_seconds",
            "association_method",
            "matched",
            "delta_seconds"
        ])
        for a in associations:
            d = a.to_dict() if isinstance(a, GroundTruthAssociation) else a
            writer.writerow([
                d["image_id"],
                d["imgid"],
                d["filename"],
                d["image_timestamp_seconds"],
                d["ground_truth_imgid"] if d["ground_truth_imgid"] is not None else "",
                d["ground_truth_pose_timestamp_seconds"] if d["ground_truth_pose_timestamp_seconds"] is not None else "",
                d["association_method"],
                str(d["matched"]).lower(),
                d["delta_seconds"] if d["delta_seconds"] is not None else ""
            ])
