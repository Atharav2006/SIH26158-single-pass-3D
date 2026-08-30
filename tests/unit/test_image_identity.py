import os
import sys
import re
import pytest
from pathlib import Path

# Ensure project root is on sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.pose.models import Position, Quaternion, Pose
from src.pose.association import (
    AssociationMethod,
    GroundTruthAssociation,
    associate_groundtruth_by_imgid,
    export_image_groundtruth_associations_csv
)

def test_extract_imgid_from_filename():
    def extract_id(fname: str) -> int:
        match = re.search(r'(\d+)$', Path(fname).stem)
        return int(match.group(1)) if match else -1

    assert extract_id("00001.jpg") == 1
    assert extract_id("00350.jpg") == 350
    assert extract_id("Calibration_Image_01.png") == 1
    assert extract_id("frame_000042.png") == 42
    assert extract_id("no_digits.jpg") == -1

def test_associate_groundtruth_by_imgid_exact_and_unmatched():
    images = [
        {"image_id": 1, "imgid": 1, "filename": "00001.jpg", "timestamp_seconds": 7.009129},
        {"image_id": 2, "imgid": 2, "filename": "00002.jpg", "timestamp_seconds": 7.042462},
        {"image_id": 31, "imgid": 31, "filename": "00031.jpg", "timestamp_seconds": 7.988182},
    ]

    p1 = Pose(timestamp_seconds=7.009129, position_xyz=Position(0, 0, 0), orientation_xyzw=Quaternion(0, 0, 0, 1), imgid=1)
    p31 = Pose(timestamp_seconds=7.988182, position_xyz=Position(1, 1, 1), orientation_xyzw=Quaternion(0, 0, 0, 1), imgid=31)
    p61 = Pose(timestamp_seconds=9.002265, position_xyz=Position(2, 2, 2), orientation_xyzw=Quaternion(0, 0, 0, 1), imgid=61)

    poses = [p1, p31, p61]

    assocs = associate_groundtruth_by_imgid(images, poses)

    assert len(assocs) == 3

    # Image 1 -> EXACT_ID match
    assert assocs[0].matched is True
    assert assocs[0].association_method == AssociationMethod.EXACT_ID.value
    assert assocs[0].ground_truth_imgid == 1
    assert assocs[0].delta_seconds == 0.0
    assert assocs[0].pose is not None

    # Image 2 -> UNMATCHED intermediate frame
    assert assocs[1].matched is False
    assert assocs[1].association_method == AssociationMethod.UNMATCHED.value
    assert assocs[1].ground_truth_imgid is None
    assert assocs[1].delta_seconds is None
    assert assocs[1].pose is None

    # Image 31 -> EXACT_ID match
    assert assocs[2].matched is True
    assert assocs[2].association_method == AssociationMethod.EXACT_ID.value
    assert assocs[2].ground_truth_imgid == 31
    assert assocs[2].delta_seconds == 0.0

def test_export_image_groundtruth_associations_csv(tmp_path):
    assoc = GroundTruthAssociation(
        image_id=1,
        imgid=1,
        filename="00001.jpg",
        image_timestamp_seconds=7.009129,
        ground_truth_imgid=1,
        ground_truth_pose_timestamp_seconds=7.009129,
        association_method="EXACT_ID",
        matched=True,
        delta_seconds=0.0
    )

    out_csv = tmp_path / "test_assocs.csv"
    export_image_groundtruth_associations_csv([assoc], out_csv)

    assert out_csv.exists()
    content = out_csv.read_text(encoding="utf-8")
    assert "image_id,imgid,filename,image_timestamp_seconds" in content
    assert "1,1,00001.jpg,7.009129,1,7.009129,EXACT_ID,true,0.0" in content
