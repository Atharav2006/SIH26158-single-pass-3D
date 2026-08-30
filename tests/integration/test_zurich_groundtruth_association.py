import os
import sys
import csv
import pytest
from pathlib import Path

# Ensure project root is on sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.pose.pose_loader import load_poses_from_csv, load_image_metadata
from src.pose.association import (
    associate_groundtruth_by_imgid,
    export_image_groundtruth_associations_csv,
    AssociationMethod
)

@pytest.fixture
def normalized_dir():
    p = Path("datasets/normalized/zurich_mav_sample").resolve()
    if not (p / "images.csv").exists():
        p = Path("outputs/reports/zurich_mav").resolve()
    if not (p / "images.csv").exists():
        pytest.skip(f"Normalized Zurich MAV dataset not found in {p}")
    return p

def test_image_identity_and_groundtruth_association(normalized_dir, tmp_path):
    images_csv = normalized_dir / "images.csv"
    pose_csv = normalized_dir / "pose.csv"

    images = load_image_metadata(images_csv)
    poses = load_poses_from_csv(pose_csv)

    assert len(images) > 0
    assert len(poses) > 0

    # 1. Verify imgid presence and uniqueness
    imgids = [img["imgid"] for img in images]
    assert len(imgids) == len(set(imgids)), "Duplicate imgid found in images.csv"
    for img in images:
        assert isinstance(img["imgid"], int)
        # Check filename consistency (e.g. 00001.jpg -> 1)
        expected_id = int(Path(img["filename"]).stem)
        assert img["imgid"] == expected_id

    # 2. Run authoritative exact-ID association
    assocs = associate_groundtruth_by_imgid(images, poses)
    assert len(assocs) == len(images)

    matched = [a for a in assocs if a.matched]
    unmatched = [a for a in assocs if not a.matched]

    # Calculate ground truth poses with imgid within the sample range dynamically
    max_imgid = max(img["imgid"] for img in images)
    min_imgid = min(img["imgid"] for img in images)
    expected_gt_in_sample = [p for p in poses if p.imgid is not None and min_imgid <= p.imgid <= max_imgid]

    assert len(matched) == len(expected_gt_in_sample), (
        f"Expected {len(expected_gt_in_sample)} exact keyframe matches, got {len(matched)}"
    )
    assert len(unmatched) == len(images) - len(expected_gt_in_sample)

    # 3. Verify matched keyframe properties
    for m in matched:
        assert m.association_method == AssociationMethod.EXACT_ID.value
        assert m.ground_truth_imgid == m.imgid
        assert m.delta_seconds is not None
        assert m.delta_seconds < 1e-4  # Zero residual for exact keyframes
        assert m.pose is not None

    # 4. Verify unmatched intermediate frame properties
    for u in unmatched:
        assert u.association_method == AssociationMethod.UNMATCHED.value
        assert u.ground_truth_imgid is None
        assert u.delta_seconds is None
        assert u.pose is None

    # 5. Export and check CSV output
    out_csv = tmp_path / "image_groundtruth_associations.csv"
    export_image_groundtruth_associations_csv(assocs, out_csv)
    assert out_csv.exists()

    with open(out_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) == len(images)
        assert "association_method" in reader.fieldnames
        assert "matched" in reader.fieldnames
