import pytest
import json
import cv2
import numpy as np
from pathlib import Path
from src.validation.schemas import DatasetRegistryEntry, ValidationStatus
from src.validation.intake_validator import DatasetIntakeValidator
from src.validation.evaluation_contract import ValidationContract

@pytest.fixture
def registry_file():
    return Path("datasets/india_validation/indian_dataset_registry.json")

def test_manifest_parsing(registry_file):
    assert registry_file.exists()
    with open(registry_file) as f:
        data = json.load(f)
    
    # Verify all 8 required datasets exist
    ids = [d["dataset_id"] for d in data]
    expected = [
        "UASG2023_DELHI_DENSE_URBAN", "UASG2023_DELHI_URBAN_AGRICULTURE",
        "UASG2023_GUJARAT_DENSE_FOREST", "UASG2023_NAGALAND_LANDSLIDE",
        "MANIPAL_UAVID", "MUAAD", "SKYEYE_AHMEDABAD", "IHUB_DATA_DRONE_INFRASTRUCTURE"
    ]
    for e in expected:
        assert e in ids

def test_missing_dataset_handling(tmp_path):
    validator = DatasetIntakeValidator(tmp_path / "nonexistent")
    res = validator.validate("dummy")
    assert res.files_exist is False
    assert res.readable_media is False
    assert "Directory not found" in res.warnings

def test_corrupted_files(tmp_path):
    d = tmp_path / "corrupt_data"
    d.mkdir()
    (d / "bad.jpg").write_text("not an image")
    
    validator = DatasetIntakeValidator(d)
    res = validator.validate("dummy")
    assert "bad.jpg" in res.duplicate_or_corrupt_files

def test_session_isolation(tmp_path):
    d1 = tmp_path / "set1"
    d2 = tmp_path / "set2"
    d1.mkdir()
    d2.mkdir()
    
    (d1 / "img.jpg").write_bytes(cv2.imencode('.jpg', np.zeros((10,10,3), np.uint8))[1].tobytes())
    (d2 / "img.jpg").write_bytes(cv2.imencode('.jpg', np.zeros((10,10,3), np.uint8))[1].tobytes())
    
    v1 = DatasetIntakeValidator(d1)
    v2 = DatasetIntakeValidator(d2)
    
    assert v1.validate("s1").files_exist
    assert v2.validate("s2").files_exist

def test_metric_fail_closed_and_relative_fallback(tmp_path):
    d = tmp_path / "dataset"
    d.mkdir()
    (d / "img.jpg").write_bytes(cv2.imencode('.jpg', np.zeros((10,10,3), np.uint8))[1].tobytes())
    (d / "gps.csv").write_text("lat,lon\n1,1")
    (d / "LICENSE").write_text("MIT")
    
    validator = DatasetIntakeValidator(d)
    intake = validator.validate("test_id")
    
    # Metadata has GPS but no ground truth expected
    meta = DatasetRegistryEntry(
        dataset_id="test_id", dataset_name="test", country="India", region="Delhi",
        scene_type="Urban", source_url="", access_status="downloaded", license_status="RESEARCH_ONLY",
        expected_data_type=[], expected_camera_metadata=True, expected_pose_metadata=False,
        expected_ground_truth=False, reconstruction_role="PRIMARY_3D_BENCHMARK",
        citation="", permission_required=True, notes=""
    )
    
    status = ValidationContract.evaluate_readiness(intake, meta)
    # Fails closed on metric, falls back to relative
    assert status == ValidationStatus.READY_FOR_RELATIVE_VALIDATION

    # Now with ground truth
    (d / "gt.ply").write_text("ply")
    intake_gt = validator.validate("test_id")
    meta.expected_ground_truth = True
    status_gt = ValidationContract.evaluate_readiness(intake_gt, meta)
    assert status_gt == ValidationStatus.READY_FOR_METRIC_VALIDATION

def test_dataset_status_transitions(tmp_path):
    d = tmp_path / "dataset"
    d.mkdir()
    (d / "img.jpg").write_bytes(cv2.imencode('.jpg', np.zeros((10,10,3), np.uint8))[1].tobytes())
    (d / "LICENSE").write_text("MIT")
    
    intake = DatasetIntakeValidator(d).validate("id")
    meta = DatasetRegistryEntry(
        dataset_id="id", dataset_name="test", country="India", region="Delhi",
        scene_type="Urban", source_url="", access_status="requested", # requested not downloaded
        license_status="RESEARCH_ONLY", expected_data_type=[], expected_camera_metadata=True,
        expected_pose_metadata=False, expected_ground_truth=False, reconstruction_role="",
        citation="", permission_required=True, notes=""
    )
    
    assert ValidationContract.evaluate_readiness(intake, meta) == ValidationStatus.NOT_READY
    
    meta.access_status = "downloaded"
    assert ValidationContract.evaluate_readiness(intake, meta) == ValidationStatus.READY_FOR_RELATIVE_VALIDATION
