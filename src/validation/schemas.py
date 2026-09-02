import json
from dataclasses import dataclass, field
from typing import List, Optional, Literal, Dict, Any
from enum import Enum

class AccessStatus(str, Enum):
    REQUESTED = "requested"
    APPROVED = "approved"
    DOWNLOADED = "downloaded"
    VERIFIED = "verified"
    READY = "ready for evaluation"
    IDENTIFIED = "identified"

class ReconstructionRole(str, Enum):
    PRIMARY_3D_BENCHMARK = "PRIMARY_3D_BENCHMARK"
    SECONDARY_3D_TEST = "SECONDARY_3D_TEST"
    VIDEO_ONLY_TEST = "VIDEO_ONLY_TEST"
    DYNAMIC_OBJECT = "DYNAMIC_OBJECT"
    SEMANTIC_ONLY = "SEMANTIC_ONLY"
    NOT_SUITABLE_FOR_3D = "NOT_SUITABLE_FOR_3D"

class ValidationStatus(str, Enum):
    READY_FOR_METRIC_VALIDATION = "READY_FOR_METRIC_VALIDATION"
    READY_FOR_RELATIVE_VALIDATION = "READY_FOR_RELATIVE_VALIDATION"
    NOT_READY = "NOT_READY"

@dataclass
class DatasetRegistryEntry:
    dataset_id: str
    dataset_name: str
    country: str
    region: str
    scene_type: str
    source_url: str
    access_status: str
    license_status: str
    expected_data_type: List[str]
    expected_camera_metadata: bool
    expected_pose_metadata: bool
    expected_ground_truth: bool
    reconstruction_role: str
    citation: str
    permission_required: bool
    notes: str

@dataclass
class IntakeValidationResult:
    dataset_id: str
    files_exist: bool
    readable_media: bool
    frame_count: int
    resolution: Optional[str]
    fps: Optional[float]
    timestamps_available: bool
    calibration_available: bool
    pose_available: bool
    gps_rtk_available: bool
    reference_geometry_available: bool
    license_metadata_present: bool
    duplicate_or_corrupt_files: List[str]
    warnings: List[str]

@dataclass
class EvaluationResultSchema:
    dataset_id: str
    session_id: str
    input_mode: str
    reconstruction_mode: str
    frame_count: int
    registered_frames: int
    calibration_source: str
    pose_source: str
    metric_anchor_category: str
    point_count: int
    runtime_sec: float
    memory_usage_mb: Optional[float]
    reconstruction_status: str
    failure_reason: Optional[str]
    metric_accuracy: Optional[Dict[str, Any]]
    relative_quality: Optional[Dict[str, Any]]
