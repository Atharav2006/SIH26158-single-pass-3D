import math
import sys
import pytest
from pathlib import Path

# Ensure project root is on sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.pose.models import Position, Quaternion, Pose
from src.pose.coordinate_frames import (
    FRAME_GLOBAL_UTM_ENU,
    FRAME_LOCAL_ENU,
    transform_to_local_enu
)

def test_transform_to_local_enu():
    origin = Position(x=465666.0, y=5247973.0, z=469.0)
    utm_pose = Pose(
        timestamp_seconds=10.0,
        position=Position(x=465766.0, y=5248073.0, z=479.0),
        orientation=Quaternion(qx=0.0, qy=0.0, qz=0.0, qw=1.0),
        source_frame=FRAME_GLOBAL_UTM_ENU,
        target_frame="Camera_RDF"
    )

    local_pose = transform_to_local_enu(utm_pose, origin)

    # Invariants
    assert local_pose.source_frame == FRAME_LOCAL_ENU
    assert local_pose.position.x == 100.0
    assert local_pose.position.y == 100.0
    assert local_pose.position.z == 10.0
    assert local_pose.position.unit == "meter"
    assert local_pose.orientation.qw == 1.0

def test_distance_invariant_under_transformation():
    origin = Position(x=500000.0, y=5000000.0, z=500.0)
    p1 = Pose(0.0, Position(500100.0, 500200.0, 520.0), Quaternion(0, 0, 0, 1), FRAME_GLOBAL_UTM_ENU, "Cam")
    p2 = Pose(1.0, Position(500150.0, 500250.0, 530.0), Quaternion(0, 0, 0, 1), FRAME_GLOBAL_UTM_ENU, "Cam")

    # UTM distance
    dx_utm = p2.position.x - p1.position.x
    dy_utm = p2.position.y - p1.position.y
    dz_utm = p2.position.z - p1.position.z
    dist_utm = math.sqrt(dx_utm**2 + dy_utm**2 + dz_utm**2)

    # Local distance
    p1_local = transform_to_local_enu(p1, origin)
    p2_local = transform_to_local_enu(p2, origin)

    dx_loc = p2_local.position.x - p1_local.position.x
    dy_loc = p2_local.position.y - p1_local.position.y
    dz_loc = p2_local.position.z - p1_local.position.z
    dist_loc = math.sqrt(dx_loc**2 + dy_loc**2 + dz_loc**2)

    assert abs(dist_utm - dist_loc) < 1e-6
