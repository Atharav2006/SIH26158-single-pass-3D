from typing import Tuple, Optional
from src.pose.models import Position, Quaternion, Pose

# Standardized Frame Identifiers
FRAME_GLOBAL_UTM_ENU = "UTM_Zone_32N_ENU"
FRAME_LOCAL_ENU = "Local_ENU"
FRAME_BODY_FLU = "Body_FLU"
FRAME_CAMERA_RDF = "Camera_RDF"

def transform_to_local_enu(pose: Pose, origin: Position) -> Pose:
    """
    Transform a pose expressed in global UTM Zone 32N ENU to a local ENU frame centered at origin.
    
    Mathematical definition:
      p_local = p_utm - p_origin
      R_local = R_utm (orientation remains referenced to East-North-Up)
    
    Args:
        pose: Pose in UTM_Zone_32N_ENU frame.
        origin: Reference origin Position in UTM coordinates (meters).
        
    Returns:
        Pose in Local_ENU frame.
    """
    if pose.source_frame != FRAME_GLOBAL_UTM_ENU and pose.source_frame != FRAME_LOCAL_ENU:
        raise ValueError(f"Expected source frame {FRAME_GLOBAL_UTM_ENU} or {FRAME_LOCAL_ENU}, got {pose.source_frame}")

    local_pos = Position(
        x=round(pose.position_xyz.x - origin.x, 6),
        y=round(pose.position_xyz.y - origin.y, 6),
        z=round(pose.position_xyz.z - origin.z, 6),
        unit="meter"
    )

    quat = pose.orientation_xyzw.normalized()

    return Pose(
        timestamp_seconds=pose.timestamp_seconds,
        position_xyz=local_pos,
        orientation_xyzw=quat,
        source_frame=FRAME_LOCAL_ENU,
        target_frame=pose.target_frame,
        pose_semantics=pose.pose_semantics,
        imgid=pose.imgid
    )
