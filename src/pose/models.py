from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class Position:
    """Explicit 3D Cartesian position with units."""
    x: float
    y: float
    z: float
    unit: str = "meter"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "unit": self.unit
        }

@dataclass
class Quaternion:
    """Unit quaternion for 3D rotation in Hamilton convention (qx, qy, qz, qw) with scalar-last ordering."""
    qx: float
    qy: float
    qz: float
    qw: float
    convention: str = "Hamilton"

    def norm(self) -> float:
        return (self.qx**2 + self.qy**2 + self.qz**2 + self.qw**2)**0.5

    def is_normalized(self, tol: float = 1e-4) -> bool:
        return abs(self.norm() - 1.0) < tol

    def normalized(self) -> 'Quaternion':
        n = self.norm()
        if n < 1e-12:
            return Quaternion(0.0, 0.0, 0.0, 1.0, self.convention)
        return Quaternion(self.qx / n, self.qy / n, self.qz / n, self.qw / n, self.convention)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "qx": self.qx,
            "qy": self.qy,
            "qz": self.qz,
            "qw": self.qw,
            "convention": self.convention
        }

class Pose:
    """
    Explicit 6DoF Pose structure.
    Represents transformation from source_frame to target_frame or object position in source_frame.
    """
    def __init__(
        self,
        timestamp_seconds: float,
        position_xyz: Optional[Position] = None,
        orientation_xyzw: Optional[Quaternion] = None,
        source_frame: str = "UTM_Zone_32N_ENU",
        target_frame: str = "Camera_RDF",
        pose_semantics: str = "camera_optical_center_in_world",
        position: Optional[Position] = None,
        orientation: Optional[Quaternion] = None,
        imgid: Optional[int] = None
    ):
        self.timestamp_seconds = float(timestamp_seconds)
        self.position_xyz = position_xyz or position
        self.orientation_xyzw = orientation_xyzw or orientation
        self.source_frame = str(source_frame)
        self.target_frame = str(target_frame)
        self.pose_semantics = str(pose_semantics)
        self.imgid = int(imgid) if imgid is not None else None

        if self.position_xyz is None:
            raise ValueError("Position must be provided via position_xyz or position")
        if self.orientation_xyzw is None:
            raise ValueError("Orientation must be provided via orientation_xyzw or orientation")

    @property
    def position(self) -> Position:
        """Alias for position_xyz for clean property access."""
        return self.position_xyz

    @property
    def orientation(self) -> Quaternion:
        """Alias for orientation_xyzw for clean property access."""
        return self.orientation_xyzw

    def to_dict(self) -> Dict[str, Any]:
        return {
            "imgid": self.imgid,
            "timestamp_seconds": self.timestamp_seconds,
            "position_xyz": self.position_xyz.to_dict(),
            "orientation_xyzw": self.orientation_xyzw.to_dict(),
            "source_frame": self.source_frame,
            "target_frame": self.target_frame,
            "pose_semantics": self.pose_semantics
        }

    def __repr__(self) -> str:
        return (f"Pose(imgid={self.imgid}, "
                f"timestamp_seconds={self.timestamp_seconds}, "
                f"position_xyz={self.position_xyz}, "
                f"orientation_xyzw={self.orientation_xyzw}, "
                f"source_frame='{self.source_frame}', "
                f"target_frame='{self.target_frame}', "
                f"pose_semantics='{self.pose_semantics}')")
