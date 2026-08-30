import abc
import csv
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timezone

class BaseDatasetAdapter(abc.ABC):
    """
    Abstract Base Class for UAV and Multi-view Dataset Adapters.
    Converts heterogeneous raw datasets into a normalized SIH26158 data schema.
    """
    def __init__(self, dataset_root: Union[str, Path]):
        self.dataset_root = Path(dataset_root).resolve()
        self.images: List[Dict[str, Any]] = []
        self.gps: List[Dict[str, Any]] = []
        self.imu: List[Dict[str, Any]] = []
        self.pose: List[Dict[str, Any]] = []
        self.camera: Optional[Dict[str, Any]] = None
        self.dataset_info: Dict[str, Any] = {}

    @abc.abstractmethod
    def validate_root(self) -> bool:
        """Verify that the dataset root contains expected directory structures and files."""
        pass

    @abc.abstractmethod
    def parse(self) -> None:
        """Parse raw dataset records into internal data structures."""
        pass

    def export_normalized(self, output_dir: Union[str, Path]) -> Dict[str, Path]:
        """
        Export parsed data into the normalized SIH26158 representation:
          output_dir/
          ├── dataset.json
          ├── images.csv
          ├── gps.csv
          ├── imu.csv
          ├── pose.csv
          └── camera.json
        """
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        exported_paths = {}

        # 1. dataset.json
        dataset_meta = {
            "dataset_name": self.dataset_info.get("name", "Unknown Dataset"),
            "source_type": self.dataset_info.get("type", "UAV"),
            "original_root": str(self.dataset_root),
            "export_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "record_counts": {
                "images": len(self.images),
                "gps": len(self.gps),
                "imu": len(self.imu),
                "pose": len(self.pose)
            },
            "metadata": self.dataset_info.get("metadata", {})
        }
        dataset_json_path = output_dir / "dataset.json"
        with open(dataset_json_path, "w", encoding="utf-8") as f:
            json.dump(dataset_meta, f, indent=4)
        exported_paths["dataset"] = dataset_json_path

        # 2. images.csv
        images_csv_path = output_dir / "images.csv"
        with open(images_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["image_id", "imgid", "filename", "timestamp_seconds", "width", "height"])
            for row in self.images:
                writer.writerow([
                    row.get("image_id"),
                    row.get("imgid", row.get("image_id")),
                    row.get("filename"),
                    row.get("timestamp_seconds"),
                    row.get("width"),
                    row.get("height")
                ])
        exported_paths["images"] = images_csv_path

        # 3. gps.csv
        gps_csv_path = output_dir / "gps.csv"
        with open(gps_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp_seconds", "latitude", "longitude", "altitude_if_available"])
            for row in self.gps:
                writer.writerow([
                    row.get("timestamp_seconds"),
                    row.get("latitude"),
                    row.get("longitude"),
                    row.get("altitude_if_available")
                ])
        exported_paths["gps"] = gps_csv_path

        # 4. imu.csv
        imu_csv_path = output_dir / "imu.csv"
        with open(imu_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp_seconds", "accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"])
            for row in self.imu:
                writer.writerow([
                    row.get("timestamp_seconds"),
                    row.get("accel_x"),
                    row.get("accel_y"),
                    row.get("accel_z"),
                    row.get("gyro_x"),
                    row.get("gyro_y"),
                    row.get("gyro_z")
                ])
        exported_paths["imu"] = imu_csv_path

        # 5. pose.csv
        pose_csv_path = output_dir / "pose.csv"
        with open(pose_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            has_imgid = any(row.get("imgid") is not None for row in self.pose)
            if has_imgid:
                writer.writerow(["imgid", "timestamp_seconds", "tx", "ty", "tz", "qx", "qy", "qz", "qw"])
                for row in self.pose:
                    writer.writerow([
                        row.get("imgid"),
                        row.get("timestamp_seconds"),
                        row.get("tx"),
                        row.get("ty"),
                        row.get("tz"),
                        row.get("qx"),
                        row.get("qy"),
                        row.get("qz"),
                        row.get("qw")
                    ])
            else:
                writer.writerow(["timestamp_seconds", "tx", "ty", "tz", "qx", "qy", "qz", "qw"])
                for row in self.pose:
                    writer.writerow([
                        row.get("timestamp_seconds"),
                        row.get("tx"),
                        row.get("ty"),
                        row.get("tz"),
                        row.get("qx"),
                        row.get("qy"),
                        row.get("qz"),
                        row.get("qw")
                    ])
        exported_paths["pose"] = pose_csv_path

        # 6. camera.json
        camera_json_path = output_dir / "camera.json"
        with open(camera_json_path, "w", encoding="utf-8") as f:
            json.dump(self.camera or {}, f, indent=4)
        exported_paths["camera"] = camera_json_path

        return exported_paths
