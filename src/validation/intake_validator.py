from pathlib import Path
import json
import cv2
from src.validation.schemas import IntakeValidationResult

class DatasetIntakeValidator:
    def __init__(self, dataset_dir: Path):
        self.dataset_dir = dataset_dir

    def validate(self, dataset_id: str) -> IntakeValidationResult:
        if not self.dataset_dir.exists() or not self.dataset_dir.is_dir():
            return IntakeValidationResult(
                dataset_id=dataset_id, files_exist=False, readable_media=False,
                frame_count=0, resolution=None, fps=None, timestamps_available=False,
                calibration_available=False, pose_available=False, gps_rtk_available=False,
                reference_geometry_available=False, license_metadata_present=False,
                duplicate_or_corrupt_files=[], warnings=["Directory not found"]
            )

        files = list(self.dataset_dir.rglob("*"))
        if not files:
            return IntakeValidationResult(
                dataset_id=dataset_id, files_exist=False, readable_media=False,
                frame_count=0, resolution=None, fps=None, timestamps_available=False,
                calibration_available=False, pose_available=False, gps_rtk_available=False,
                reference_geometry_available=False, license_metadata_present=False,
                duplicate_or_corrupt_files=[], warnings=["Directory empty"]
            )

        images = list(self.dataset_dir.rglob("*.jpg")) + list(self.dataset_dir.rglob("*.png")) + list(self.dataset_dir.rglob("*.tif")) + list(self.dataset_dir.rglob("*.tiff"))
        videos = list(self.dataset_dir.rglob("*.mp4")) + list(self.dataset_dir.rglob("*.avi"))

        corrupt = []
        res = None
        readable = False
        
        for img_path in images:
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    corrupt.append(img_path.name)
                else:
                    readable = True
                    res = f"{img.shape[1]}x{img.shape[0]}"
                    break
            except cv2.error:
                # E.g. decompression bomb / size limit exceeded
                corrupt.append(img_path.name)

        fps = None
        for vid_path in videos:
            cap = cv2.VideoCapture(str(vid_path))
            if not cap.isOpened():
                corrupt.append(vid_path.name)
            else:
                readable = True
                fps = cap.get(cv2.CAP_PROP_FPS)
                res = f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
                cap.release()
                break

        has_calib = len(list(self.dataset_dir.rglob("*calib*"))) > 0 or len(list(self.dataset_dir.rglob("*camera*"))) > 0
        has_pose = len(list(self.dataset_dir.rglob("*pose*"))) > 0 or len(list(self.dataset_dir.rglob("*trajectory*"))) > 0
        has_gps = len(list(self.dataset_dir.rglob("*gps*"))) > 0 or len(list(self.dataset_dir.rglob("*rtk*"))) > 0
        has_gt = len(list(self.dataset_dir.rglob("*.ply"))) > 0 or len(list(self.dataset_dir.rglob("*.las"))) > 0
        has_license = (self.dataset_dir / "LICENSE").exists() or (self.dataset_dir / "README.md").exists()

        return IntakeValidationResult(
            dataset_id=dataset_id,
            files_exist=True,
            readable_media=readable,
            frame_count=len(images) + len(videos), # simplification
            resolution=res,
            fps=fps,
            timestamps_available=False, # needs complex parsing
            calibration_available=has_calib,
            pose_available=has_pose,
            gps_rtk_available=has_gps,
            reference_geometry_available=has_gt,
            license_metadata_present=has_license,
            duplicate_or_corrupt_files=corrupt,
            warnings=[] if readable else ["No readable media found"]
        )
