import cv2
import numpy as np
from typing import Dict, Any, Tuple

class CameraPreprocessor:
    """
    Handles camera distortion and rectification. 
    Strictly preserves the validated FULL_OPENCV intrinsic parameters.
    """
    def __init__(self, calibration_data: Dict[str, Any]):
        """
        calibration_data: The exact dictionary from B0 JSON camera_calibration.
        """
        self.source_model = calibration_data.get('model', 'FULL_OPENCV')
        self.width = calibration_data['width']
        self.height = calibration_data['height']
        
        self.K_source = np.array([
            [calibration_data['fx'], 0.0, calibration_data['cx']],
            [0.0, calibration_data['fy'], calibration_data['cy']],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
        
        self.dist_coeffs = np.array(calibration_data['distortion_k1_k2_p1_p2_k3'], dtype=np.float64)
        
        # Calculate optimal new camera matrix for pinhole rectification
        # alpha=0 means no black pixels (crop to valid), alpha=1 means retain all pixels.
        # Following B4 protocol, we use alpha=0 to ensure valid rays.
        self.K_rect, self.roi = cv2.getOptimalNewCameraMatrix(
            self.K_source, self.dist_coeffs, (self.width, self.height), 0, (self.width, self.height)
        )
        
        # Precompute remap maps
        self.map1, self.map2 = cv2.initUndistortRectifyMap(
            self.K_source, self.dist_coeffs, None, self.K_rect, (self.width, self.height), cv2.CV_32FC1
        )

    def rectify_image(self, image: np.ndarray) -> np.ndarray:
        """
        Takes a distorted RGB image (H, W, 3) and returns the rectified pinhole equivalent.
        """
        if image.shape[:2] != (self.height, self.width):
            raise ValueError(f"Image shape {image.shape} does not match calibration {(self.height, self.width)}")
            
        undistorted = cv2.remap(image, self.map1, self.map2, cv2.INTER_LINEAR)
        return undistorted

    def get_rectified_intrinsics(self) -> np.ndarray:
        return self.K_rect.copy()

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "source_model": self.source_model,
            "original_dimensions": (self.width, self.height),
            "rectified_dimensions": (self.width, self.height),
            "K_source": self.K_source.tolist(),
            "K_rect": self.K_rect.tolist(),
            "distortion_coeffs": self.dist_coeffs.tolist(),
            "roi": self.roi
        }
