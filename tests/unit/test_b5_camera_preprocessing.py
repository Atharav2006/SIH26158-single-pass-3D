import numpy as np
import pytest
from src.depth_fusion.camera_preprocessing import CameraPreprocessor

@pytest.fixture
def dummy_calibration():
    return {
        'model': 'FULL_OPENCV',
        'width': 1920,
        'height': 1080,
        'fx': 893.4,
        'fy': 898.3,
        'cx': 951.1,
        'cy': 555.1,
        'distortion_k1_k2_p1_p2_k3': [
            -0.2805, 0.1158, -0.001, 0.0001, -0.027
        ]
    }

def test_camera_preprocessor_initialization(dummy_calibration):
    preprocessor = CameraPreprocessor(dummy_calibration)
    
    meta = preprocessor.get_metadata()
    assert meta["source_model"] == "FULL_OPENCV"
    assert meta["original_dimensions"] == (1920, 1080)
    
    K_source = np.array(meta["K_source"])
    K_rect = np.array(meta["K_rect"])
    
    assert K_source[0, 0] == 893.4
    
    # K_rect must be valid, finite, and consistent with alpha=0 (which may equal K_source)
    assert K_rect.shape == (3, 3)
    assert np.all(np.isfinite(K_rect))
    assert K_rect[0, 0] > 0 and K_rect[1, 1] > 0  # Focal lengths must be positive
    assert K_rect[2, 0] == 0 and K_rect[2, 1] == 0 and K_rect[2, 2] == 1  # Bottom row must be [0,0,1]
    
    # Check that focal lengths are positive and valid
    assert K_rect[0, 0] > 0
    assert K_rect[1, 1] > 0

def test_camera_rectification_shape(dummy_calibration):
    preprocessor = CameraPreprocessor(dummy_calibration)
    
    dummy_img = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    rectified = preprocessor.rectify_image(dummy_img)
    
    assert rectified.shape == (1080, 1920, 3)
    assert not np.isnan(rectified).any()

def test_camera_rectification_invalid_shape(dummy_calibration):
    preprocessor = CameraPreprocessor(dummy_calibration)
    
    bad_img = np.random.randint(0, 255, (500, 500, 3), dtype=np.uint8)
    
    with pytest.raises(ValueError):
        preprocessor.rectify_image(bad_img)
