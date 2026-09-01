import pytest
import torch
import numpy as np
import cv2
import tempfile
import csv
import json
from pathlib import Path

from src.neural_reconstruction.dataset import NeRFDataset
from src.metrics.alignment import rotation_matrix_to_quaternion

@pytest.fixture
def mock_dataset_env(tmp_path):
    # Create fake images
    img_dir = tmp_path / "MAV Images"
    img_dir.mkdir()
    
    img_size = (1920, 1080)
    for i in range(3):
        img_np = np.random.randint(0, 255, (img_size[1], img_size[0], 3), dtype=np.uint8)
        cv2.imwrite(str(img_dir / f"{i:05d}.jpg"), img_np)
        
    # Create images.csv
    images_csv = tmp_path / "images.csv"
    with open(images_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['timestamp_seconds', 'filename'])
        w.writeheader()
        w.writerow({'timestamp_seconds': '0.0', 'filename': '00000.jpg'})
        w.writerow({'timestamp_seconds': '0.1', 'filename': '00001.jpg'})
        w.writerow({'timestamp_seconds': '0.2', 'filename': '00002.jpg'})
        
    # Create poses_csv (B2 fused trajectory)
    poses_csv = tmp_path / "b2_fused_trajectory.csv"
    
    # Let's make explicit poses
    # Pose 0: Identity R, Translation [1, 2, 3]
    q0 = [0, 0, 0, 1] # Identity
    c0 = [1, 2, 3]
    
    # Pose 1: Identity R, Translation [0, 0, 0]
    q1 = [0, 0, 0, 1]
    c1 = [0, 0, 0]
    
    # Pose 2: 90 deg rotation around X, Translation [0, 0, 0]
    # R_x(90) = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
    R_x90 = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    q2 = rotation_matrix_to_quaternion(R_x90)
    c2 = [0, 0, 0]
    
    with open(poses_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['timestamp', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'])
        w.writeheader()
        w.writerow({'timestamp': '0.0', 'x': c0[0], 'y': c0[1], 'z': c0[2], 'qx': q0[0], 'qy': q0[1], 'qz': q0[2], 'qw': q0[3]})
        w.writerow({'timestamp': '0.1', 'x': c1[0], 'y': c1[1], 'z': c1[2], 'qx': q1[0], 'qy': q1[1], 'qz': q1[2], 'qw': q1[3]})
        w.writerow({'timestamp': '0.2', 'x': c2[0], 'y': c2[1], 'z': c2[2], 'qx': q2[0], 'qy': q2[1], 'qz': q2[2], 'qw': q2[3]})
        
    # Create intrinsics JSON
    intrinsics_json = tmp_path / "reconstruction_summary.json"
    # Pinhole, no distortion for testing simple ray math
    # fx=1000, fy=1000, cx=960, cy=540, all k=0
    cam_data = {
        "camera_calibration": {
            "model": "FULL_OPENCV",
            "width": 1920,
            "height": 1080,
            "fx": 1000.0,
            "fy": 1000.0,
            "cx": 960.0,
            "cy": 540.0,
            "distortion_k1_k2_p1_p2_k3": [0.0, 0.0, 0.0, 0.0, 0.0]
        }
    }
    with open(intrinsics_json, 'w') as f:
        json.dump(cam_data, f)
        
    return images_csv, poses_csv, intrinsics_json

def test_b4_dataset_loading_and_resolution(mock_dataset_env):
    images_csv, poses_csv, intrinsics_json = mock_dataset_env
    
    res = (256, 144)
    image_dir = images_csv.parent / "MAV Images"
    ds = NeRFDataset(images_csv, poses_csv, intrinsics_json, image_dir, resolution=res, split_indices=[0, 1])
    
    assert len(ds) == 2
    assert ds.images.shape == (2, res[1], res[0], 3)
    assert ds.images.dtype == torch.float32
    # Ensure values in [0, 1]
    assert ds.images.max() <= 1.0
    assert ds.images.min() >= 0.0
    
    # Test intrinsics scaling
    # orig fx=1000, target width=256, orig width=1920
    expected_fx = 1000.0 * (256 / 1920)
    assert np.isclose(ds.K_scaled[0, 0], expected_fx)

def test_b4_dataset_ray_geometry(mock_dataset_env):
    images_csv, poses_csv, intrinsics_json = mock_dataset_env
    
    # No downscaling to make math direct
    res = (1920, 1080)
    image_dir = images_csv.parent / "MAV Images"
    ds = NeRFDataset(images_csv, poses_csv, intrinsics_json, image_dir, resolution=res)
    
    # 1. Test Pose 0 (Identity R, Translation [1,2,3])
    rays_o, rays_d, rgb = ds.get_image_rays(0)
    assert rays_o.shape == (1080, 1920, 3)
    # Origin should be exactly [1, 2, 3] everywhere
    assert torch.allclose(rays_o[0, 0], torch.tensor([1.0, 2.0, 3.0]))
    
    # Test Ray Direction at principal point (cx=960, cy=540)
    # At (960, 540), (u-cx)=0, (v-cy)=0, so dir = [0, 0, 1]
    # R_wc is Identity, so world dir is [0, 0, 1]
    center_ray_d = rays_d[540, 960]
    assert torch.allclose(center_ray_d, torch.tensor([0.0, 0.0, 1.0]), atol=1e-5)
    
    # 2. Test Ray Normalization everywhere
    norms = torch.norm(rays_d, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms))
    
    # 3. Test Pose 2 (R_x90, Translation [0,0,0])
    rays_o2, rays_d2, rgb2 = ds.get_image_rays(2)
    # Camera +Z points into world +Y (since R_x90 rotates +Z to +Y, +Y to -Z)
    center_ray_d2 = rays_d2[540, 960]
    # In OpenCV camera, Z is [0, 0, 1]. R_x90 @ [0, 0, 1] = [0, -1, 0] ?
    # Let's check: R_x90 = [[1,0,0], [0,0,-1], [0,1,0]]
    # R_x90 @ [0,0,1]^T = [0, -1, 0]^T
    assert torch.allclose(center_ray_d2, torch.tensor([0.0, -1.0, 0.0]), atol=1e-5)

def test_b4_dataset_random_batch(mock_dataset_env):
    images_csv, poses_csv, intrinsics_json = mock_dataset_env
    
    res = (32, 32)
    image_dir = images_csv.parent / "MAV Images"
    ds = NeRFDataset(images_csv, poses_csv, intrinsics_json, image_dir, resolution=res)
    
    rays_o, rays_d, rgb = ds.get_random_batch(1024)
    assert rays_o.shape == (1024, 3)
    assert rays_d.shape == (1024, 3)
    assert rgb.shape == (1024, 3)
