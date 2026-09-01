import json
import csv
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Tuple

from src.metrics.alignment import quaternion_to_rotation_matrix

class NeRFDataset(Dataset):
    def __init__(self, 
                 images_csv: Path, 
                 poses_csv: Path, 
                 intrinsics_json: Path, 
                 image_dir: Path,
                 split: str = 'train',
                 resolution: Tuple[int, int] = (256, 144),
                 split_indices: List[int] = None,
                 device: torch.device = torch.device('cpu'),
                 depth_prior = None):
        """
        Args:
            images_csv: Path to images.csv (timestamp, filename)
            poses_csv: Path to b2_fused_trajectory.csv
            intrinsics_json: Path to reconstruction_summary.json (FULL_OPENCV params)
            image_dir: Path to directory containing the actual images
            split: 'train', 'val', or 'test'
            resolution: Target (width, height) for training to fit VRAM
            split_indices: List of frame indices to include in this split.
            device: Torch device (can store dataset on CPU and move to GPU in batches)
            depth_prior: Optional DepthPrior instance to generate pseudo-depth maps
        """
        self.split = split
        self.target_width, self.target_height = resolution
        self.device = device
        
        # 1. Load Camera Intrinsics (FULL_OPENCV)
        self._load_intrinsics(intrinsics_json)
        
        # 2. Load Trajectory and select split
        all_poses = self._load_poses(poses_csv, images_csv, image_dir)
        if split_indices is not None:
            self.poses = [all_poses[i] for i in split_indices]
        else:
            self.poses = all_poses
            
        if len(self.poses) == 0:
            raise ValueError(f"No poses found for split {split}")
            
        # 3. Preprocess and load images (Undistort, Resize)
        self._load_and_preprocess_images()
        
        # 4. Extract depth prior if provided
        self.depths = None
        if depth_prior is not None:
            print(f"[{split}] Extracting depth priors using {depth_prior.source_type}...")
            # batch predict to save time
            # do it in small chunks to save VRAM
            depths_list = []
            chunk_sz = 8
            for i in range(0, len(self.images), chunk_sz):
                batch = self.images[i:i+chunk_sz].to(depth_prior.device)
                d = depth_prior.predict_batch(batch)
                depths_list.append(d.cpu())
            self.depths = torch.cat(depths_list, dim=0).to(self.device)
            
        # 5. Precompute Ray Origins and Directions
        self._precompute_rays()

    def _load_intrinsics(self, json_path: Path):
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        cam_data = data.get('camera_calibration')
        if not cam_data:
            raise ValueError("No camera data found in reconstruction summary.")
            
        # FULL_OPENCV: fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6
        self.orig_width = int(cam_data['width'])
        self.orig_height = int(cam_data['height'])
        
        fx, fy, cx, cy = cam_data['fx'], cam_data['fy'], cam_data['cx'], cam_data['cy']
        dist = cam_data['distortion_k1_k2_p1_p2_k3']
        params = [fx, fy, cx, cy] + dist + [0.0, 0.0, 0.0]
        
        self.dist_coeffs = np.array(params[4:12], dtype=np.float32)
        
        self.K_orig = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Compute optimal new camera matrix for undistortion
        self.K_rectified, self.roi = cv2.getOptimalNewCameraMatrix(
            self.K_orig, self.dist_coeffs, (self.orig_width, self.orig_height), 1, (self.orig_width, self.orig_height)
        )
        
        # Compute scale factors for target resolution
        scale_x = self.target_width / self.orig_width
        scale_y = self.target_height / self.orig_height
        
        # Scale the rectified K matrix
        self.K_scaled = self.K_rectified.copy()
        self.K_scaled[0, 0] *= scale_x
        self.K_scaled[1, 1] *= scale_y
        self.K_scaled[0, 2] *= scale_x
        self.K_scaled[1, 2] *= scale_y

    def _load_poses(self, poses_csv: Path, images_csv: Path, image_dir: Path) -> List[Dict]:
        # Map timestamp -> filename
        ts_to_img = {}
        with open(images_csv, 'r') as f:
            for r in csv.DictReader(f):
                ts_to_img[float(r['timestamp_seconds'])] = image_dir / r['filename']
                
        poses = []
        with open(poses_csv, 'r') as f:
            for r in csv.DictReader(f):
                ts = float(r['timestamp'])
                if ts not in ts_to_img:
                    continue
                    
                x, y, z = float(r['x']), float(r['y']), float(r['z'])
                qx, qy, qz, qw = float(r['qx']), float(r['qy']), float(r['qz']), float(r['qw'])
                
                # B2 poses are Camera-to-World (R_wc, t_wc)
                # Ensure quaternion is normalized
                q = np.array([qx, qy, qz, qw])
                norm = np.linalg.norm(q)
                if norm > 1e-12:
                    q /= norm
                else:
                    q = np.array([0., 0., 0., 1.])
                    
                R_wc = quaternion_to_rotation_matrix(q)
                C_w = np.array([x, y, z])
                
                poses.append({
                    'timestamp': ts,
                    'image_path': ts_to_img[ts],
                    'R_wc': R_wc,
                    'C_w': C_w
                })
        return poses

    def _load_and_preprocess_images(self):
        self.images = []
        
        # Precompute rectification maps
        map1, map2 = cv2.initUndistortRectifyMap(
            self.K_orig, self.dist_coeffs, None, self.K_rectified, 
            (self.orig_width, self.orig_height), cv2.CV_32FC1
        )
        
        for p in self.poses:
            img_path = str(p['image_path'])
            img = cv2.imread(img_path)
            if img is None:
                raise FileNotFoundError(f"Missing image: {img_path}")
                
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Undistort
            undistorted = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)
            
            # Resize to target resolution
            resized = cv2.resize(undistorted, (self.target_width, self.target_height), interpolation=cv2.INTER_AREA)
            
            # Convert to float32 [0, 1]
            resized = resized.astype(np.float32) / 255.0
            
            self.images.append(torch.from_numpy(resized))
            
        self.images = torch.stack(self.images).to(self.device)  # [N, H, W, 3]

    def _precompute_rays(self):
        N = len(self.poses)
        H, W = self.target_height, self.target_width
        
        # Grid of pixel coordinates
        i, j = torch.meshgrid(
            torch.arange(W, dtype=torch.float32), 
            torch.arange(H, dtype=torch.float32), 
            indexing='xy'
        )
        
        # Ray directions in camera coordinates
        # OpenCV convention: +X right, +Y down, +Z forward
        fx = self.K_scaled[0, 0]
        fy = self.K_scaled[1, 1]
        cx = self.K_scaled[0, 2]
        cy = self.K_scaled[1, 2]
        
        dirs = torch.stack([
            (i - cx) / fx,
            (j - cy) / fy,
            torch.ones_like(i)
        ], dim=-1)  # [H, W, 3]
        
        self.rays_o = torch.zeros((N, H, W, 3), dtype=torch.float32, device=self.device)
        self.rays_d = torch.zeros((N, H, W, 3), dtype=torch.float32, device=self.device)
        
        for idx, p in enumerate(self.poses):
            R_wc = torch.from_numpy(p['R_wc']).float().to(self.device)
            C_w = torch.from_numpy(p['C_w']).float().to(self.device)
            
            # Transform to world coordinates: ray_d_world = R_wc @ ray_d_cam
            rays_d_world = torch.einsum('ij,hwj->hwi', R_wc, dirs.to(self.device))
            
            # Normalize
            rays_d_world = rays_d_world / torch.norm(rays_d_world, dim=-1, keepdim=True)
            
            rays_o_world = C_w.expand_as(rays_d_world)
            
            self.rays_o[idx] = rays_o_world
            self.rays_d[idx] = rays_d_world

    def __len__(self):
        return len(self.poses)

    def get_image_rays(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns all rays for a given image. Useful for validation/rendering.
        Outputs: rays_o [H,W,3], rays_d [H,W,3], rgb [H,W,3]
        """
        return self.rays_o[idx], self.rays_d[idx], self.images[idx]
        
    def get_random_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns a random batch of rays across all images in the dataset.
        Useful for training.
        Outputs: rays_o [B,3], rays_d [B,3], rgb [B,3]
        """
        N, H, W, _ = self.rays_o.shape
        # Flattened indexing
        indices = torch.randint(0, N * H * W, (batch_size,), device=self.device)
        
        # Flatten tensors temporarily for sampling
        rays_o_flat = self.rays_o.view(-1, 3)
        rays_d_flat = self.rays_d.view(-1, 3)
        rgb_flat = self.images.view(-1, 3)
        
        return rays_o_flat[indices], rays_d_flat[indices], rgb_flat[indices]

    def get_single_image_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Samples a batch of rays entirely from a single randomly selected image.
        This is necessary for scale-invariant depth loss which must solve scale/shift per image.
        Outputs: rays_o [B,3], rays_d [B,3], rgb [B,3], depth [B] (if available, else None)
        """
        N, H, W, _ = self.rays_o.shape
        
        img_idx = torch.randint(0, N, (1,), device=self.device).item()
        
        # Random pixels within the chosen image
        indices = torch.randint(0, H * W, (batch_size,), device=self.device)
        
        rays_o_flat = self.rays_o[img_idx].view(-1, 3)
        rays_d_flat = self.rays_d[img_idx].view(-1, 3)
        rgb_flat = self.images[img_idx].view(-1, 3)
        
        depth_flat = None
        if self.depths is not None:
            depth_flat = self.depths[img_idx].view(-1)[indices]
            
        return rays_o_flat[indices], rays_d_flat[indices], rgb_flat[indices], depth_flat
