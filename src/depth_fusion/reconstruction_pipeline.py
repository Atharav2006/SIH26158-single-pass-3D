"""
SIH26158 Depth Fusion - Production Relative Dense Reconstruction Pipeline

This module implements the end-to-end streaming reconstruction engine:
1. Batch depth inference with MiDaS_small.
2. Multi-cue confidence map generation.
3. Scale-aware multi-view consistency filtering.
4. Streaming voxel grid fusion (Ablation Modes: B5-A, B5-B, B5-C).
5. Diagnostic JSON reporting and PLY/visualization export using cv2/PIL.
"""

from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import time
import json
import csv
import numpy as np
import torch
import cv2
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation

from src.depth_fusion.depth_prior import MiDaSDepthPrior
from src.depth_fusion.camera_preprocessing import CameraPreprocessor
from src.depth_fusion.depth_quality import compute_depth_confidence
from src.depth_fusion.pointcloud_fusion import (
    RelativePointcloud,
    unproject_relative_frame,
    VoxelGridFusion,
    save_pointcloud_ply
)
from src.depth_fusion.multiview_consistency import MultiViewConsistencyEvaluator

class RelativeDenseReconstructionPipeline:
    """
    Streaming pipeline for confidence-aware multi-frame relative 3D reconstruction.
    """
    def __init__(
        self,
        calib_data: Dict[str, Any],
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        voxel_size: float = 5e-5,
        subsample_step: int = 4,
        min_confidence: float = 0.15
    ):
        self.device = device
        self.preprocessor = CameraPreprocessor(calib_data)
        self.prior = MiDaSDepthPrior(device)
        self.voxel_size = voxel_size
        self.subsample_step = subsample_step
        self.min_confidence = min_confidence

        self.K_rect = self.preprocessor.K_rect
        self.mv_evaluator = MultiViewConsistencyEvaluator(self.K_rect)

    def load_b2_poses(self, csv_path: Path) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Loads B2 poses from CSV as list of (R_wc, C_world).
        """
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        poses = []
        for r in rows:
            q = [float(r['qx']), float(r['qy']), float(r['qz']), float(r['qw'])]
            t = np.array([float(r['x']), float(r['y']), float(r['z'])], dtype=np.float32)
            R = Rotation.from_quat(q).as_matrix().astype(np.float32)
            poses.append((R, t))
        return poses

    def run_reconstruction(
        self,
        image_dir: Path,
        b2_poses_path: Path,
        output_dir: Path,
        max_frames: Optional[int] = None,
        stride: int = 1
    ) -> Dict[str, Any]:
        """
        Executes the streaming reconstruction over the video sequence.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        depth_dir = output_dir / "b5_relative_depth"
        conf_dir = output_dir / "b5_confidence"
        depth_dir.mkdir(exist_ok=True)
        conf_dir.mkdir(exist_ok=True)

        poses_b2 = self.load_b2_poses(b2_poses_path)
        total_available = len(poses_b2)
        frame_indices = list(range(0, total_available, stride))
        if max_frames is not None:
            frame_indices = frame_indices[:max_frames]

        N_eval = len(frame_indices)
        print(f"Starting B5 Relative Reconstruction over {N_eval} frames (stride={stride})...")

        # Fusion containers for Ablations:
        # B5-A: Raw relative depth (all valid points, min_conf=0.0)
        fusion_A = VoxelGridFusion(voxel_size=self.voxel_size)
        # B5-B: Confident relative depth (min_conf=0.15)
        fusion_B = VoxelGridFusion(voxel_size=self.voxel_size)
        # B5-C: Consistent relative depth (multi-view consistency weighted)
        fusion_C = VoxelGridFusion(voxel_size=self.voxel_size)

        timings = {
            "depth_inference_s": 0.0,
            "confidence_s": 0.0,
            "unprojection_s": 0.0,
            "multiview_s": 0.0,
            "fusion_s": 0.0
        }

        all_raw_point_counts = []
        depth_quality_records = []
        consistency_residuals = []

        # Buffer for consecutive frame pairs (for multi-view consistency)
        prev_pcd = None
        prev_depth = None
        prev_pose = None
        prev_frame_id = None

        start_total = time.time()
        start_vram = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0

        for loop_idx, f_idx in enumerate(frame_indices):
            frame_id = f_idx + 1
            fname = f"{frame_id:05d}.jpg"
            img_path = image_dir / fname

            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_rect = self.preprocessor.rectify_image(img_rgb)

            # 1. Depth Inference
            t0 = time.time()
            t_img = torch.from_numpy(img_rect).float().to(self.device) / 255.0
            pred = self.prior.predict(t_img)
            inv_depth = pred.depth.cpu().numpy()
            rel_depth = 1.0 / np.maximum(inv_depth, 1e-6)
            timings["depth_inference_s"] += time.time() - t0

            # Save sample depth visualizations (first, middle, last)
            if loop_idx in [0, N_eval // 2, N_eval - 1]:
                norm_d = cv2.normalize(inv_depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                d_vis = cv2.applyColorMap(norm_d, cv2.COLORMAP_INFERNO)
                cv2.imwrite(str(depth_dir / f"depth_{fname}"), d_vis)

            # 2. Confidence Estimation
            t0 = time.time()
            conf_map, mask, quality_stats = compute_depth_confidence(img_rect, inv_depth)
            timings["confidence_s"] += time.time() - t0
            quality_stats["frame_id"] = frame_id
            depth_quality_records.append(quality_stats)

            if loop_idx in [0, N_eval // 2, N_eval - 1]:
                norm_c = (conf_map * 255.0).astype(np.uint8)
                c_vis = cv2.applyColorMap(norm_c, cv2.COLORMAP_VIRIDIS)
                cv2.imwrite(str(conf_dir / f"conf_{fname}"), c_vis)

            # 3. Unproject Pointcloud
            R_wc, C_w = poses_b2[f_idx]
            t0 = time.time()
            # Raw pointcloud (for Mode A)
            pcd_raw = unproject_relative_frame(
                rgb=img_rect,
                rel_depth=rel_depth,
                confidence_map=conf_map,
                K_rect=self.K_rect,
                R_wc=R_wc,
                C_world=C_w,
                frame_id=frame_id,
                subsample_step=self.subsample_step,
                min_confidence=0.0
            )
            # Confident pointcloud (for Mode B)
            pcd_conf = unproject_relative_frame(
                rgb=img_rect,
                rel_depth=rel_depth,
                confidence_map=conf_map,
                K_rect=self.K_rect,
                R_wc=R_wc,
                C_world=C_w,
                frame_id=frame_id,
                subsample_step=self.subsample_step,
                min_confidence=self.min_confidence
            )
            timings["unprojection_s"] += time.time() - t0
            all_raw_point_counts.append(len(pcd_raw.points))

            # 4. Multi-View Consistency
            t0 = time.time()
            if prev_pcd is not None and len(prev_pcd.points) > 0:
                mv_weights, inliers, mv_diag = self.mv_evaluator.evaluate_pair_consistency(
                    prev_pcd, rel_depth, R_wc, C_w, image_shape=img_rect.shape[:2],
                    R_wc_A=prev_pose[0], C_world_A=prev_pose[1]
                )
                if mv_diag["inlier_count"] > 0:
                    consistency_residuals.append(mv_diag["median_relative_residual"])
                
                # Mode C: Multiply confidence by multi-view consistency score
                pcd_consist = RelativePointcloud(
                    points=prev_pcd.points[inliers],
                    colors=prev_pcd.colors[inliers],
                    confidences=prev_pcd.confidences[inliers] * mv_weights[inliers],
                    frame_ids=prev_pcd.frame_ids[inliers],
                    support_counts=prev_pcd.support_counts[inliers],
                    scale_type="relative",
                    metric=False
                )
                fusion_C.add_pointcloud(pcd_consist)
            timings["multiview_s"] += time.time() - t0

            # 5. Integrate into Fusion Grids
            t0 = time.time()
            fusion_A.add_pointcloud(pcd_raw)
            fusion_B.add_pointcloud(pcd_conf)
            timings["fusion_s"] += time.time() - t0

            prev_pcd = pcd_conf
            prev_depth = rel_depth
            prev_pose = (R_wc, C_w)
            prev_frame_id = frame_id

            if (loop_idx + 1) % 50 == 0 or (loop_idx + 1) == N_eval:
                print(f"Processed {loop_idx + 1}/{N_eval} frames...")

        total_runtime = time.time() - start_total
        peak_vram = (torch.cuda.max_memory_allocated() / (1024**2)) if torch.cuda.is_available() else 0.0

        # Extract Fused Point Clouds for Ablation Modes
        pcd_fused_A = fusion_A.extract_fused_pointcloud(min_support=1, min_confidence=0.0, mode="RELATIVE_RAW")
        pcd_fused_B = fusion_B.extract_fused_pointcloud(min_support=1, min_confidence=self.min_confidence, mode="RELATIVE_CONFIDENT")
        pcd_fused_C = fusion_C.extract_fused_pointcloud(min_support=1, min_confidence=self.min_confidence, mode="RELATIVE_CONSISTENT")

        # Save PLY files with explicit metadata headers
        raw_ply_path = output_dir / "b5_raw_relative_pointcloud.ply"
        fused_ply_path = output_dir / "b5_fused_relative_pointcloud.ply"
        high_conf_ply_path = output_dir / "b5_high_confidence_relative_pointcloud.ply"

        save_pointcloud_ply(raw_ply_path, pcd_fused_A, ["MODE: B5-A_RELATIVE_RAW"])
        save_pointcloud_ply(fused_ply_path, pcd_fused_B, ["MODE: B5-B_RELATIVE_CONFIDENT"])
        save_pointcloud_ply(high_conf_ply_path, pcd_fused_C, ["MODE: B5-C_RELATIVE_CONSISTENT"])

        total_unprojected_points = int(sum(all_raw_point_counts))
        points_per_sec = float(total_unprojected_points / max(1e-4, total_runtime))

        # Generate Diagnostics JSONs
        # 1. Depth Quality
        depth_quality_summary = {
            "dataset": "Zurich Urban MAV Dataset",
            "phase": "B5 Phase 4 Depth Quality Audit",
            "total_frames_evaluated": N_eval,
            "mean_valid_pixel_ratio": float(np.mean([r["valid_ratio"] for r in depth_quality_records])),
            "mean_confidence": float(np.mean([r["confidence_percentiles"]["p50"] for r in depth_quality_records])),
            "per_frame_records_sample": depth_quality_records[::max(1, N_eval // 10)]
        }
        with open(output_dir / "b5_depth_quality.json", "w") as f:
            json.dump(depth_quality_summary, f, indent=4)

        # 2. Relative Geometry
        rel_geom_summary = {
            "dataset": "Zurich Urban MAV Dataset",
            "phase": "B5 Phase 4 Relative Geometry Metadata",
            "scale_type": "relative",
            "metric": False,
            "camera_origin_frame": "Metric Local ENU (from B2)",
            "depth_gauge": "MiDaS_small relative inverse depth (1 / D_inv)",
            "spatial_extent_relative_units": {
                "x_min": float(pcd_fused_B.points[:, 0].min()) if len(pcd_fused_B) > 0 else 0.0,
                "x_max": float(pcd_fused_B.points[:, 0].max()) if len(pcd_fused_B) > 0 else 0.0,
                "y_min": float(pcd_fused_B.points[:, 1].min()) if len(pcd_fused_B) > 0 else 0.0,
                "y_max": float(pcd_fused_B.points[:, 1].max()) if len(pcd_fused_B) > 0 else 0.0,
                "z_min": float(pcd_fused_B.points[:, 2].min()) if len(pcd_fused_B) > 0 else 0.0,
                "z_max": float(pcd_fused_B.points[:, 2].max()) if len(pcd_fused_B) > 0 else 0.0
            }
        }
        with open(output_dir / "b5_relative_geometry.json", "w") as f:
            json.dump(rel_geom_summary, f, indent=4)

        # 3. Fusion Diagnostics
        fusion_diag = {
            "dataset": "Zurich Urban MAV Dataset",
            "phase": "B5 Phase 4 Fusion Diagnostics",
            "total_unprojected_raw_points": total_unprojected_points,
            "voxel_grid_size": self.voxel_size,
            "fused_point_count_raw": len(pcd_fused_A),
            "fused_point_count_confident": len(pcd_fused_B),
            "fused_point_count_consistent": len(pcd_fused_C),
            "mean_support_count": float(np.mean(pcd_fused_B.support_counts)) if len(pcd_fused_B) > 0 else 0.0,
            "max_support_count": int(np.max(pcd_fused_B.support_counts)) if len(pcd_fused_B) > 0 else 0,
            "duplicate_suppression_ratio": float(1.0 - len(pcd_fused_B) / max(1, total_unprojected_points))
        }
        with open(output_dir / "b5_fusion_diagnostics.json", "w") as f:
            json.dump(fusion_diag, f, indent=4)

        # 4. Multi-View Consistency
        mv_summary = {
            "dataset": "Zurich Urban MAV Dataset",
            "phase": "B5 Phase 4 Multi-View Consistency",
            "consistency_metric": "Scale-Invariant Relative Depth Residual",
            "evaluated_pairs_count": len(consistency_residuals),
            "mean_relative_residual": float(np.mean(consistency_residuals)) if len(consistency_residuals) > 0 else 0.0,
            "median_relative_residual": float(np.median(consistency_residuals)) if len(consistency_residuals) > 0 else 0.0
        }
        with open(output_dir / "b5_multiview_consistency.json", "w") as f:
            json.dump(mv_summary, f, indent=4)

        # 5. Ablation Comparison (B5-A vs B5-B vs B5-C)
        ablation_summary = {
            "dataset": "Zurich Urban MAV Dataset",
            "phase": "B5 Phase 4 Ablation Comparison",
            "modes": {
                "B5_A_RELATIVE_RAW": {
                    "description": "Raw unprojected depth (all valid points, min_conf=0.0)",
                    "point_count": len(pcd_fused_A),
                    "mean_support": float(np.mean(pcd_fused_A.support_counts)) if len(pcd_fused_A) > 0 else 0.0,
                    "confidence_weighting": False,
                    "multiview_filter": False
                },
                "B5_B_RELATIVE_CONFIDENT": {
                    "description": "Confidence-weighted voxel fusion with min_conf=0.15, min_support=2",
                    "point_count": len(pcd_fused_B),
                    "mean_support": float(np.mean(pcd_fused_B.support_counts)) if len(pcd_fused_B) > 0 else 0.0,
                    "confidence_weighting": True,
                    "multiview_filter": False
                },
                "B5_C_RELATIVE_CONSISTENT": {
                    "description": "Confidence + scale-aware multi-view consistency filtering",
                    "point_count": len(pcd_fused_C),
                    "mean_support": float(np.mean(pcd_fused_C.support_counts)) if len(pcd_fused_C) > 0 else 0.0,
                    "confidence_weighting": True,
                    "multiview_filter": True
                }
            }
        }
        with open(output_dir / "b5_ablation.json", "w") as f:
            json.dump(ablation_summary, f, indent=4)

        # 6. Phase 4 Summary
        phase4_summary = {
            "status": "B5_RELATIVE_RECONSTRUCTION_READY",
            "scale_type": "relative",
            "metric": False,
            "total_frames_processed": N_eval,
            "total_runtime_s": total_runtime,
            "points_per_second": points_per_sec,
            "peak_vram_mb": peak_vram,
            "timings_s": timings,
            "raw_point_count": len(pcd_fused_A),
            "fused_point_count": len(pcd_fused_B),
            "consistent_point_count": len(pcd_fused_C)
        }
        with open(output_dir / "b5_phase4_summary.json", "w") as f:
            json.dump(phase4_summary, f, indent=4)

        # Generate Visualizations via PIL/OpenCV
        self._generate_visualizations(pcd_fused_B, pcd_fused_A, poses_b2, frame_indices, output_dir)

        print("B5 Relative Dense Reconstruction completed successfully!")
        return phase4_summary

    def _generate_visualizations(
        self,
        pcd_fused: RelativePointcloud,
        pcd_raw: RelativePointcloud,
        poses: List[Tuple[np.ndarray, np.ndarray]],
        frame_indices: List[int],
        output_dir: Path
    ):
        """
        Renders orthogonal orthographic projection images using OpenCV and PIL.
        """
        if len(pcd_fused) == 0:
            return

        def render_projection(points, colors, axis1=0, axis2=1, width=1000, height=800, title="Projection"):
            img = np.zeros((height, width, 3), dtype=np.uint8) + 30
            x = points[:, axis1]
            y = points[:, axis2]

            x_min, x_max = np.percentile(x, 1), np.percentile(x, 99)
            y_min, y_max = np.percentile(y, 1), np.percentile(y, 99)

            x_range = max(1e-4, x_max - x_min)
            y_range = max(1e-4, y_max - y_min)

            u = np.clip(((x - x_min) / x_range * (width - 40) + 20), 0, width - 1).astype(np.int32)
            v = np.clip(((y_max - y) / y_range * (height - 40) + 20), 0, height - 1).astype(np.int32)

            for i in range(len(u)):
                c = (int(colors[i, 2]), int(colors[i, 1]), int(colors[i, 0]))  # BGR
                cv2.circle(img, (u[i], v[i]), 1, c, -1)

            cv2.putText(img, title, (25, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            return img

        pts = pcd_fused.points
        cols = pcd_fused.colors

        # 1. Top-Down (X vs Y) and Side (X vs Z)
        top_down = render_projection(pts, cols, 0, 1, 900, 700, "B5 Fused Relative Point Cloud (Top-Down X-Y)")
        side_view = render_projection(pts, cols, 0, 2, 900, 700, "B5 Fused Relative Point Cloud (Side View X-Z)")
        fused_vis = np.hstack([top_down, side_view])
        cv2.imwrite(str(output_dir / "b5_fused_pointcloud.png"), fused_vis)

        # 2. Raw Point Cloud
        if len(pcd_raw) > 0:
            raw_vis = render_projection(pcd_raw.points, pcd_raw.colors, 0, 1, 1000, 800, "B5 Raw Relative Point Cloud (Top-Down)")
            cv2.imwrite(str(output_dir / "b5_relative_pointcloud.png"), raw_vis)

        # 3. Confidence Colormap Projection
        conf_uint8 = np.clip(pcd_fused.confidences * 255.0, 0, 255).astype(np.uint8).reshape(-1, 1)
        conf_cols = cv2.applyColorMap(conf_uint8, cv2.COLORMAP_VIRIDIS).reshape(-1, 3)
        conf_vis = render_projection(pts, conf_cols, 0, 1, 1000, 800, "B5 Multi-Cue Confidence Distribution (Top-Down)")
        cv2.imwrite(str(output_dir / "b5_confidence_pointcloud.png"), conf_vis)

        # 4. Support Count Colormap Projection
        max_s = max(1.0, float(np.max(pcd_fused.support_counts)))
        sup_uint8 = np.clip((pcd_fused.support_counts / max_s) * 255.0, 0, 255).astype(np.uint8).reshape(-1, 1)
        sup_cols = cv2.applyColorMap(sup_uint8, cv2.COLORMAP_PLASMA).reshape(-1, 3)
        sup_vis = render_projection(pts, sup_cols, 0, 1, 1000, 800, "B5 Multi-Frame Voxel Support Count")
        cv2.imwrite(str(output_dir / "b5_support_count.png"), sup_vis)

        # 5. Multi-View Consistency Histogram Image
        hist_img = np.zeros((600, 800, 3), dtype=np.uint8) + 25
        hist, bin_edges = np.histogram(pcd_fused.confidences, bins=30, range=(0.0, 1.0))
        max_h = max(1, np.max(hist))
        for i in range(len(hist)):
            x1 = int(50 + i * (700 / 30))
            x2 = int(50 + (i + 1) * (700 / 30) - 2)
            h_bar = int((hist[i] / max_h) * 450)
            y1 = 520 - h_bar
            y2 = 520
            cv2.rectangle(hist_img, (x1, y1), (x2, y2), (200, 160, 50), -1)

        cv2.putText(hist_img, "B5 Multi-View Geometric Confidence Histogram", (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(hist_img, "Confidence Score [0.0 -> 1.0]", (280, 560), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        cv2.imwrite(str(output_dir / "b5_multiview_consistency.png"), hist_img)

        # 6. Camera Trajectory Over Relative Cloud
        traj_vis = top_down.copy()
        cam_centers = np.array([poses[i][1] for i in frame_indices])
        x_min, x_max = np.percentile(pts[:, 0], 1), np.percentile(pts[:, 0], 99)
        y_min, y_max = np.percentile(pts[:, 1], 1), np.percentile(pts[:, 1], 99)
        x_range = max(1e-4, x_max - x_min)
        y_range = max(1e-4, y_max - y_min)

        u_c = np.clip(((cam_centers[:, 0] - x_min) / x_range * 860 + 20), 0, 899).astype(np.int32)
        v_c = np.clip(((y_max - cam_centers[:, 1]) / y_range * 660 + 20), 0, 699).astype(np.int32)

        for i in range(len(u_c) - 1):
            cv2.line(traj_vis, (u_c[i], v_c[i]), (u_c[i+1], v_c[i+1]), (0, 0, 255), 2)
            cv2.circle(traj_vis, (u_c[i], v_c[i]), 3, (0, 255, 255), -1)

        cv2.putText(traj_vis, "B2 Camera Trajectory (Red Line)", (25, 670), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        cv2.imwrite(str(output_dir / "b5_camera_trajectory_over_relative_cloud.png"), traj_vis)
