# Smart India Hackathon 2026: Problem Statement 26158

## Title
**Single-Pass Drone Video to Accurate 3D Model Generation System**

## Description
The goal of this project is to develop an automated system that can take a single-pass video recorded by a drone and reconstruct an accurate 3D model of the scene. Traditional 3D reconstruction pipelines (like standard Structure-from-Motion and Multi-View Stereo) require highly overlapping, multi-angle photos and long processing times. This project focuses on producing high-quality 3D assets quickly and efficiently from a single, continuous video flight path.

## Key Challenges & Objectives
1. **Single-Pass Constraints**: The drone flies over a target area only once. The baseline reconstruction must handle limited parallax, sequential frame redundancy, and linear flight paths.
2. **Video Ingestion & Preprocessing**: Robust ingestion of various drone video formats, removing motion blur, lens distortion, and stabilization artifacts.
3. **Keyframe Selection**: Filtering out redundant and blurry frames to choose optimal keyframes that maximize baseline coverage and minimize processing overhead.
4. **Camera Pose Estimation**: Estimating precise camera trajectories (poses) from sequential frames, either using traditional SfM (e.g., COLMAP) or deep learning-based visual odometry.
5. **Depth & 3D Reconstruction**: Generating dense 3D point clouds and meshes using modern techniques such as Multi-View Stereo (MVS), Neural Radiance Fields (NeRF), or 3D Gaussian Splatting (3DGS).
6. **Efficiency**: Streamlining computation to minimize processing time so that 3D models can be generated as quickly as possible.
