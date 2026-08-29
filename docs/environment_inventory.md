# Environment Inventory

This document lists the hardware and software specifications of the current development environment for **SIH Problem Statement 26158: Single-Pass Drone Video to Accurate 3D Model Generation System**.

## Hardware Information

| Parameter | Value |
| :--- | :--- |
| **Operating System** | Microsoft Windows 11 Home Single Language (Version: 10.0.26200, 64-bit) |
| **CPU** | 12th Gen Intel(R) Core(TM) i5-12500H (12 Cores, 16 Logical Processors) |
| **RAM** | 16 GB Total Physical Memory (17,179,869,184 bytes, 2 channels) |
| **GPU** | NVIDIA GeForce RTX 3050 Laptop GPU |
| **GPU VRAM** | 4096 MiB (4 GB) |
| **NVIDIA Driver Version** | 581.95 |
| **CUDA Capability (GPU)** | CUDA 13.0 (Reported by `nvidia-smi`) |
| **Disk Space (C: Drive)** | ~19.21 GB Free (336.32 GB Used) |
| **Disk Space (D: Drive)** | ~84.05 GB Free (13.60 GB Used) |

## Software & Tools Installed

| Tool / Dependency | Command / Import Check Status | Version |
| :--- | :--- | :--- |
| **Python** | `python --version` | 3.10.0 |
| **pip** | `pip --version` | 25.3 |
| **Git** | `git --version` | 2.45.1.windows.1 |
| **FFmpeg** | `ffmpeg -version` | *Not found in PATH* |
| **CMake** | `cmake --version` | *Not found in PATH* |
| **COLMAP** | `colmap -h` | *Not found in PATH* |
| **OpenCV** | `import cv2` | 4.10.0 |
| **PyTorch** | `import torch` | 2.10.0+cpu (CUDA available: False) |
| **Open3D** | `import open3d` | *ModuleNotFoundError* |
| **CUDA Toolkit (nvcc)** | `nvcc --version` | *Not found in PATH* |
