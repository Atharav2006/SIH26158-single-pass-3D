# Setup Inventory Before Installation

This document records the baseline environment and hardware inventory for the **Single-Pass Drone Video to Accurate 3D Model Generation System** (SIH26158) before any new installation.

---

## 1. Hardware Specifications

| Component | Specification / Observed Value |
| :--- | :--- |
| **CPU** | 12th Gen Intel(R) Core(TM) i5-12500H (12 Cores, 16 Logical Processors) |
| **RAM** | 16 GB Total Physical Memory |
| **GPU** | NVIDIA GeForce RTX 3050 Laptop GPU |
| **GPU VRAM** | 4096 MiB (4 GB) |
| **NVIDIA Driver Version** | 581.95 |
| **CUDA Version (Driver)** | 13.0 |
| **Disk Space (C: Drive)** | ~20.54 GB free |
| **Disk Space (D: Drive)** | ~79.16 GB free |

---

## 2. Python Environment (`D:\SIH26158\env\sih26158`)

| Dependency | Target / Observed Status | Version |
| :--- | :--- | :--- |
| **Python** | `READY` | 3.10.0 |
| **Python Executable** | `D:\SIH26158\env\sih26158\Scripts\python.exe` | N/A |
| **Virtual Env Status** | Active / Target Venv | N/A |
| **PyTorch (with CUDA)** | `READY` | 2.12.0+cu130 |
| **PyTorch CUDA Build** | `READY` | 13.0 |
| **CUDA Available** | `True` | N/A |
| **NumPy** | `READY` | 2.2.6 |
| **OpenCV** | `READY` | 5.0.0 |
| **pytest** | `READY` | 9.1.1 |

---

## 3. External Tools & Packages (Before Installation)

| Tool / Library | Status | Version | Path / Executable |
| :--- | :--- | :--- | :--- |
| **FFmpeg** | `NOT INSTALLED` | N/A | Not found in system PATH |
| **FFprobe** | `NOT INSTALLED` | N/A | Not found in system PATH |
| **CMake** | `NOT INSTALLED` | N/A | Not found in system PATH |
| **COLMAP** | `NOT INSTALLED` | N/A | Not found in system PATH |
| **Open3D** | `NOT INSTALLED` | N/A | ModuleNotFoundError in virtual env |

---

## 4. Observations & Baseline Conclusions

- The virtual environment at `D:\SIH26158\env\sih26158` contains the correctly compiled PyTorch CUDA release (version `2.12.0+cu130`) matching the GPU's CUDA 13.0 driver.
- The standard system PATH does not expose any external executables (FFmpeg, FFprobe, CMake, COLMAP).
- The environment requires manual installation of FFmpeg, CMake, COLMAP, and Open3D into their respective project-local directories under `D:\SIH26158`.
