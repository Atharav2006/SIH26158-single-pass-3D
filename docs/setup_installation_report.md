# Setup and Installation Report (Step 2 - Rigorous Verification)

This report documents the installation and verification of external tools, libraries, and runtime variables for the **Single-Pass Drone Video to Accurate 3D Model Generation System** (SIH26158).

---

## 1. Machine & Operating System Specifications

- **Operating System**: Microsoft Windows 11 Home Single Language (64-bit, Build 10.0.26200)
- **CPU**: 12th Gen Intel(R) Core(TM) i5-12500H (12 Cores, 16 Logical Processors)
- **RAM**: 16 GB Total Physical Memory
- **GPU**: NVIDIA GeForce RTX 3050 Laptop GPU
- **GPU VRAM**: 4096 MiB (4 GB)
- **NVIDIA Driver Version**: 581.95
- **CUDA Version (Driver)**: 13.0
- **Available Disk Space (Post-Installation)**:
  - **C: Drive**: ~20.45 GB Free
  - **D: Drive**: ~77.77 GB Free

---

## 2. Python Virtual Environment Details

- **Virtual Environment Directory**: `D:\SIH26158\env\sih26158`
- **Python Executable**: `D:\SIH26158\env\sih26158\Scripts\python.exe`
- **Python Version**: `3.10.0`
- **Package Manager**: `pip` version `26.2.1`

### Installed Packages Summary
- **PyTorch**: `2.12.0+cu130`
- **NumPy**: `2.2.6`
- **OpenCV**: `5.0.0`
- **pytest**: `9.1.1`
- **Open3D**: `0.19.0`

### CUDA & PyTorch Validation
- **CUDA Availability in PyTorch**: `True`
- **PyTorch CUDA Build Version**: `13.0`
- **CUDA Matrix Multiplication Test**: `PASS`
  - *Test Command*:
    ```powershell
    python -c "import torch; x=torch.randn((2048,2048),device='cuda'); y=x@x; torch.cuda.synchronize(); print('CUDA computation: PASS')"
    ```
  - *Result*: Output returned `CUDA computation: PASS` successfully.

---

## 3. Installed Components & Paths

| Tool | Version | Source URL | Installation Directory | Executable Path |
| :--- | :--- | :--- | :--- | :--- |
| **FFmpeg** | `9.0.1-essentials` | [gyan.dev](https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip) | `D:\SIH26158\tools\ffmpeg` | `D:\SIH26158\tools\ffmpeg\bin\ffmpeg.exe` |
| **FFprobe** | `9.0.1-essentials` | [gyan.dev](https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip) | `D:\SIH26158\tools\ffmpeg` | `D:\SIH26158\tools\ffmpeg\bin\ffprobe.exe` |
| **CMake** | `4.4.3` | [Kitware/CMake GitHub](https://github.com/Kitware/CMake/releases/download/v4.4.3/cmake-4.4.3-windows-x86_64.zip) | `D:\SIH26158\tools\cmake` | `D:\SIH26158\tools\cmake\bin\cmake.exe` |
| **COLMAP** | `4.1.1` | [colmap/colmap GitHub](https://github.com/colmap/colmap/releases/download/4.1.1/colmap-x64-windows-cuda.zip) | `D:\SIH26158\tools\colmap` | `D:\SIH26158\tools\colmap\colmap.exe` |
| **Open3D** | `0.19.0` | PyPI | `D:\SIH26158\env\sih26158\Lib\site-packages\open3d` | N/A (Python Library) |

---

## 4. PATH Modifications

The User-level environment variables were permanently updated to include:
- `D:\SIH26158\tools\ffmpeg\bin`
- `D:\SIH26158\tools\cmake\bin`
- `D:\SIH26158\tools\colmap`

No global Machine-level environment variables or administrative privileges were modified, isolating the setup configurations to the developer's user workspace.

---

## 5. Verification Commands and Results

All verification checks were run inside the `D:\SIH26158\env\sih26158` virtual environment.

### 5.1 verify_environment.py Verification
```powershell
python scripts/verify_environment.py
```
**Output:**
```text
============================================================
SIH26158: Environment Verification (Step 2 - Rigorous)
============================================================

--- Required Dependencies ---
[READY        ] Python: Python version 3.10.0
[READY        ] pip: pip version 26.2.1
[READY        ] Git: git version 2.45.1.windows.1
[READY        ] opencv-python (cv2): cv2 version 5.0.0
PyTorch Version: 2.12.0+cu130
PyTorch CUDA Build: 13.0
CUDA Availability: True
GPU Name: NVIDIA GeForce RTX 3050 Laptop GPU
GPU Total Memory: 4.00 GB
Running CUDA matrix multiplication test...
CUDA matrix multiplication test passed successfully!
[READY        ] PyTorch (torch): PyTorch is ready on GPU NVIDIA GeForce RTX 3050 Laptop GPU (version 2.12.0+cu130)
[READY        ] pytest: pytest version 9.1.1

--- Future Pipeline Dependencies (Informational) ---
[READY        ] open3d: open3d version 0.19.0
[READY        ] FFmpeg: executable found in PATH
[READY        ] CMake: executable found in PATH
[READY        ] COLMAP: executable found in PATH
============================================================
VERIFICATION RESULT: READY (All required dependencies met and CUDA validated)
```

### 5.2 Unit/Structure Tests
```powershell
pytest -q
```
**Output:**
```text
.....                                                                    [100%]
5 passed in 0.06s
```

### 5.3 Open3D Point Cloud and Geometry Operation
```powershell
python -c "import open3d as o3d, numpy as np; pcd=o3d.geometry.PointCloud(); pcd.points=o3d.utility.Vector3dVector(np.random.rand(10,3)); print(pcd.get_center())"
```
**Output:**
```text
[0.47916181 0.5089129  0.55581326]
```

### 5.4 FFmpeg and COLMAP Verification
```powershell
ffmpeg -version
colmap version
```
**Output:**
```text
ffmpeg version 9.0.1-essentials_build-www.gyan.dev Copyright (c) 2000-2026 the FFmpeg developers
COLMAP 4.1.1 (Commit a0d785f on 2026-07-17 with CUDA)
```

---

## 6. Installation Directories & Storage Policies

All heavy binaries, datasets, caches, and models are isolated outside of the Git repository to keep the repository lightweight and prevent unintended commits of binaries or large model checkpoints.

Directories established under `D:\SIH26158\`:
- `env\sih26158\`: Python virtual environment.
- `tools\`: Contains subfolders for `ffmpeg`, `cmake`, and `colmap`.
- `models\`: Reserved directory for pretrained weights/models.
- `datasets\`: Reserved directory for video and telemetry datasets.
- `cache\`: Intermediate outputs/processing cache directory.
- `pip-cache\`: Offline installation cache.
- `temp\`: Workspace for extraction and file setup.

---

## 7. Rollback & Uninstallation Instructions

In case a clean rollback is required, execute the following steps:

1. **Remove Packages from Python Virtual Environment**:
   Run:
   ```powershell
   D:\SIH26158\env\sih26158\Scripts\pip.exe uninstall open3d
   ```
2. **Delete Binaries and Tooling Folders**:
   Permanently delete the folders created in the tools directory:
   - Remove `D:\SIH26158\tools\ffmpeg`
   - Remove `D:\SIH26158\tools\cmake`
   - Remove `D:\SIH26158\tools\colmap`
3. **Revert PATH Environment Variables**:
   Retrieve your current User PATH:
   ```powershell
   $userPath = [System.Environment]::GetEnvironmentVariable("Path", "User")
   ```
   Remove the entries for `D:\SIH26158\tools\ffmpeg\bin`, `D:\SIH26158\tools\cmake\bin`, and `D:\SIH26158\tools\colmap` from the string, then save it:
   ```powershell
   [System.Environment]::SetEnvironmentVariable("Path", $cleanedUserPath, "User")
   ```

---

## 8. Warnings and Unresolved Issues

- **None**. All package conflicts were bypassed by selecting `open3d-0.19.0` (which natively supports `numpy 2.x`), and all validation tests completed successfully.
