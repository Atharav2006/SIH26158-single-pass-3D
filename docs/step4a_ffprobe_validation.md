# Step 4A: FFprobe Path Validation Report

This report documents the validation and troubleshooting of the FFmpeg and FFprobe binary locations and PATH availability, resolving why `ffprobe` was initially not recognized by PowerShell.

---

## 1. Executive Finding (PATH Status)

### Cause of Error
The environment variable addition of `D:\SIH26158\tools\ffmpeg\bin` was successfully committed to the Windows registry (as a permanent User environment variable) in Step 2. However, the current active PowerShell session/terminal is **stale** because it was opened *before* the PATH variable update. In Windows, active shells do not automatically hot-reload registry environment changes.

### Solutions / Workarounds
1. **New Terminal (Recommended)**: Launching a new PowerShell window or terminal session will correctly load the updated environment variables.
2. **Current Session Reload**: To avoid restarting the terminal, the PATH environment variable was dynamically reloaded in the current active session using the following command:
   ```powershell
   $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "User") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "Machine")
   ```

---

## 2. Binary Verification via Absolute Paths

Both executable files were validated to exist and execute correctly using their absolute target installation paths.

### 2.1 FFmpeg
- **Path**: `D:\SIH26158\tools\ffmpeg\bin\ffmpeg.exe`
- **Execution Command**:
  ```powershell
  & "D:\SIH26158\tools\ffmpeg\bin\ffmpeg.exe" -version
  ```
- **Output Result**:
  ```text
  ffmpeg version 9.0.1-essentials_build-www.gyan.dev Copyright (c) 2000-2026 the FFmpeg developers
  built with gcc 16.1.0 (Rev2, Built by MSYS2 project)
  ```

### 2.2 FFprobe
- **Path**: `D:\SIH26158\tools\ffmpeg\bin\ffprobe.exe`
- **Execution Command**:
  ```powershell
  & "D:\SIH26158\tools\ffmpeg\bin\ffprobe.exe" -version
  ```
- **Output Result**:
  ```text
  ffprobe version 9.0.1-essentials_build-www.gyan.dev Copyright (c) 2007-2026 the FFmpeg developers
  built with gcc 16.1.0 (Rev2, Built by MSYS2 project)
  ```

---

## 3. Metadata Verification Result

We verified the synthetic controlled video using the absolute path to the `ffprobe` executable.

- **Execution Command**:
  ```powershell
  & "D:\SIH26158\tools\ffmpeg\bin\ffprobe.exe" -v error -select_streams v:0 -show_entries stream=codec_name,width,height,r_frame_rate,duration,nb_frames -of json data/samples/controlled_test/test_video.mp4
  ```

- **Output Result**:
  ```json
  {
      "programs": [
  
      ],
      "stream_groups": [
  
      ],
      "streams": [
          {
              "codec_name": "mpeg4",
              "width": 1920,
              "height": 1080,
              "r_frame_rate": "30/1",
              "duration": "10.000000",
              "nb_frames": "300"
          }
      ]
  }
  ```

This output successfully confirms that the test video matches the ground-truth specification:
* **Duration**: 10.0 seconds
* **Width**: 1920 pixels
* **Height**: 1080 pixels
* **Frame Rate**: 30 FPS (`30/1`)
* **Frame Count**: 300 frames
* **Codec**: `mpeg4`
