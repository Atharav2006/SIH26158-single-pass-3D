# Step 4A: Controlled Video Ingestion and Metadata Verification

This document summarizes the execution and verification of **Step 4A** (creating a controlled video ingestion test asset and validating FFmpeg/ffprobe integration).

---

## 1. Test Video Properties

A synthetic, controlled video asset was programmatically generated to serve as a baseline verification input. 
- **File Path**: [test_video.mp4](file:///d:/SIH26158-single-pass-3D/data/samples/controlled_test/test_video.mp4)
- **Ground Truth Config**: [ground_truth.json](file:///d:/SIH26158-single-pass-3D/data/samples/controlled_test/ground_truth.json)
- **Features Included**:
  - A static dark gray background with a grid pattern for noise control.
  - A blue-green circle moving diagonally across the screen to simulate motion.
  - Rendered frame numbers ("Frame: X") burned into each frame.
  - Rendered elapsed timestamps ("Time: X.XXs") burned into each frame.

---

## 2. Tool Environment

- **FFmpeg/ffprobe Version**: `9.0.1-essentials_build-www.gyan.dev`
- **Python Virtual Env Interpreter**: `D:\SIH26158\env\sih26158\Scripts\python.exe`
- **Packages used**: OpenCV (cv2 version `5.0.0`), NumPy (version `2.2.6`), pytest (version `9.1.1`).

---

## 3. Raw ffprobe Command & Output

**Command:**
```powershell
ffprobe -v error -select_streams v:0 -show_entries stream=codec_name,width,height,r_frame_rate,duration,nb_frames -of json data/samples/controlled_test/test_video.mp4
```

**JSON Output:**
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

---

## 4. Extracted Metadata Output

The python ingestion module [video_metadata.py](file:///d:/SIH26158-single-pass-3D/src/ingestion/video_metadata.py) wraps `ffprobe` subprocess calls and exposes a structured JSON interface.

**Command:**
```powershell
python -m src.ingestion.video_metadata data/samples/controlled_test/test_video.mp4
```

**JSON Output:**
```json
{
    "duration": 10.0,
    "width": 1920,
    "height": 1080,
    "average_frame_rate": 30.0,
    "codec": "mpeg4",
    "frame_count": 300
}
```

---

## 5. Expected vs. Actual Comparison

| Parameter | Expected (Ground Truth) | Actual (Extracted) | Status |
| :--- | :--- | :--- | :--- |
| **Duration** | `10.0 s` | `10.0 s` | **MATCH** |
| **FPS (Frame Rate)** | `30.0` | `30.0` | **MATCH** |
| **Frame Count** | `300` | `300` | **MATCH** |
| **Width** | `1920` | `1920` | **MATCH** |
| **Height** | `1080` | `1080` | **MATCH** |
| **Codec** | `mpeg4` (implicit target) | `mpeg4` | **MATCH** |

---

## 6. Test Results

The metadata extraction was validated via a dedicated test file: [test_video_metadata.py](file:///d:/SIH26158-single-pass-3D/tests/unit/test_video_metadata.py).

- **Execution Command**:
  ```powershell
  pytest -v tests/unit/test_video_metadata.py
  ```
- **Results**:
  - `tests/unit/test_video_metadata.py::test_video_metadata_extraction` **PASSED** (100% assertions satisfied, matching with strict tolerances for duration/FPS).
