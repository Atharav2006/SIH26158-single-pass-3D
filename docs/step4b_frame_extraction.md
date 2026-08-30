# Step 4B: Reproducible Video Frame Extraction

This document details the architecture, usage, schemas, and verification results for **Step 4B** (reproducible frame extraction with frame-to-timestamp indexing).

---

## 1. Architecture

The frame extraction subsystem is part of the `src/ingestion` package. It bridges external multimedia tooling (FFmpeg/FFprobe) with the pipeline data model:

```
+--------------------+        +--------------------+        +----------------------------+
|  Input Video (.mp4)| -----> |  FrameExtractor    | -----> |  <output_dir>/             |
|  (Drone or Sample) |        |  - Validation      |        |  ├── frames/ (frame_%06d)  |
+--------------------+        |  - FFprobe Meta    |        |  ├── frame_index.csv       |
                              |  - FFmpeg Pipe     |        |  └── extraction_metadata   |
                              +--------------------+        +----------------------------+
```

### Components
- **[frame_extractor.py](file:///d:/SIH26158-single-pass-3D/src/ingestion/frame_extractor.py)**: The core class `FrameExtractor` handles validation, dynamic parameter resolution, FFmpeg filter graph construction (`fps`, `scale`), frame integrity assertion, CSV creation, and JSON metadata compilation.
- **[extract_frames.py](file:///d:/SIH26158-single-pass-3D/pipelines/baseline/extract_frames.py)**: Executable baseline CLI providing command-line arguments and status telemetry.
- **[test_frame_extractor.py](file:///d:/SIH26158-single-pass-3D/tests/unit/test_frame_extractor.py)**: Automated unit test suite verifying frame counting, sequence continuity, monotonicity, timestamp accuracy, and error paths.

---

## 2. Command Usage

### Standard Baseline Command
To extract all frames from a video at source resolution and FPS:
```powershell
python -m pipelines.baseline.extract_frames --input data/samples/controlled_test/test_video.mp4 --output outputs/test_extraction
```

### Custom Extraction FPS & Resizing
To downsample the extraction to 5 FPS and scale frames to 640x360:
```powershell
python -m pipelines.baseline.extract_frames -i data/samples/controlled_test/test_video.mp4 -o outputs/test_extraction_5fps --fps 5.0 --width 640 --height 360
```

### Options
| Flag | Description | Default |
| :--- | :--- | :--- |
| `--input`, `-i` | Path to source video file (Required) | None |
| `--output`, `-o` | Target directory for frames and metadata (Required) | None |
| `--fps` | Extraction frame rate | Keep original source FPS |
| `--width` | Output frame width | Keep original source width |
| `--height` | Output frame height | Keep original source height |
| `--format` | Output image extension (`jpg`, `png`, etc.) | `jpg` |
| `--quality` | JPEG compression quality (1-31 scale, 2 = highest) | `2` |
| `--config`, `-c` | Custom JSON configuration override | `configs/default_config.json` |

---

## 3. Input & Output Structure

```
<output_dir>/
├── frames/
│   ├── frame_000001.jpg
│   ├── frame_000002.jpg
│   ├── frame_000003.jpg
│   └── ...
├── frame_index.csv
└── extraction_metadata.json
```

---

## 4. Frame Index CSV Schema (`frame_index.csv`)

| Column Name | Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `frame_id` | Integer | One-based sequential frame identifier (`1, 2, ...`) | `1` |
| `filename` | String | Filename of extracted image in `frames/` | `frame_000001.jpg` |
| `timestamp_seconds` | Float | Elapsed video time from zero in seconds | `0.0` |
| `source_frame_number`| Integer | Corresponding frame index in the source stream | `1` |
| `width` | Integer | Frame image width in pixels | `1920` |
| `height` | Integer | Frame image height in pixels | `1080` |

---

## 5. Extraction Metadata JSON Schema (`extraction_metadata.json`)

```json
{
    "source_video": "D:\\SIH26158-single-pass-3D\\data\\samples\\controlled_test\\test_video.mp4",
    "source_duration_seconds": 10.0,
    "source_width": 1920,
    "source_height": 1080,
    "source_fps": 30.0,
    "source_codec": "mpeg4",
    "source_frame_count": 300,
    "extraction_fps": 30.0,
    "extracted_frame_count": 300,
    "output_width": 1920,
    "output_height": 1080,
    "output_image_format": "jpg",
    "extraction_timestamp_utc": "2026-08-29T18:46:28.665893+00:00",
    "ffmpeg_version": "ffmpeg version 9.0.1-essentials_build-www.gyan.dev Copyright (c) 2000-2026 the FFmpeg developers",
    "ffprobe_version": "ffprobe version 9.0.1-essentials_build-www.gyan.dev Copyright (c) 2007-2026 the FFmpeg developers"
}
```

---

## 6. Timestamp Semantics

- **Reference Time**: The start of the first video packet corresponds to $t = 0.0$ seconds.
- **Formula**: For frame $k \in \{1, 2, \dots, N\}$, the timestamp is calculated as:
  $$t_k = \frac{k - 1}{\text{effective\_fps}}$$
- **Monotonicity**: $t_k > t_{k-1}$ is guaranteed for all $k > 1$.
- **Mapping to Source**: When downsampling via `--fps`, `source_frame_number` estimates the corresponding temporal frame index in the raw stream via:
  $$\text{source\_frame\_number} = \left\lfloor \frac{k - 1}{\text{effective\_fps}} \times \text{source\_fps} + 0.5 \right\rfloor + 1$$

---

## 7. Error Handling

1. **File Not Found**: Raises `FileNotFoundError` if the video path does not exist.
2. **Invalid Format**: Raises `ValueError` if the file extension is not in `ingestion.allowed_formats`.
3. **Invalid Video Stream**: Raises `ValueError` if FFprobe cannot read valid positive dimensions or FPS.
4. **Invalid Extraction FPS**: Raises `ValueError` if user-specified FPS is $\le 0$.
5. **Subprocess Failure**: Catches `subprocess.CalledProcessError` and raises a descriptive `RuntimeError` including standard error diagnostics from FFmpeg.
6. **Zero Frames Extracted**: Raises `RuntimeError` if FFmpeg exits normally but leaves the target directory empty.
7. **Sequence Discontinuity**: Verifies every frame file against `frame_{idx:06d}.ext`; raises `RuntimeError` if missing indices or gaps are detected.

---

## 8. Verification & Test Results

### Execution Output (10-Second 1920x1080 @ 30 FPS Input Video)
```text
============================================================
SIH26158: Baseline Frame Extraction Pipeline
============================================================
Input Video: data\samples\controlled_test\test_video.mp4
Output Directory: outputs\test_extraction

--- Source Video Metadata ---
  Duration:   10.00 seconds
  Resolution: 1920 x 1080
  Frame Rate: 30.0 FPS
  Codec:      mpeg4
  Frames:     300

--- Running Extraction ---

--- Extraction Summary ---
  Extracted Frames:   300
  Extraction FPS:     30.0 FPS
  Output Resolution:  1920 x 1080
  Elapsed Time:       0.64 s
  Throughput:         466.3 frames/s
  Total Output Size:  17.65 MB
  Frame Index CSV:    outputs\test_extraction\frame_index.csv
  Metadata JSON:      outputs\test_extraction\extraction_metadata.json
============================================================
RESULT: SUCCESS
```

### Test Suite (`pytest -v`)
```text
tests/test_project_structure.py::test_directories_exist PASSED           [ 11%]
tests/test_project_structure.py::test_imports PASSED                     [ 22%]
tests/test_project_structure.py::test_project_version PASSED             [ 33%]
tests/test_project_structure.py::test_config_system PASSED               [ 44%]
tests/test_project_structure.py::test_logging_system PASSED              [ 55%]
tests/unit/test_frame_extractor.py::test_frame_extractor_full_fps PASSED [ 66%]
tests/unit/test_frame_extractor.py::test_frame_extractor_custom_fps_and_resize PASSED [ 77%]
tests/unit/test_frame_extractor.py::test_frame_extractor_invalid_inputs PASSED [ 88%]
tests/unit/test_video_metadata.py::test_video_metadata_extraction PASSED [100%]

============================== 9 passed in 1.15s ==============================
```

---

## 9. Known Limitations

- **Variable Frame Rate (VFR)**: For true VFR videos without fixed packet timestamps, FFmpeg's `fps` filter converts streams to constant rate slices. Exact per-packet PTS extraction will be introduced if variable-rate drone telemetry synchronization requires microsecond-level packet timestamps in future stages.
