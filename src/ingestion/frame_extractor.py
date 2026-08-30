import os
import sys
import csv
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, Union, List

from src.config import load_config, Configuration
from src.ingestion.video_metadata import get_video_metadata

def _find_binary(name: str, fallback_path: str) -> str:
    """Find a binary in PATH or fallback to project tools path."""
    if shutil.which(name):
        return name
    if os.name == "nt" and os.path.exists(fallback_path):
        return fallback_path
    return name

def _get_binary_version(binary_cmd: str) -> str:
    """Query the version string from ffmpeg or ffprobe."""
    try:
        res = subprocess.run([binary_cmd, "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        lines = res.stdout.strip().splitlines()
        return lines[0] if lines else "unknown"
    except Exception:
        return "unknown"

class FrameExtractor:
    """
    Reproducible video frame extraction engine using FFmpeg.
    Extracts numbered frames, generates frame_index.csv, and produces extraction_metadata.json.
    """
    def __init__(self, config: Optional[Configuration] = None):
        self.config = config or load_config()
        self.allowed_formats: List[str] = self.config.get("ingestion.allowed_formats", [".mp4", ".avi", ".mov", ".mkv"])
        
        self.ffmpeg_bin = _find_binary("ffmpeg", r"D:\SIH26158\tools\ffmpeg\bin\ffmpeg.exe")
        self.ffprobe_bin = _find_binary("ffprobe", r"D:\SIH26158\tools\ffmpeg\bin\ffprobe.exe")

    def extract(
        self,
        video_path: Union[str, Path],
        output_dir: Union[str, Path],
        extraction_fps: Optional[float] = None,
        output_width: Optional[int] = None,
        output_height: Optional[int] = None,
        output_image_format: Optional[str] = None,
        output_image_quality: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Extract frames from a video file into an organized directory structure.

        Args:
            video_path: Path to the input video.
            output_dir: Target output directory.
            extraction_fps: Custom FPS to extract. If None, preserves source video FPS.
            output_width: Desired output width. If None, keeps original.
            output_height: Desired output height. If None, keeps original.
            output_image_format: Target format (e.g. 'jpg', 'png'). Defaults to config or 'jpg'.
            output_image_quality: Quality setting (for jpg, 1-31 scale where 2 is highest).

        Returns:
            Dictionary containing extraction metadata and summary.
        """
        video_path = Path(video_path).resolve()
        output_dir = Path(output_dir).resolve()

        # 1. Validation
        if not video_path.exists():
            raise FileNotFoundError(f"Input video file not found: {video_path}")

        ext = video_path.suffix.lower()
        if ext not in self.allowed_formats:
            raise ValueError(f"Unsupported video format '{ext}'. Allowed formats: {self.allowed_formats}")

        # 2. Extract authoritative source metadata
        source_meta = get_video_metadata(str(video_path))
        source_fps = float(source_meta.get("average_frame_rate", 0.0))
        source_duration = float(source_meta.get("duration", 0.0))
        source_width = int(source_meta.get("width", 0))
        source_height = int(source_meta.get("height", 0))
        source_codec = str(source_meta.get("codec", "unknown"))
        source_frame_count = source_meta.get("frame_count")

        if source_fps <= 0:
            raise ValueError(f"Invalid source video frame rate: {source_fps}")
        if source_width <= 0 or source_height <= 0:
            raise ValueError(f"Invalid source video dimensions: {source_width}x{source_height}")

        # 3. Resolve configuration parameters
        cfg_fps = self.config.get("ingestion.extraction_fps", None)
        effective_fps = float(extraction_fps) if extraction_fps is not None else (float(cfg_fps) if cfg_fps else source_fps)
        if effective_fps <= 0:
            raise ValueError(f"Invalid extraction FPS: {effective_fps}")

        target_width = output_width or self.config.get("ingestion.output_width", None)
        target_height = output_height or self.config.get("ingestion.output_height", None)
        actual_out_width = int(target_width) if target_width else source_width
        actual_out_height = int(target_height) if target_height else source_height

        img_format = output_image_format or self.config.get("ingestion.output_image_format", "jpg")
        img_format = img_format.lstrip(".").lower()
        img_quality = output_image_quality or self.config.get("ingestion.output_image_quality", 2)

        # 4. Prepare directory structure
        frames_dir = output_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        # 5. Build FFmpeg command
        cmd = [self.ffmpeg_bin, "-y", "-i", str(video_path)]
        vf_filters: List[str] = []

        # If extraction_fps differs from source, or is explicitly requested
        if extraction_fps is not None or (cfg_fps is not None and cfg_fps != source_fps):
            vf_filters.append(f"fps={effective_fps}")

        # If custom dimensions are requested
        if target_width and target_height and (target_width != source_width or target_height != source_height):
            vf_filters.append(f"scale={actual_out_width}:{actual_out_height}")

        if vf_filters:
            cmd.extend(["-vf", ",".join(vf_filters)])

        if img_format in ["jpg", "jpeg"]:
            cmd.extend(["-q:v", str(img_quality)])

        output_pattern = str(frames_dir / f"frame_%06d.{img_format}")
        cmd.append(output_pattern)

        # 6. Execute extraction
        try:
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"FFmpeg frame extraction failed with return code {e.returncode}:\n{e.stderr}")
        except FileNotFoundError:
            raise RuntimeError(f"FFmpeg binary '{self.ffmpeg_bin}' not found on system.")

        # 7. Verify extracted images
        image_files = sorted(list(frames_dir.glob(f"frame_*.{img_format}")))
        extracted_frame_count = len(image_files)

        if extracted_frame_count == 0:
            raise RuntimeError("FFmpeg completed execution, but no output frames were produced.")

        # Verify contiguous 1-based naming
        for idx, img_path in enumerate(image_files, start=1):
            expected_name = f"frame_{idx:06d}.{img_format}"
            if img_path.name != expected_name:
                raise RuntimeError(f"Frame sequence error: expected '{expected_name}', found '{img_path.name}'")

        # 8. Generate frame_index.csv
        csv_path = output_dir / "frame_index.csv"
        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["frame_id", "filename", "timestamp_seconds", "source_frame_number", "width", "height"])
            
            for idx, img_path in enumerate(image_files, start=1):
                timestamp_sec = round((idx - 1) / effective_fps, 6)
                source_frame_num = int(round((idx - 1) / effective_fps * source_fps)) + 1
                writer.writerow([idx, img_path.name, timestamp_sec, source_frame_num, actual_out_width, actual_out_height])

        # 9. Generate extraction_metadata.json
        ffmpeg_version = _get_binary_version(self.ffmpeg_bin)
        ffprobe_version = _get_binary_version(self.ffprobe_bin)

        metadata: Dict[str, Any] = {
            "source_video": str(video_path),
            "source_duration_seconds": source_duration,
            "source_width": source_width,
            "source_height": source_height,
            "source_fps": source_fps,
            "source_codec": source_codec,
            "source_frame_count": source_frame_count,
            "extraction_fps": effective_fps,
            "extracted_frame_count": extracted_frame_count,
            "output_width": actual_out_width,
            "output_height": actual_out_height,
            "output_image_format": img_format,
            "extraction_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "ffmpeg_version": ffmpeg_version,
            "ffprobe_version": ffprobe_version
        }

        json_path = output_dir / "extraction_metadata.json"
        with open(json_path, mode="w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)

        return metadata

def extract_frames(
    video_path: Union[str, Path],
    output_dir: Union[str, Path],
    config: Optional[Configuration] = None,
    **kwargs
) -> Dict[str, Any]:
    """Helper function to extract frames with FrameExtractor."""
    extractor = FrameExtractor(config=config)
    return extractor.extract(video_path=video_path, output_dir=output_dir, **kwargs)
