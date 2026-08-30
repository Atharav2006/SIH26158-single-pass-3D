import argparse
import sys
import time
from pathlib import Path
from src.config import load_config
from src.ingestion.video_metadata import get_video_metadata
from src.ingestion.frame_extractor import FrameExtractor

def get_directory_size(path: Path) -> int:
    """Calculate total size of files in a directory."""
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total

def main():
    parser = argparse.ArgumentParser(description="SIH26158 Baseline: Extract video frames with timestamp indexing.")
    parser.add_argument("--input", "-i", required=True, help="Path to input video file.")
    parser.add_argument("--output", "-o", required=True, help="Directory to save extracted frames and metadata.")
    parser.add_argument("--fps", type=float, default=None, help="Extraction frame rate (default: keep source FPS).")
    parser.add_argument("--width", type=int, default=None, help="Output image width.")
    parser.add_argument("--height", type=int, default=None, help="Output image height.")
    parser.add_argument("--format", type=str, default="jpg", help="Output image format (default: jpg).")
    parser.add_argument("--quality", type=int, default=2, help="JPEG quality scale (default: 2).")
    parser.add_argument("--config", "-c", type=str, default=None, help="Optional path to custom config JSON.")

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)

    print("=" * 60)
    print("SIH26158: Baseline Frame Extraction Pipeline")
    print("=" * 60)
    print(f"Input Video: {input_path}")
    print(f"Output Directory: {output_dir}")

    # 1. Validation and Metadata Inspection
    if not input_path.exists():
        print(f"[ERROR] Input video not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    try:
        source_meta = get_video_metadata(str(input_path))
        print("\n--- Source Video Metadata ---")
        print(f"  Duration:   {source_meta['duration']:.2f} seconds")
        print(f"  Resolution: {source_meta['width']} x {source_meta['height']}")
        print(f"  Frame Rate: {source_meta['average_frame_rate']} FPS")
        print(f"  Codec:      {source_meta['codec']}")
        print(f"  Frames:     {source_meta.get('frame_count', 'N/A')}")
    except Exception as e:
        print(f"[ERROR] Failed to read video metadata: {e}", file=sys.stderr)
        sys.exit(1)

    # 2. Frame Extraction
    config = load_config(args.config)
    extractor = FrameExtractor(config=config)

    print("\n--- Running Extraction ---")
    start_time = time.perf_counter()
    try:
        metadata = extractor.extract(
            video_path=input_path,
            output_dir=output_dir,
            extraction_fps=args.fps,
            output_width=args.width,
            output_height=args.height,
            output_image_format=args.format,
            output_image_quality=args.quality
        )
    except Exception as e:
        print(f"[ERROR] Frame extraction failed: {e}", file=sys.stderr)
        sys.exit(1)

    elapsed = time.perf_counter() - start_time
    extracted_count = metadata["extracted_frame_count"]
    fps_throughput = extracted_count / elapsed if elapsed > 0 else 0
    total_size_bytes = get_directory_size(output_dir)
    total_size_mb = total_size_bytes / (1024 * 1024)

    # 3. Print Extraction Summary
    print("\n--- Extraction Summary ---")
    print(f"  Extracted Frames:   {extracted_count}")
    print(f"  Extraction FPS:     {metadata['extraction_fps']} FPS")
    print(f"  Output Resolution:  {metadata['output_width']} x {metadata['output_height']}")
    print(f"  Elapsed Time:       {elapsed:.2f} s")
    print(f"  Throughput:         {fps_throughput:.1f} frames/s")
    print(f"  Total Output Size:  {total_size_mb:.2f} MB")
    print(f"  Frame Index CSV:    {output_dir / 'frame_index.csv'}")
    print(f"  Metadata JSON:      {output_dir / 'extraction_metadata.json'}")
    print("=" * 60)
    print("RESULT: SUCCESS")

if __name__ == "__main__":
    main()
