import sys
import os
import json
import subprocess
from typing import Dict, Any

def get_video_metadata(video_path: str) -> Dict[str, Any]:
    """
    Extract video metadata using ffprobe.
    Returns a dictionary containing duration, width, height, average_frame_rate, codec, and frame_count.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
        
    # Attempt to locate ffprobe
    ffprobe_cmd = "ffprobe"
    # Fallback to absolute path on Windows if the command isn't on the current PATH
    if os.name == "nt":
        import shutil
        if not shutil.which("ffprobe"):
            fallback_path = r"D:\SIH26158\tools\ffmpeg\bin\ffprobe.exe"
            if os.path.exists(fallback_path):
                ffprobe_cmd = fallback_path

    cmd = [
        ffprobe_cmd,
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,width,height,avg_frame_rate,r_frame_rate,duration,nb_frames",
        "-of", "json",
        video_path
    ]
    
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        data = json.loads(result.stdout)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffprobe execution failed: {e.stderr}")
    except (json.JSONDecodeError, KeyError) as e:
        raise RuntimeError(f"Failed to parse ffprobe output: {e}")
        
    streams = data.get("streams", [])
    if not streams:
        raise RuntimeError("No video streams found in file.")
        
    stream = streams[0]
    
    # Process average frame rate (e.g., "30/1" or "2997/100")
    avg_fps = 0.0
    avg_fps_str = stream.get("avg_frame_rate", "0/0")
    if "/" in avg_fps_str:
        num, denom = avg_fps_str.split("/")
        try:
            if float(denom) != 0:
                avg_fps = float(num) / float(denom)
        except ValueError:
            pass
            
    # Process duration
    duration = 0.0
    try:
        duration = float(stream.get("duration", 0.0))
    except ValueError:
        pass
        
    # Process frame count
    frame_count = None
    nb_frames_str = stream.get("nb_frames")
    if nb_frames_str is not None:
        try:
            frame_count = int(nb_frames_str)
        except ValueError:
            pass
            
    # Return formatted metadata
    metadata = {
        "duration": duration,
        "width": int(stream.get("width", 0)),
        "height": int(stream.get("height", 0)),
        "average_frame_rate": avg_fps,
        "codec": stream.get("codec_name", "unknown"),
        "frame_count": frame_count
    }
    
    return metadata

def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: python -m src.ingestion.video_metadata <path_to_video>"}), file=sys.stderr)
        sys.exit(1)
        
    video_path = sys.argv[1]
    try:
        metadata = get_video_metadata(video_path)
        print(json.dumps(metadata, indent=4))
        sys.exit(0)
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
