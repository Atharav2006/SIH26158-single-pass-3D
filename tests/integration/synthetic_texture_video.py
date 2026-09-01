import cv2
import numpy as np
from pathlib import Path
import json

def generate_textured_video(out_path: Path, num_frames=20, width=640, height=480):
    """
    Generates a synthetic video with a textured background (e.g. checkerboard or noise)
    and a moving 'camera' so that SfM can find features and match them.
    """
    # Create a large texture map (virtual world plane)
    plane = np.random.randint(0, 256, (height * 3, width * 3, 3), dtype=np.uint8)
    
    # Add some distinctive geometric shapes to ensure good SIFT features
    for _ in range(50):
        pt1 = (np.random.randint(0, width*3), np.random.randint(0, height*3))
        pt2 = (pt1[0] + np.random.randint(20, 100), pt1[1] + np.random.randint(20, 100))
        color = (int(np.random.randint(0, 255)), int(np.random.randint(0, 255)), int(np.random.randint(0, 255)))
        cv2.rectangle(plane, pt1, pt2, color, -1)
        
    out = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*'mp4v'), 10, (width, height))
    
    frames_dir = out_path.parent / (out_path.stem + "_frames")
    frames_dir.mkdir(parents=True, exist_ok=True)
    
    # Simulate a camera moving right and slightly down
    start_x = width
    start_y = height
    for i in range(num_frames):
        # Crop a width x height window from the plane
        # Camera translates uniformly
        x = start_x + int(i * 15)
        y = start_y + int(i * 5)
        
        frame = plane[y:y+height, x:x+width].copy()
        out.write(frame)
        cv2.imwrite(str(frames_dir / f"{i:04d}.jpg"), frame)
        
    out.release()
    
    return frames_dir
