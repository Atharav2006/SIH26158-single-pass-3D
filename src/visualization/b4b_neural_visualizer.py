import cv2
import numpy as np
from pathlib import Path
import json

def generate_comparison(out_dir: Path):
    """
    Combines B4, B4-B, and B4-B+ rendered depth and RGB into a side-by-side comparison grid.
    """
    variants = ["B4", "B4_B", "B4_B_Plus"]
    
    rgb_imgs = []
    depth_imgs = []
    
    for v in variants:
        rgb_path = out_dir / f"{v}_val_pred.png"
        depth_path = out_dir / f"{v}_val_depth.png"
        
        if rgb_path.exists():
            rgb = cv2.imread(str(rgb_path))
            # Add label
            cv2.putText(rgb, v, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            rgb_imgs.append(rgb)
            
        if depth_path.exists():
            depth = cv2.imread(str(depth_path))
            cv2.putText(depth, v + " Depth", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            depth_imgs.append(depth)
            
    if rgb_imgs and depth_imgs:
        rgb_row = np.hstack(rgb_imgs)
        depth_row = np.hstack(depth_imgs)
        grid = np.vstack([rgb_row, depth_row])
        
        out_path = out_dir / "b4b_comparison_grid.png"
        cv2.imwrite(str(out_path), grid)
        print(f"Comparison grid saved to {out_path}")

if __name__ == "__main__":
    out_dir = Path("outputs/reports/zurich_mav/b4b")
    generate_comparison(out_dir)
