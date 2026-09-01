import open3d as o3d
import numpy as np
from pathlib import Path
import json

def validate_and_visualize_b3(b3_ply_path: Path, b2_csv_path: Path, output_dir: Path):
    if not b3_ply_path.exists():
        print(f"PLY not found: {b3_ply_path}")
        return
        
    pcd = o3d.io.read_point_cloud(str(b3_ply_path))
    pts = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    
    if len(pts) == 0:
        print("Point cloud is empty!")
        return
        
    print(f"Loaded point cloud with {len(pts)} points")
    
    # 1. Metric Scale Validation
    # We will compute the bounding box size
    min_pt = pts.min(axis=0)
    max_pt = pts.max(axis=0)
    extents = max_pt - min_pt
    print(f"Extents [X, Y, Z]: {extents}")
    print(f"Bounding Box Volume: {extents[0]*extents[1]*extents[2]:.2f} m^3")
    
    # We know the drone flies at ~2 meters in this local ENU frame, and building/ground
    # is usually between Z = -5 and Z = 5.
    
    # 2. Extract trajectory for visualization
    import csv
    traj_pts = []
    with open(b2_csv_path, "r") as f:
        for r in csv.DictReader(f):
            traj_pts.append([float(r["x"]), float(r["y"]), float(r["z"])])
    traj_pts = np.array(traj_pts)
    
    traj_pcd = o3d.geometry.PointCloud()
    traj_pcd.points = o3d.utility.Vector3dVector(traj_pts)
    traj_pcd.paint_uniform_color([1, 0, 0]) # Red trajectory
    
    # Create line set for trajectory
    lines = [[i, i+1] for i in range(len(traj_pts)-1)]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(traj_pts)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in range(len(lines))])
    
    # 3. Save visualization artifacts
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Render from top-down and isometric views
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=1920, height=1080)
    vis.add_geometry(pcd)
    vis.add_geometry(line_set)
    
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    opt.point_size = 2.0
    
    # Isometric view
    ctr = vis.get_view_control()
    ctr.set_lookat(pts.mean(axis=0))
    ctr.set_front([0.5, 0.5, -0.707])
    ctr.set_up([0, 0, 1])
    ctr.set_zoom(0.8)
    
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(str(output_dir / "b3_reconstruction_isometric.png"))
    
    # Top-down view
    ctr.set_front([0, 0, -1])
    ctr.set_up([0, 1, 0])
    ctr.set_zoom(0.8)
    
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(str(output_dir / "b3_reconstruction_topdown.png"))
    
    vis.destroy_window()
    
    # 4. Save validation stats
    stats = {
        "point_count": len(pts),
        "extents_m": {
            "x": extents[0],
            "y": extents[1],
            "z": extents[2]
        },
        "center_m": {
            "x": pts.mean(axis=0)[0],
            "y": pts.mean(axis=0)[1],
            "z": pts.mean(axis=0)[2]
        }
    }
    with open(output_dir / "b3_pointcloud_stats.json", "w") as f:
        json.dump(stats, f, indent=4)
        
    print("Validation and visualization complete.")

if __name__ == "__main__":
    b3_ply = Path("outputs/reports/zurich_mav/b3/fused.ply")
    b2_csv = Path("outputs/reports/zurich_mav/b2/b2_fused_trajectory.csv")
    out_dir = Path("outputs/reports/zurich_mav/b3/")
    
    # Fallback to smoke if full doesn't exist yet
    if not b3_ply.exists():
        b3_ply = Path("outputs/reports/zurich_mav/b3/fused_smoke.ply")
        
    validate_and_visualize_b3(b3_ply, b2_csv, out_dir)
