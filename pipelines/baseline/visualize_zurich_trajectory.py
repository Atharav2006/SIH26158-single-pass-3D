import argparse
import sys
import csv
import json
from pathlib import Path

from src.pose.pose_loader import (
    load_poses_from_csv,
    load_image_metadata,
    associate_poses_to_images,
    export_image_pose_associations_csv
)
from src.pose.association import (
    associate_groundtruth_by_imgid,
    export_image_groundtruth_associations_csv,
    AssociationMethod
)
from src.pose.trajectory import Trajectory
from src.visualization.trajectory_plot import plot_topdown_trajectory, plot_3d_trajectory

def calculate_gps_statistics(gps_csv_path: Path) -> dict:
    """Calculate min/max latitude, longitude, and altitude from gps.csv."""
    if not gps_csv_path.exists():
        return {}

    lats, lons, alts = [], [], []
    with open(gps_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                lats.append(float(row["latitude"]))
                lons.append(float(row["longitude"]))
                alt_str = row.get("altitude_if_available")
                if alt_str and alt_str.strip():
                    alts.append(float(alt_str))
            except (KeyError, ValueError):
                pass

    if not lats:
        return {}

    return {
        "gps_record_count": len(lats),
        "min_latitude": round(min(lats), 7),
        "max_latitude": round(max(lats), 7),
        "latitude_span_deg": round(max(lats) - min(lats), 7),
        "min_longitude": round(min(lons), 7),
        "max_longitude": round(max(lons), 7),
        "longitude_span_deg": round(max(lons) - min(lons), 7),
        "min_altitude_meters": round(min(alts), 2) if alts else None,
        "max_altitude_meters": round(max(alts), 2) if alts else None,
        "altitude_span_meters": round(max(alts) - min(alts), 2) if alts else None
    }

def main():
    parser = argparse.ArgumentParser(description="SIH26158 Baseline: Load, transform, validate, and visualize Zurich Urban MAV trajectory.")
    parser.add_argument("--normalized", "-n", required=True, help="Directory containing normalized pose.csv, images.csv, and gps.csv.")
    parser.add_argument("--output", "-o", required=True, help="Directory to save trajectory.csv, trajectory.json, validation, and plots.")

    args = parser.parse_args()

    norm_dir = Path(args.normalized).resolve()
    out_dir = Path(args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SIH26158: Zurich Urban MAV Trajectory Processing & Visualization")
    print("=" * 60)
    print(f"Normalized Input Dir: {norm_dir}")
    print(f"Output Directory:     {out_dir}")

    pose_csv = norm_dir / "pose.csv"
    images_csv = norm_dir / "images.csv"
    gps_csv = norm_dir / "gps.csv"

    # 1. Load Poses and Image Metadata
    try:
        poses_utm = load_poses_from_csv(pose_csv)
        print(f"\nLoaded {len(poses_utm)} ground-truth poses from {pose_csv.name}")
    except Exception as e:
        print(f"[ERROR] Failed to load poses: {e}", file=sys.stderr)
        sys.exit(1)

    images = []
    if images_csv.exists():
        try:
            images = load_image_metadata(images_csv)
            print(f"Loaded {len(images)} image records from {images_csv.name}")
        except Exception as e:
            print(f"[WARNING] Failed to load images metadata: {e}")

    # 2. Authoritative Ground-Truth Exact-ID Association
    exact_assocs = []
    exact_sync_stats = {}
    if images and poses_utm:
        exact_assocs = associate_groundtruth_by_imgid(images, poses_utm)
        gt_assoc_csv_path = out_dir / "image_groundtruth_associations.csv"
        export_image_groundtruth_associations_csv(exact_assocs, gt_assoc_csv_path)
        print(f"Exported {gt_assoc_csv_path.name}")

        matched_exact = [a for a in exact_assocs if a.matched]
        exact_sync_stats = {
            "total_images": len(images),
            "exact_keyframe_matches": len(matched_exact),
            "unmatched_intermediate_frames": len(images) - len(matched_exact),
            "match_rate": round(len(matched_exact) / len(images), 4) if images else 0.0,
            "association_method": AssociationMethod.EXACT_ID.value
        }

    # 3. Asynchronous Nearest-Neighbor Telemetry Association (Preserved for continuous streams)
    ts_associations = []
    ts_sync_stats = {}
    if images and poses_utm:
        ts_associations = associate_poses_to_images(poses_utm, images, max_tolerance=0.05)
        assoc_csv_path = out_dir / "image_pose_associations.csv"
        export_image_pose_associations_csv(ts_associations, assoc_csv_path)
        print(f"Exported {assoc_csv_path.name}")

        matched = [a for a in ts_associations if a["matched_pose_timestamp"] is not None]
        dts = [a["absolute_delta_seconds"] for a in matched if a["absolute_delta_seconds"] is not None]
        ts_sync_stats = {
            "total_images": len(images),
            "matched_poses": len(matched),
            "unmatched_images": len(images) - len(matched),
            "match_rate": round(len(matched) / len(images), 4) if images else 0.0,
            "mean_time_diff_sec": round(sum(dts) / len(dts), 6) if dts else 0.0,
            "max_time_diff_sec": round(max(dts), 6) if dts else 0.0,
            "association_method": AssociationMethod.TIMESTAMP_NEAREST.value
        }

    # 4. Convert Trajectory to Local ENU
    traj_utm = Trajectory.from_poses(poses_utm)
    origin = poses_utm[0].position_xyz
    traj_local = traj_utm.to_local_enu(origin=origin)

    # 5. Trajectory Sanity Validation
    val_result = traj_local.validate_trajectory()
    val_json_path = out_dir / "trajectory_validation.json"
    with open(val_json_path, "w", encoding="utf-8") as f:
        json.dump(val_result, f, indent=4)
    print(f"Exported {val_json_path.name} (Status: {val_result['status']})")

    # 6. Compute GPS and Trajectory Statistics
    traj_stats = traj_local.compute_statistics()
    gps_stats = calculate_gps_statistics(gps_csv)

    extra_metadata = {
        "dataset_source": "Zurich Urban MAV Dataset (AGZ)",
        "origin_utm_coordinates": {
            "x_easting": origin.x,
            "y_northing": origin.y,
            "z_altitude": origin.z,
            "unit": "meter"
        },
        "gps_statistics": gps_stats,
        "authoritative_groundtruth_association": exact_sync_stats,
        "asynchronous_timestamp_synchronization": ts_sync_stats
    }

    # 7. Export trajectory.csv and trajectory.json
    out_csv = out_dir / "trajectory.csv"
    out_json = out_dir / "trajectory.json"

    traj_local.export_csv(out_csv)
    traj_local.export_json(out_json, extra_metadata=extra_metadata)
    print(f"Exported {out_csv.name}")
    print(f"Exported {out_json.name}")

    # 8. Render Trajectory Plots
    plot_3d_path = out_dir / "trajectory_3d.png"
    plot_3d_alias = out_dir / "trajectory.png"
    plot_2d_path = out_dir / "trajectory_topdown.png"

    plot_3d_trajectory(traj_local, plot_3d_path)
    plot_3d_trajectory(traj_local, plot_3d_alias)
    plot_topdown_trajectory(traj_local, plot_2d_path)
    print(f"Generated 3D Plot:       {plot_3d_path.name}")
    print(f"Generated Top-Down Plot: {plot_2d_path.name}")

    # 9. Print Summary
    print("\n--- Trajectory & Association Summary ---")
    print(f"  Validation Status:       {val_result['status']}")
    print(f"  Pose Count:              {traj_stats['pose_count']}")
    print(f"  Duration:                {traj_stats['duration_seconds']:.2f} s ({traj_stats['duration_seconds']/60:.2f} min)")
    print(f"  Trajectory Length:       {traj_stats['trajectory_length_meters']:.2f} m ({traj_stats['trajectory_length_meters']/1000:.3f} km)")
    print(f"  Mean Speed:              {traj_stats['mean_speed_mps']:.2f} m/s")
    print(f"  Median Speed:            {traj_stats['median_speed_mps']:.2f} m/s")
    print(f"  Max Speed:               {traj_stats['max_speed_mps']:.2f} m/s")
    print(f"  Local Extents:           X: [{traj_stats['spatial_extent']['min_x']}, {traj_stats['spatial_extent']['max_x']}] m")
    print(f"                           Y: [{traj_stats['spatial_extent']['min_y']}, {traj_stats['spatial_extent']['max_y']}] m")
    print(f"                           Z: [{traj_stats['spatial_extent']['min_z']}, {traj_stats['spatial_extent']['max_z']}] m")
    if exact_sync_stats:
        print(f"  Authoritative GT Match:  {exact_sync_stats['exact_keyframe_matches']}/{exact_sync_stats['total_images']} exact keyframes ({exact_sync_stats['match_rate']*100:.1f}%)")
        print(f"  Intermediate Frames:     {exact_sync_stats['unmatched_intermediate_frames']}/{exact_sync_stats['total_images']} frames")
    print("=" * 60)

    if val_result["status"] == "FAIL":
        print("IDENTITY/ASSOCIATION STATUS: BLOCKED", file=sys.stderr)
        sys.exit(1)
    else:
        print("IDENTITY/ASSOCIATION STATUS: PASS")

if __name__ == "__main__":
    main()
