import argparse
import sys
import json
from pathlib import Path

from src.ingestion.datasets.zurich_mav import ZurichMAVAdapter
from src.ingestion.dataset_validator import DatasetValidator

def main():
    parser = argparse.ArgumentParser(description="SIH26158 Baseline: Profile and validate Zurich Urban MAV dataset.")
    parser.add_argument("--dataset", "-d", required=True, help="Path to Zurich MAV dataset root directory.")
    parser.add_argument("--output", "-o", default="outputs/reports/zurich_mav", help="Directory for normalized outputs and reports.")

    args = parser.parse_args()

    dataset_path = Path(args.dataset).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SIH26158: Zurich Urban MAV Dataset Ingestion & Validation")
    print("=" * 60)
    print(f"Dataset Path:     {dataset_path}")
    print(f"Output Directory: {output_dir}")

    # 1. Parse dataset
    try:
        adapter = ZurichMAVAdapter(dataset_path)
        print("\n--- Validating & Parsing Dataset ---")
        adapter.parse()
    except Exception as e:
        print(f"[ERROR] Failed to parse Zurich MAV dataset: {e}", file=sys.stderr)
        sys.exit(1)

    # 2. Export normalized representation
    try:
        normalized_paths = adapter.export_normalized(output_dir)
        print("\n--- Exported Normalized Files ---")
        for key, p in normalized_paths.items():
            print(f"  {key:<8}: {p}")
    except Exception as e:
        print(f"[ERROR] Failed to export normalized dataset metadata: {e}", file=sys.stderr)
        sys.exit(1)

    # 3. Validate dataset
    validator = DatasetValidator(adapter)
    validation_report = validator.validate()

    report_path = output_dir / "zurich_mav_validation.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(validation_report, f, indent=4)

    # Also save to outputs/reports/zurich_mav_validation.json
    alt_report_path = Path("outputs/reports/zurich_mav_validation.json").resolve()
    alt_report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(alt_report_path, "w", encoding="utf-8") as f:
        json.dump(validation_report, f, indent=4)

    # 4. Print Summary
    print("\n--- Validation Summary ---")
    print(f"  Status:             {validation_report['status']}")
    print(f"  Images Found:       {len(adapter.images)}")
    print(f"  GPS Records:        {len(adapter.gps)}")
    print(f"  IMU Records:        {len(adapter.imu)}")
    print(f"  Pose Records:       {len(adapter.pose)}")
    if adapter.camera:
        print(f"  Camera Model:       {adapter.camera['model']} ({adapter.camera['image_width']}x{adapter.camera['image_height']})")
        print(f"  Focal Length (fx):  {adapter.camera['fx']:.2f} px")

    sync_stats = validation_report["checks"].get("synchronization", {})
    if "image_to_gps" in sync_stats:
        print(f"  Image-GPS Sync:     {sync_stats['image_to_gps']['matched_count']} matched (mean dt: {sync_stats['image_to_gps']['mean_time_diff_sec']*1000:.2f} ms)")

    print(f"\n  Validation Report:  {report_path}")
    print("=" * 60)

    if validation_report["status"] == "FAIL":
        print("RESULT: FAILED", file=sys.stderr)
        sys.exit(1)
    else:
        print("RESULT: SUCCESS")

if __name__ == "__main__":
    main()
