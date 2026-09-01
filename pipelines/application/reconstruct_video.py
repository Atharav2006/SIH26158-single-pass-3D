import argparse
import sys
import json
import time
from pathlib import Path

from src.reconstruction import (
    VideoInputSpec,
    ReconstructionSession,
    ModeSelector,
    RelativeDepthBackend,
    ReconstructionResult
)
from src.ingestion.video_session import VideoValidator
from src.ingestion.optional_sensors import SensorDetector, SensorStatus

def reconstruct_video(args):
    start_time = time.time()
    session_dir = Path(args.output)
    session_id = session_dir.name
    workspace_dir = str(session_dir.parent)
    
    # Phase 4 - Create Session
    session = ReconstructionSession(session_id, workspace_dir)
    
    # Phase 1 - Generalized Input Contract
    try:
        spec = VideoInputSpec(
            video_path=Path(args.video),
            gps_path=Path(args.gps) if args.gps else None,
            imu_path=Path(args.imu) if args.imu else None,
            calibration_path=Path(args.calibration) if args.calibration else None,
            poses_path=Path(args.poses) if args.poses else None,
            rtk_path=Path(args.rtk) if args.rtk else None
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
        
    # Phase 2 - Validate Video
    metadata = VideoValidator.validate(spec.video_path)
    
    # Phase 5 - Sensor Inspection
    sensors = {
        "gps": SensorDetector.detect(spec.gps_path).value,
        "imu": SensorDetector.detect(spec.imu_path).value,
        "rtk": SensorDetector.detect(spec.rtk_path).value
    }
    
    # B6.1 Phase 8 - Generic Pipeline Providers
    from src.reconstruction.providers import ColmapPoseProvider, ColmapCalibrationProvider
    
    pose_diagnostics = {}
    calib_diagnostics = {}
    
    # If poses are missing, attempt auto-pose estimation
    if not spec.poses_path:
        print("No poses provided. Attempting COLMAP automatic pose estimation...")
        pose_provider = ColmapPoseProvider(session)
        pose_res = pose_provider.estimate_poses()
        pose_diagnostics = pose_res
        if pose_res["status"] in ["POSE_ESTIMATION_READY", "POSE_QUALITY_LOW"]:
            # Auto-pose was successful enough to produce a file
            spec.poses_path = Path(pose_res["poses_path"])
            
            # If calibration was missing, we can now parse the COLMAP estimated calibration
            if not spec.calibration_path:
                print("No calibration provided. Attempting COLMAP auto-calibration extraction...")
                calib_provider = ColmapCalibrationProvider(session)
                calib_res = calib_provider.estimate_calibration()
                calib_diagnostics = calib_res
                if calib_res["status"] in ["CALIBRATION_READY", "CALIBRATION_UNCERTAIN"]:
                    spec.calibration_path = Path(calib_res["calibration_path"])
    
    # Phase 6 - Select Mode
    mode = ModeSelector.evaluate(spec)
    
    # Phase 11 - Structured diagnostic error
    if mode.status == "RECONSTRUCTION_BLOCKED":
        output = {
            "status": mode.status,
            "missing_requirements": mode.missing_requirements,
            "selected_mode": mode.selected_mode,
            "recommended_action": mode.recommended_action
        }
        out_file = session.get_path("diagnostics/status.json")
        with open(out_file, 'w') as f:
            json.dump(output, f, indent=4)
        print(json.dumps(output, indent=2))
        return output
        
    # Phase 7 & 8 - Run Backend
    # If mode allows relative, run relative backend (we don't have a metric backend implemented yet)
    backend = RelativeDepthBackend()
    backend.prepare(session, mode)
    geom_path = backend.run(session)
    
    # Final Result
    is_metric = (mode.selected_mode == "METRIC_RECONSTRUCTION")
    
    result = ReconstructionResult(
        geometry_path=geom_path,
        metric=is_metric,
        scale_type="metric" if is_metric else "relative",
        coordinate_frame="Local_ENU" if is_metric else "relative_world_gauge",
        status=mode.status,
        anchor_source=mode.anchor_source,
        provenance="Supplied RTK Anchor" if is_metric else None
    )
    
    end_time = time.time()
    
    # Save output
    final_output = {
        "session_id": session_id,
        "geometry_path": result.geometry_path,
        "metric": result.metric,
        "scale_type": result.scale_type,
        "coordinate_frame": result.coordinate_frame,
        "status": result.status,
        "warnings": result.warnings,
        "runtime_sec": end_time - start_time,
        "pose_diagnostics": pose_diagnostics,
        "calib_diagnostics": calib_diagnostics
    }
    
    with open(session.get_path("exports/reconstruction_summary.json"), 'w') as f:
        json.dump(final_output, f, indent=4)
        
    print(json.dumps(final_output, indent=2))
    return final_output

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generalized Video-to-3D Reconstruction Engine")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--output", required=True, help="Session output directory")
    parser.add_argument("--gps", help="Optional GPS CSV")
    parser.add_argument("--imu", help="Optional IMU CSV")
    parser.add_argument("--calibration", help="Optional Calibration JSON")
    parser.add_argument("--poses", help="Optional Poses CSV")
    parser.add_argument("--rtk", help="Optional RTK CSV")
    
    args = parser.parse_args()
    reconstruct_video(args)
