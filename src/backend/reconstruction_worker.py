import argparse
from typing import Dict, Any

from src.backend.session_manager import BackendSessionManager
from src.backend.input_manager import BackendInputManager
from src.backend.job_manager import BackendJobManager, JobManagerError
from pipelines.application.reconstruct_video import reconstruct_video

class BackendReconstructionWorker:
    """
    Worker class responsible for executing reconstruction jobs.
    Connects BackendJobManager state to the existing reconstruction pipeline.
    """
    def __init__(
        self,
        session_manager: BackendSessionManager,
        input_manager: BackendInputManager,
        job_manager: BackendJobManager
    ):
        self.session_manager = session_manager
        self.input_manager = input_manager
        self.job_manager = job_manager

    def run_job(self, job_id: str) -> Dict[str, Any]:
        """
        Executes a job synchronously.
        Transitions the job from queued -> processing -> completed/failed.
        """
        try:
            job_data = self.job_manager.get_job(job_id)
        except JobManagerError as e:
            raise ValueError(f"Cannot run job: {str(e)}")

        if job_data["status"] != "queued":
            # Don't run a job that is already processing, completed, or failed
            raise ValueError(f"Job {job_id} is in status '{job_data['status']}' and cannot be run.")

        session_id = job_data["session_id"]

        # 1. Mark as processing
        self.job_manager.update_job_status(job_id, "processing")

        try:
            # 2. Discover inputs in the session
            inputs = self.input_manager.list_inputs(session_id)
            
            video_path = None
            gps_path = None
            imu_path = None
            calib_path = None
            poses_path = None
            rtk_path = None
            
            # Sort inputs by stored_filename to ensure deterministic order if needed
            inputs = sorted(inputs, key=lambda x: x.get("stored_filename", ""))
            
            # Map inputs to reconstruction pipeline requirements based on input_type or extension
            for input_meta in inputs:
                ext = input_meta.get("extension", "").lower()
                in_type = input_meta.get("input_type")
                orig_name = input_meta.get("original_filename", "").lower()
                path = str(self.input_manager.get_input_path(session_id, input_meta["stored_filename"]))
                
                # Heuristic mapping for pipeline arguments based on InputManager metadata
                if in_type == "video" or ext in [".mp4", ".mov", ".avi", ".mkv", ".webm"]:
                    if video_path is not None:
                        raise ValueError("Multiple video inputs found in session. Cannot deterministically select primary video.")
                    video_path = path
                elif in_type == "gps" or (ext == ".csv" and "gps" in orig_name):
                    gps_path = path
                elif in_type == "imu" or (ext == ".csv" and "imu" in orig_name):
                    imu_path = path
                elif in_type == "calibration" or (ext == ".json" and "calib" in orig_name):
                    calib_path = path
                elif in_type == "poses" or (ext == ".csv" and ("pos" in orig_name or "trajectory" in orig_name)):
                    poses_path = path
                elif in_type == "rtk" or (ext == ".csv" and "rtk" in orig_name):
                    rtk_path = path

            if not video_path:
                raise ValueError("No video input found in session.")

            # 3. Retrieve session workspace to isolate outputs
            workspace_dir = self.session_manager.get_session_workspace(session_id)

            # 4. Construct existing pipeline arguments
            args = argparse.Namespace(
                video=video_path,
                output=str(workspace_dir),
                gps=gps_path,
                imu=imu_path,
                calibration=calib_path,
                poses=poses_path,
                rtk=rtk_path
            )

            # 5. Execute reconstruction engine directly
            result_metadata = reconstruct_video(args)

            # 6. Evaluate success/failure based on pipeline result
            if result_metadata.get("status") == "RECONSTRUCTION_BLOCKED":
                # Pipeline handled a failure (e.g. missing files, bad modes)
                error_msg = result_metadata.get("recommended_action", "Reconstruction blocked.")
                self.job_manager.update_job_status(
                    job_id, 
                    "failed", 
                    error=error_msg, 
                    result_metadata=result_metadata
                )
            else:
                # Success
                self.job_manager.update_job_status(
                    job_id, 
                    "completed", 
                    result_metadata=result_metadata
                )

        except Exception as e:
            # Uncaught exception during execution or mapping
            # Ensure we don't leave the job permanently in processing state
            # Strip absolute paths (e.g. anything looking like a Windows or Unix absolute path) for safety
            raw_err = str(e)
            workspace_dir_str = str(self.session_manager.base_dir)
            safe_err = raw_err.replace(workspace_dir_str, "<WORKSPACE>")
            
            safe_error_msg = f"Worker execution failed: {type(e).__name__} - {safe_err}"
            self.job_manager.update_job_status(job_id, "failed", error=safe_error_msg)
            
        return self.job_manager.get_job(job_id)
