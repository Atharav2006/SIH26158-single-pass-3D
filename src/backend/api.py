import os
import shutil
import tempfile
from typing import Optional, Dict, Any, List
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, status
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from contextlib import asynccontextmanager

from src.backend.metadata_store import MetadataStore
from src.backend.session_manager import BackendSessionManager, SessionManagerError, SessionConflictError
from src.backend.input_manager import BackendInputManager, InputManagerError
from src.backend.job_manager import BackendJobManager, JobManagerError
from src.backend.reconstruction_worker import BackendReconstructionWorker
from src.backend.execution_manager import BackgroundExecutionManager
from src.backend.result_manager import BackendResultManager, ResultManagerError, ResultConflictError

# --- Global State ---
# In a full app, these would ideally be bound to app.state
_metadata_store = MetadataStore()
# Initialize the database schema on startup
_metadata_store.initialize()

_session_manager = BackendSessionManager(metadata_store=_metadata_store)
_input_manager = BackendInputManager(_session_manager)
_job_manager = BackendJobManager(_session_manager)
_result_manager = BackendResultManager(_session_manager, _job_manager)
_worker = BackendReconstructionWorker(_session_manager, _input_manager, _job_manager)
_execution_manager = BackgroundExecutionManager(_worker)

import sys

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Reap any jobs left in 'queued' or 'processing' due to a restart
    _execution_manager.reap_stuck_jobs()
    yield
    # Shutdown
    if "pytest" not in sys.modules:
        _execution_manager.shutdown(wait=False)

app = FastAPI(
    title="Member 4 Backend API",
    description="REST API layer for session, input, and job management.",
    version="0.1.0",
    lifespan=lifespan
)

# Configuration for CORS - Do not use wildcard unless explicitly required for dev.
# Configurable through environment, defaulting to none to avoid permissive production default.
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "").split(",")
if ALLOWED_ORIGINS and ALLOWED_ORIGINS != [""]:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# --- Dependency Providers ---

def get_session_manager() -> BackendSessionManager:
    return _session_manager

def get_input_manager(session_manager: BackendSessionManager = Depends(get_session_manager)) -> BackendInputManager:
    # We still need to return an instance bound to the resolved session_manager
    # In production this is _session_manager and returns the global _input_manager,
    # but in tests it might be a mocked session manager.
    if session_manager is _session_manager:
        return _input_manager
    return BackendInputManager(session_manager)

def get_job_manager(session_manager: BackendSessionManager = Depends(get_session_manager)) -> BackendJobManager:
    if session_manager is _session_manager:
        return _job_manager
    return BackendJobManager(session_manager)

def get_execution_manager(
    job_manager: BackendJobManager = Depends(get_job_manager),
    input_manager: BackendInputManager = Depends(get_input_manager),
    session_manager: BackendSessionManager = Depends(get_session_manager)
) -> BackgroundExecutionManager:
    if session_manager is _session_manager:
        return _execution_manager
    # If overridden for tests, create a temporary worker and execution manager
    worker = BackendReconstructionWorker(session_manager, input_manager, job_manager)
    return BackgroundExecutionManager(worker)

def get_result_manager(
    session_manager: BackendSessionManager = Depends(get_session_manager),
    job_manager: BackendJobManager = Depends(get_job_manager)
) -> BackendResultManager:
    if session_manager is _session_manager:
        return _result_manager
    return BackendResultManager(session_manager, job_manager)

# --- Schemas ---

class SessionCreateRequest(BaseModel):
    metadata: Optional[Dict[str, Any]] = None

class JobCreateRequest(BaseModel):
    reconstruction_mode: Optional[str] = None

class JobStatusUpdateRequest(BaseModel):
    status: str
    error: Optional[str] = None
    result_metadata: Optional[Dict[str, Any]] = None

class ExportRequest(BaseModel):
    destination_filename: str

# --- Error Mappers ---

def map_exception_to_http(e: Exception):
    """Safely maps internal exceptions to HTTP errors without leaking sensitive data."""
    err_str = str(e).lower()
    
    if isinstance(e, SessionConflictError):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
        
    if isinstance(e, SessionManagerError):
        if "not found" in err_str or "does not exist" in err_str or "invalid session id format" in err_str:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
        
    if isinstance(e, InputManagerError):
        if "exceeds maximum allowed" in err_str:
            raise HTTPException(status_code=status.HTTP_413_CONTENT_TOO_LARGE, detail="Upload file is too large")
        if "unsupported file extension" in err_str:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unsupported file extension")
        if "missing" in err_str or "not found" in err_str or "does not exist" in err_str or "invalid session id format" in err_str:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Input file or session not found")
        if "traversal" in err_str or "invalid" in err_str:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid request path")
        if "locked" in err_str:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
        
    if isinstance(e, JobManagerError):
        if "not found" in err_str or "does not exist" in err_str or "invalid job id format" in err_str:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
        if "transition" in err_str or "invalid state" in err_str:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Invalid job state transition")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
        
    if isinstance(e, ResultConflictError):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
        
    if isinstance(e, ResultManagerError):
        if "not found" in err_str:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
        
    raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An unexpected internal error occurred")

# --- Session Endpoints ---

@app.post("/sessions", status_code=status.HTTP_201_CREATED)
def create_session(
    request: Optional[SessionCreateRequest] = None,
    session_manager: BackendSessionManager = Depends(get_session_manager)
):
    try:
        initial_meta = request.metadata if request else {}
        session_id = session_manager.create_session(initial_meta)
        return {"session_id": session_id, "status": "created"}
    except Exception as e:
        map_exception_to_http(e)

@app.get("/sessions/{session_id}")
def get_session(
    session_id: str,
    session_manager: BackendSessionManager = Depends(get_session_manager)
):
    try:
        session_data = session_manager.get_session(session_id)
        return session_data
    except Exception as e:
        map_exception_to_http(e)

@app.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_session(
    session_id: str,
    session_manager: BackendSessionManager = Depends(get_session_manager)
):
    try:
        session_manager.delete_session(session_id)
        return None
    except Exception as e:
        map_exception_to_http(e)

# --- Input Endpoints ---

@app.post("/sessions/{session_id}/inputs", status_code=status.HTTP_201_CREATED)
async def upload_input(
    session_id: str,
    file: UploadFile = File(...),
    input_type: Optional[str] = Form(None),
    input_manager: BackendInputManager = Depends(get_input_manager)
):
    original_filename = file.filename or "unknown"
    temp_path = None
    
    try:
        # Securely stage upload to a temporary file first
        fd, temp_path_str = tempfile.mkstemp()
        temp_path = Path(temp_path_str)
        
        with os.fdopen(fd, 'wb') as out_file:
            shutil.copyfileobj(file.file, out_file)
            
        # Delegate validation and actual staging to BackendInputManager
        record = input_manager.save_input(
            session_id=session_id,
            source_path=temp_path,
            original_filename=original_filename,
            input_type=input_type
        )
        return record
        
    except Exception as e:
        map_exception_to_http(e)
    finally:
        # Cleanup temporary upload artifact
        if temp_path and temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass

@app.get("/sessions/{session_id}/inputs")
def list_inputs(
    session_id: str,
    input_manager: BackendInputManager = Depends(get_input_manager)
):
    try:
        inputs = input_manager.list_inputs(session_id)
        return inputs
    except Exception as e:
        map_exception_to_http(e)

@app.get("/sessions/{session_id}/inputs/{stored_filename}")
def download_input(
    session_id: str,
    stored_filename: str,
    input_manager: BackendInputManager = Depends(get_input_manager)
):
    try:
        file_path = input_manager.get_input_path(session_id, stored_filename)
        return FileResponse(path=file_path, filename=stored_filename)
    except Exception as e:
        map_exception_to_http(e)

@app.delete("/sessions/{session_id}/inputs/{stored_filename}")
def delete_input(
    session_id: str,
    stored_filename: str,
    input_manager: BackendInputManager = Depends(get_input_manager)
):
    try:
        input_manager.delete_input(session_id, stored_filename)
        return {"detail": "Input deleted successfully"}
    except Exception as e:
        map_exception_to_http(e)

# --- Job Endpoints ---

@app.post("/sessions/{session_id}/jobs", status_code=status.HTTP_201_CREATED)
def create_job(
    session_id: str,
    request: JobCreateRequest,
    job_manager: BackendJobManager = Depends(get_job_manager)
):
    try:
        job_id = job_manager.create_job(session_id, request.reconstruction_mode)
        return {"job_id": job_id, "session_id": session_id, "status": "queued"}
    except Exception as e:
        map_exception_to_http(e)

@app.get("/sessions/{session_id}/jobs")
def list_jobs(
    session_id: str,
    job_manager: BackendJobManager = Depends(get_job_manager)
):
    try:
        jobs = job_manager.list_jobs(session_id)
        return jobs
    except Exception as e:
        map_exception_to_http(e)

@app.post("/sessions/{session_id}/jobs/{job_id}/submit", status_code=status.HTTP_202_ACCEPTED)
def submit_job(
    session_id: str,
    job_id: str,
    execution_manager: BackgroundExecutionManager = Depends(get_execution_manager),
    session_manager: BackendSessionManager = Depends(get_session_manager)
):
    """
    Submits a queued job for background execution.
    Returns immediately after queueing.
    """
    try:
        with session_manager.session_lock(session_id):
            execution_manager.submit(job_id, session_id)
        return {"job_id": job_id, "status": "submitted"}
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        map_exception_to_http(e)

@app.get("/jobs/{job_id}")
def get_job(
    job_id: str,
    job_manager: BackendJobManager = Depends(get_job_manager)
):
    try:
        job_data = job_manager.get_job(job_id)
        return job_data
    except Exception as e:
        map_exception_to_http(e)

# --- Result Endpoints ---

@app.get("/sessions/{session_id}/jobs/{job_id}/results")
def list_results(
    session_id: str,
    job_id: str,
    result_manager: BackendResultManager = Depends(get_result_manager)
):
    try:
        results = result_manager.list_results(session_id, job_id)
        return results
    except Exception as e:
        map_exception_to_http(e)

@app.get("/sessions/{session_id}/jobs/{job_id}/results/{result_id}")
def download_result(
    session_id: str,
    job_id: str,
    result_id: str,
    result_manager: BackendResultManager = Depends(get_result_manager)
):
    try:
        file_path = result_manager.get_result_path(session_id, job_id, result_id)
        # Using the actual filename for the download
        return FileResponse(path=file_path, filename=file_path.name)
    except Exception as e:
        map_exception_to_http(e)

@app.post("/sessions/{session_id}/jobs/{job_id}/results/{result_id}/export")
def export_result(
    session_id: str,
    job_id: str,
    result_id: str,
    request: ExportRequest,
    result_manager: BackendResultManager = Depends(get_result_manager)
):
    try:
        metadata = result_manager.export_result(
            session_id, job_id, result_id, request.destination_filename
        )
        return metadata
    except Exception as e:
        map_exception_to_http(e)
