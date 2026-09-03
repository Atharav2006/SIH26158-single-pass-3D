# STEP 4: Backend Input Manager

## Purpose
The `BackendInputManager` provides a secure backend file-staging layer. It accepts files for an existing reconstruction session, sanitizes filenames, enforces file limits, and stores the files strictly within the session's isolated `inputs/` directory.

## Architecture
```
BackendSessionManager
        ↓
BackendInputManager
        ↓
session/<session_id>/inputs/
        ↓
future reconstruction job
```
The Input Manager delegates session existence and workspace resolution to the `BackendSessionManager`, ensuring dry, centralized path enforcement.

## Supported Input Types
This component strictly allowlists extensions to prevent arbitrary binary staging:
- **Video/Images**: `.mp4`, `.mov`, `.avi`, `.mkv`, `.webm`, `.tif`, `.tiff`
- **Sensor/Data**: `.csv`, `.txt`, `.json`, `.yaml`, `.yml`, `.bin`

**IMPORTANT**: This component performs extension-based validation only; it does not guarantee that file contents match the declared extension.

## Filename Sanitization
User-provided filenames are heavily sanitized:
- Directory separators (`/`, `\`) and null bytes are stripped.
- Path characters are replaced with underscores (allowing alphanumeric, `.`, `_`, `-`).
- The final stored filename is structured as `<uuid4>_<sanitized_original_filename>`. 
- The original filename is preserved purely as string metadata, not as a filesystem path.

## Path Security
- The module strictly refuses to follow absolute paths, UNC paths, or traverse (`../`) strings.
- Files are resolved exclusively into `<session_workspace>/inputs/`.
- Paths are resolved using `Path.resolve()` and explicitly verified to descend from the `inputs` folder.

## File Size Limits
- By default, files are constrained to a 10GB limit (configurable via `max_file_size_bytes`).
- Attempting to stage an oversized file immediately raises an `InputManagerError` before any copying occurs.

## Copy Semantics
- Files are duplicated using `shutil.copy2` to preserve safe binary file copying without consuming unbounded memory.
- The source file is never moved or deleted by the manager.
- If a copy operation fails, the destination file is immediately unlinked (deleted) to prevent leaving a corrupted artifact, and no metadata record is created.

## Metadata Structure
When a file is successfully staged, a metadata block is safely appended to the `BackendSessionManager`'s `input_metadata` dictionary under the `files` array:
```json
{
    "stored_filename": "uuid4_sanitized.mp4",
    "original_filename": "source/video.mp4",
    "input_type": "video",
    "size_bytes": 1048576,
    "extension": ".mp4",
    "created_at": "ISO-8601 UTC timestamp"
}
```
Pre-existing unrelated metadata fields in the session remain completely untouched.

## Session Isolation
- Listing or deleting inputs targets exactly one session's `inputs/` directory.
- An attempt to fetch a file belonging to another session will trigger a traversal rejection or a file-not-found exception because the manager locks operations to the caller's `session_id`.

## Error Handling
The `InputManagerError` is raised for:
- Path traversal attempts.
- Extension violations.
- File size violations.
- Missing files/sessions.
- Unsafe destination violations.
- I/O copy errors.

## Tests
A suite of 17 lightweight tests (`tests/backend/test_input_manager.py`) verifies logic strictly outside the reconstruction engine:
- Traversal blocking (`../../`).
- Absolute path blocking (`/etc/passwd`).
- Safe copy completion.
- Metadata persistence.
- Complete isolation.

## Future REST Upload Integration
The future API will accept multipart form uploads, stage them to a temporary OS directory, and pass the temp path into `BackendInputManager.save_input()`.

## Known Limitations
- Validation is purely based on the file extension. There are no magic-byte headers checked to enforce true MIME types.
- Deduplication is based on filename rather than file hash. Two uploads with the exact same content but different filenames will be stored twice.
