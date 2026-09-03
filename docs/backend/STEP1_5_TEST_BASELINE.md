# Step 1.5 Test Baseline

## 1. Python Environment
Python 3.12.3

## 2. Full Test Collection Result
**Command:** `python -m pytest tests/ --collect-only -q`
**Result:** 183 tests collected, 17 errors in 2.57s. The collection was interrupted with the following summary:
```text
=========================== short test summary info ===========================
ERROR tests/integration/test_b4_neural_reconstruction.py
ERROR tests/integration/test_b5_metric_alignment.py
ERROR tests/integration/test_b5_relative_reconstruction.py
ERROR tests/unit/test_b2_optimizer.py - ValueError: mutable default <class 'n...
ERROR tests/unit/test_b4_checkpoint.py
ERROR tests/unit/test_b4_dataset.py
ERROR tests/unit/test_b4_model.py
ERROR tests/unit/test_b4_renderer.py
ERROR tests/unit/test_b4b_depth_loss.py
ERROR tests/unit/test_b4b_depth_prior.py
ERROR tests/unit/test_b5_depth_prior.py
ERROR tests/unit/test_b5_metric_alignment.py
ERROR tests/unit/test_b5_rays.py
ERROR tests/unit/test_b5_relative_geometry.py
ERROR tests/unit/test_b5_scale_alignment.py
ERROR tests/unit/test_b5_unprojection.py
ERROR tests/unit/test_sensor_factors.py - ValueError: mutable default <class ...
!!!!!!!!!!!!!!!!!! Interrupted: 17 errors during collection !!!!!!!!!!!!!!!!!!!
```

## 3. Missing Dependencies
- **`torch`**: Listed in `pyproject.toml` under `dependencies` (along with `numpy` and `opencv-python`), but it is not installed in the current environment. This caused the `ModuleNotFoundError` across 15 test files.
- **`ffprobe`**: A system dependency used by `subprocess` inside `src/ingestion/video_metadata.py`, which is currently missing from the system path.

## 4. sensor_fusion Failure
**File:** `src/sensor_fusion/sensor_factors.py` (Line 89)
**Error:** `ValueError: mutable default <class 'numpy.ndarray'> for field gravity_world is not allowed: use default_factory`
**Reason:** The `@dataclass` decorator strictly forbids using mutable types (like `numpy.ndarray`) as default values for class attributes, because the single instance would be shared across all class instances. The implementation currently assigns `gravity_world: np.ndarray = np.array([0.0, 0.0, -9.80665], dtype=np.float64)` directly, instead of using `field(default_factory=...)`.

## 5. Impact on Existing Tests
The `pytest tests/` command fails entirely during the collection phase, preventing the test suite from running as a whole. However, isolated tests that do not import `torch` or `sensor_factors.py` can still be executed if their paths are explicitly passed to `pytest`.

## 6. Lightweight Tests That Can Run
**Command Executed:** `python -m pytest tests/unit/test_session_isolation.py tests/unit/test_video_metadata.py`
**Result:** 
- `test_session_isolation.py`: **PASS** (2 tests executed successfully)
- `test_video_metadata.py`: **FAIL** (Failed due to `FileNotFoundError: [WinError 2] The system cannot find the file specified` when `subprocess.Popen` attempted to invoke `ffprobe`).

## 7. Backend Development Implications
- Backend CI/CD and local development pipelines cannot run a blanket `pytest tests/` command without failing.
- Backend engineers must avoid importing core ML engine code into their API tests unless they are explicitly mocking it, to prevent the `torch` requirement.
- Input validation testing will fail unless `ffmpeg`/`ffprobe` is installed locally, so mocking `subprocess.run` will be necessary for API-level tests.

## 8. Recommended Action
1. **Isolate Backend Tests**: Place API tests in a separate folder (e.g., `tests/backend/`) and run them explicitly, or mark them with `@pytest.mark.backend` to avoid collecting the broken ML tests.
2. **Mock Heavy Dependencies**: Mock `VideoValidator` and `SensorDetector` in API tests to avoid requiring `ffprobe` or system binaries.
3. **Fix Mutable Default**: The `ValueError` in `sensor_factors.py` should be fixed by the ML team.
4. **Environment Setup**: Define a separate `backend` optional dependency block in `pyproject.toml` if backend tests require different packages (like `fastapi`, `httpx`).

## 9. Step 1.5 Conclusion
The test baseline is currently broken due to a combination of missing heavy ML dependencies (`torch`), missing system binaries (`ffprobe`), and a syntax error regarding mutable defaults in a dataclass. This blocks the execution of the full test suite. Backend development can proceed safely, provided that backend tests are isolated and carefully mock the reconstruction engine's interfaces.
