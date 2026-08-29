import sys
import subprocess
import importlib.util
from typing import Tuple

def check_python() -> Tuple[str, str]:
    major, minor, micro = sys.version_info[:3]
    version_str = f"{major}.{minor}.{micro}"
    if major == 3 and minor >= 10:
        return "PASS", f"Python version {version_str}"
    else:
        return "FAIL", f"Python version {version_str} (requires >= 3.10)"

def check_pip() -> Tuple[str, str]:
    try:
        import pip
        return "PASS", f"pip version {pip.__version__}"
    except ImportError:
        return "NOT INSTALLED", "pip package manager not found"

def check_git() -> Tuple[str, str]:
    try:
        res = subprocess.run(["git", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        version_str = res.stdout.strip()
        return "PASS", version_str
    except (subprocess.SubprocessError, FileNotFoundError):
        return "NOT INSTALLED", "Git version control not found in PATH"

def check_package(package_name: str) -> Tuple[str, str]:
    spec = importlib.util.find_spec(package_name)
    if spec is None:
        return "NOT INSTALLED", f"{package_name} is not installed"
    try:
        module = importlib.import_module(package_name)
        version = getattr(module, "__version__", "unknown version")
        return "PASS", f"{package_name} version {version}"
    except Exception as e:
        return "FAIL", f"Failed to import {package_name}: {e}"

def main():
    print("=" * 60)
    print("SIH26158: Environment Verification (Scaffolding Stage)")
    print("=" * 60)
    
    # Dependencies required at this stage
    required_checks = {
        "Python": check_python,
        "pip": check_pip,
        "Git": check_git,
        "opencv-python (cv2)": lambda: check_package("cv2"),
        "PyTorch (torch)": lambda: check_package("torch"),
        "pytest": lambda: check_package("pytest"),
    }
    
    all_passed = True
    print("\n--- Required Dependencies ---")
    for name, check_func in required_checks.items():
        status, details = check_func()
        print(f"[{status:<13}] {name}: {details}")
        if status in ("FAIL", "NOT INSTALLED"):
            all_passed = False
            
    # Future pipeline dependencies (optional at this stage)
    print("\n--- Future Pipeline Dependencies (Informational) ---")
    future_packages = ["open3d"]
    for name in future_packages:
        status, details = check_package(name)
        print(f"[{status:<13}] {name}: {details}")
        
    # Check external binaries (optional at this stage)
    external_binaries = {
        "FFmpeg": ["ffmpeg", "-version"],
        "CMake": ["cmake", "--version"],
        "COLMAP": ["colmap", "help"]
    }
    for name, cmd in external_binaries.items():
        try:
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            print(f"[PASS         ] {name}: executable found in PATH")
        except (subprocess.SubprocessError, FileNotFoundError):
            print(f"[NOT INSTALLED] {name}: executable not found in PATH")

    print("=" * 60)
    if all_passed:
        print("VERIFICATION RESULT: PASS (All required dependencies met)")
        sys.exit(0)
    else:
        print("VERIFICATION RESULT: FAIL (Some required dependencies are missing or failed)")
        sys.exit(1)

if __name__ == "__main__":
    main()
