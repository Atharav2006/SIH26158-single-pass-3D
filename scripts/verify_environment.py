import sys
import subprocess
import importlib.util
from typing import Tuple

def check_python() -> Tuple[str, str]:
    major, minor, micro = sys.version_info[:3]
    version_str = f"{major}.{minor}.{micro}"
    if major == 3 and minor >= 10:
        return "READY", f"Python version {version_str}"
    else:
        return "BLOCKED", f"Python version {version_str} (requires >= 3.10)"

def check_pip() -> Tuple[str, str]:
    try:
        import pip
        return "READY", f"pip version {pip.__version__}"
    except ImportError:
        return "NOT INSTALLED", "pip package manager not found"

def check_git() -> Tuple[str, str]:
    try:
        res = subprocess.run(["git", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        version_str = res.stdout.strip()
        return "READY", version_str
    except (subprocess.SubprocessError, FileNotFoundError):
        return "NOT INSTALLED", "Git version control not found in PATH"

def check_package(package_name: str) -> Tuple[str, str]:
    spec = importlib.util.find_spec(package_name)
    if spec is None:
        return "NOT INSTALLED", f"{package_name} is not installed"
    try:
        module = importlib.import_module(package_name)
        version = getattr(module, "__version__", "unknown version")
        return "READY", f"{package_name} version {version}"
    except Exception as e:
        return "BLOCKED", f"Failed to import {package_name}: {e}"

def check_pytorch() -> Tuple[str, str]:
    try:
        import torch
    except ImportError:
        print("PyTorch info check: NOT INSTALLED")
        return "NOT INSTALLED", "PyTorch is not installed"
    except Exception as e:
        print("PyTorch info check: BLOCKED (Import failed)")
        return "BLOCKED", f"Failed to import PyTorch: {e}"

    version = getattr(torch, "__version__", "unknown version")
    cuda_build = getattr(torch.version, "cuda", "None")
    cuda_available = torch.cuda.is_available()

    print(f"PyTorch Version: {version}")
    print(f"PyTorch CUDA Build: {cuda_build}")
    print(f"CUDA Availability: {cuda_available}")

    if not cuda_available:
        print("GPU Name: N/A")
        print("GPU Total Memory: N/A")
        return "BLOCKED", "PyTorch CUDA support is unavailable"

    try:
        device_count = torch.cuda.device_count()
        if device_count == 0:
            print("GPU Name: None detected")
            print("GPU Total Memory: N/A")
            return "BLOCKED", "No GPU detected by PyTorch"
        
        gpu_name = torch.cuda.get_device_name(0)
        gpu_properties = torch.cuda.get_device_properties(0)
        total_memory_gb = gpu_properties.total_memory / (1024 ** 3)
        
        print(f"GPU Name: {gpu_name}")
        print(f"GPU Total Memory: {total_memory_gb:.2f} GB")

        # Perform a small real CUDA matrix multiplication test
        print("Running CUDA matrix multiplication test...")
        x = torch.ones((2, 2), device="cuda")
        y = torch.matmul(x, x)
        assert torch.allclose(y, torch.full((2, 2), 2.0, device="cuda"))
        print("CUDA matrix multiplication test passed successfully!")

        return "READY", f"PyTorch is ready on GPU {gpu_name} (version {version})"
    except Exception as e:
        print(f"CUDA validation failed: {e}")
        return "BLOCKED", f"CUDA computation failed: {e}"

def main():
    print("=" * 60)
    print("SIH26158: Environment Verification (Step 2 - Rigorous)")
    print("=" * 60)
    
    # Dependencies required at this stage
    required_checks = {
        "Python": check_python,
        "pip": check_pip,
        "Git": check_git,
        "opencv-python (cv2)": lambda: check_package("cv2"),
        "PyTorch (torch)": check_pytorch,
        "pytest": lambda: check_package("pytest"),
    }
    
    all_ready = True
    print("\n--- Required Dependencies ---")
    for name, check_func in required_checks.items():
        status, details = check_func()
        print(f"[{status:<13}] {name}: {details}")
        if status in ("BLOCKED", "NOT INSTALLED"):
            all_ready = False
            
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
            print(f"[READY        ] {name}: executable found in PATH")
        except (subprocess.SubprocessError, FileNotFoundError):
            print(f"[NOT INSTALLED] {name}: executable not found in PATH")

    print("=" * 60)
    if all_ready:
        print("VERIFICATION RESULT: READY (All required dependencies met and CUDA validated)")
        sys.exit(0)
    else:
        print("VERIFICATION RESULT: BLOCKED (Required dependencies missing, failing, or CPU-only)")
        sys.exit(1)

if __name__ == "__main__":
    main()
