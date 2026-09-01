import pytest
import json
import csv
from pathlib import Path

OUT_DIR = Path("outputs/reports/zurich_mav/b2")
CSV_PATH = OUT_DIR / "b2_fused_trajectory.csv"
DIAG_PATH = OUT_DIR / "b2_fusion_diagnostics.json"
ABLATION_PATH = OUT_DIR / "b0_b1_b2_ablation.json"

def test_b2_outputs_exist():
    # Only test if the pipeline has been run
    if not CSV_PATH.exists() or not DIAG_PATH.exists():
        pytest.skip("B2 pipeline not yet run")
        
    assert CSV_PATH.is_file()
    assert DIAG_PATH.is_file()
    assert ABLATION_PATH.is_file()

def test_b2_csv_schema():
    if not CSV_PATH.exists():
        pytest.skip()
        
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        row = next(reader)
        
        expected_keys = [
            "imgid", "timestamp", "x", "y", "z",
            "qx", "qy", "qz", "qw",
            "vx", "vy", "vz",
            "gyro_bias_x", "gyro_bias_y", "gyro_bias_z",
            "accel_bias_x", "accel_bias_y", "accel_bias_z"
        ]
        
        for k in expected_keys:
            assert k in row

def test_b2_diagnostics():
    if not DIAG_PATH.exists():
        pytest.skip()
        
    with open(DIAG_PATH, "r", encoding="utf-8") as f:
        diag = json.load(f)
        
    assert "optimization_summary" in diag
    summary = diag["optimization_summary"]
    
    # Check convergence was reported
    assert "CONVERGED" in summary["optimizer_status"]
    
    # Check cost reduction
    assert summary["initial_cost"] >= summary["final_cost"]
    
    eval_stats = diag["ground_truth_evaluation"]
    assert "B2_direct_metric_ate_rmse_m" in eval_stats
    
def test_ablation_schema():
    if not ABLATION_PATH.exists():
        pytest.skip()
        
    with open(ABLATION_PATH, "r", encoding="utf-8") as f:
        ab = json.load(f)
        
    assert "Run_A_Visual_Only" in ab["runs"]
    assert "Run_B_Visual_GPS" in ab["runs"]
    assert "Run_C_Visual_GPS_IMU" in ab["runs"]
