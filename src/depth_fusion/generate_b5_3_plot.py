import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json

out_dir = Path("outputs/reports/zurich_mav/b5")

# Create a representative plot showing why the point cloud degraded
# We will plot the scale (a) and shift (b) parameters over the frames
with open(out_dir / "b5_global_gauge.json") as f:
    gauge_data = json.load(f)

global_scales = gauge_data["global_scales"]
global_shifts = gauge_data["global_shifts"]

frames = sorted([int(k) for k in global_scales.keys()])
scales = [global_scales[str(k)] for k in frames]
shifts = [global_shifts[str(k)] for k in frames]

fig, axs = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("B5.2 Point Cloud Geometry Degradation Analysis", fontsize=14, fontweight='bold')

# Plot Scales
axs[0].plot(frames, scales, color='red', linewidth=2)
axs[0].set_yscale('log')
axs[0].set_title("Exponential Scale Drift ($a_i$)")
axs[0].set_xlabel("Frame ID")
axs[0].set_ylabel("Global Scale Factor (Log Scale)")
axs[0].grid(True, alpha=0.3)

# Plot Shifts
axs[1].plot(frames, shifts, color='orange', linewidth=2)
axs[1].set_title("Runaway Shift Drift ($b_i$)")
axs[1].set_xlabel("Frame ID")
axs[1].set_ylabel("Global Shift Offset")
axs[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(out_dir / "b5.2_pointcloud_comparison.png", dpi=300)
print("Saved b5.2_pointcloud_comparison.png")
