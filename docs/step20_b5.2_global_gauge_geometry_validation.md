# SIH26158 — B5.3 GLOBAL-GAUGE-AWARE DENSE 3D RECONSTRUCTION VALIDATION

## Scientific Conclusion

**`GLOBAL_GAUGE_GEOMETRY_DEGRADED`**

Despite achieving near-zero internal mathematical residuals during the $a_i, b_i$ linear optimization, the resulting unregularized global depth gauge is physically degenerate and practically destructive. The global alignment severely degrades geometric accuracy due to unchecked exponential parameter drift over the 350-frame sequence.

## Independent Geometric Validation Evidence

### 1. Massive Scale and Shift Drift
Applying the decoupled $a_i = a_j a_{ij}$ solver sequentially forced parameter values to explode over time without loop closures. By Frame 300, $a_{300} \approx 6595$ and $b_{300} \approx -3.8 \times 10^6$.
This unregularized scaling resulted in catastrophic numerical conditioning outside the valid range of physical geometries.

### 2. Invalid Negative Depths
The excessive shift parameters inverted the depth space. Our frame-by-frame analysis (`b5_global_gauge_depth_statistics.json`) revealed that **28.1% of all evaluated depth pixels** were forced into $\leq 0$ values. This means over a quarter of the entire reconstruction was projected physically behind the camera or into an undefined singularity.

### 3. Exploding Reprojection Residuals
When independently validating geometric structure using strict pixel unprojection, B2 pose-transformation, and reprojection (`b5_global_gauge_crossframe_validation.json`):
*   At `(10 -> 11)`, absolute depth discrepancy remained stable ($\sim 12.8$).
*   At `(150 -> 151)`, it blew up to $36.1$.
*   At `(300 -> 301)`, it skyrocketed to $347,611$.
The original local affine fits bounded this residual to $36.3$. The global transformation explicitly destroyed the metric compatibility of adjacent relative frames.

### 4. Hold-Out Validation Failure
When deliberately holding out edges to prevent circular evaluation (`b5_global_gauge_holdout_validation.json`), the solver proved it was highly overfit to the sequential chain. Predicting missing edges yielded absolute shift errors as high as 34 units—massive relative differences in disparity space that invalidate depth layering.

## Summary

The hypothesis that a linearly optimized, temporally unregularized sequential gauge graph could safely harmonize relative monocular depth without metric anchors is **falsified**.

The optimizer perfectly satisfied its internal loss function (scale/shift consistency errors $< 10^{-9}$), but because the scale-shift transformation group over 350 frames is not strongly bounded, it drifted into degenerate parameter regimes.

**Conclusion:** The global gauge alignment as formulated in B5.2 only forces internal variable consistency at the expense of external geometric validity. A strict global gauge *cannot* be applied to hovering monocular sequences without explicit regularizers (e.g., metric depth anchors, physical bounds, or temporal scale-smoothing loss). 

*Do NOT proceed to mesh reconstruction with this gauge.*
