# B5.2 GLOBAL GAUGE RETIREMENT

**Status:** EXPERIMENTAL / RETIRED_FOR_PRODUCTION
**Classification:** RESEARCH_NEGATIVE_RESULT

## Purpose of B5.2
The B5.2 global relative-depth gauge alignment phase was designed to establish a universally consistent relative depth scale across a hovering monocular sequence without relying on metric ground truth. It attempted to harmonise frame-to-frame local affine relationships ($D_j \approx a_{ij} D_i + b_{ij}$) into a global linear graph constraint ($a_i = a_j a_{ij}$ and $b_i = a_j b_{ij} + b_j$).

## Observed Failure Mode
B5.3 testing proved that the sequential affine gauge chain is scientifically unstable and geometrically destructive. 
While the unregularized optimizer yielded perfectly small mathematical graph residuals ($< 10^{-9}$ scale consistency error), the lack of physical constraints and loop closures allowed the parameters to experience massive unconstrained drift.

## Explicit Evidence of Degradation
1. **Exponential Scale Drift**: Global scale multiplier $a_{300}$ reached over 6,500x.
2. **Runaway Shift Drift**: Global shift offset $b_{300}$ reached $-3.8 \times 10^6$.
3. **Exploding Geometric Residuals**: Scale/shift-invariant cross-frame point reprojection error skyrocketed from 12.4 to 347,611.71 by frame 300.
4. **Physically Invalid Domains**: 28.1% of all depth pixels were pushed into values $\leq 0$ (behind the camera).
5. **Collapsed Support**: Multi-view support across the globally shifted space dropped back to a strict maximum of 1, defeating dense fusion.
6. **Prediction Instability**: Held-out temporal edges could not be predicted accurately due to severe overfitting to the sequential chain.

## Why Internal Gauge Residual Was Misleading
The linear least-squares solver accurately resolved the mathematical graph constraints for the variables in isolation, pushing algorithmic residuals to zero. However, it accomplished this by inflating variables indiscriminately into non-physical dimensions. The solver was "perfectly correct" mathematically but utterly decoupled from geometric reality.

## Conclusion for Production
B5.2 Global Gauge cannot be safely applied to relative monocular depth sequences lacking strong metric anchors or loop-closure bounding regularizers. Doing so invents false scales and corrupts real geometry.

This approach is retired for production. The code remains for historical and research reference but **must not** be invoked in the primary metric or relative reconstruction pipeline.
