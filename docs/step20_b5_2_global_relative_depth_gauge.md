# Step 20: B5.2 Global Relative-Depth Gauge Alignment

## 1. Variable Definitions
- $D_{inv,i}$: Raw MiDaS inverse depth output for frame $i$.
- $D_{rel,i} = 1 / (D_{inv,i} + \epsilon)$: Normalized relative depth for frame $i$.
- $D_i$: The chosen depth representation for frame $i$ (either $D_{inv}$ or $D_{rel}$).
- $a_{ij}, b_{ij}$: Local pairwise affine parameters mapping depths from frame $i$ into frame $j$'s scale.
- $a_i, b_i$: Global affine parameters mapping frame $i$'s depth into the shared sequence-level gauge.
- $G_i$: The global frame transform function mapping local depth to aligned relative depth.

## 2. Transformation Directions
- **Local Edge Transform:** $D_j \approx a_{ij} D_i + b_{ij}$
- **Global Frame Transform:** $G_i(D_i) = a_i D_i + b_i$

## 3. Mathematical Derivation
The fundamental constraint is that a physical point observed in both frames must have a consistent depth in the global relative gauge. 
Therefore:
$$G_j(D_j) \approx G_i(D_i)$$

Substitute the local edge transform $D_j \approx a_{ij} D_i + b_{ij}$ into $G_j$:
$$G_j(a_{ij} D_i + b_{ij}) \approx G_i(D_i)$$
$$a_j(a_{ij} D_i + b_{ij}) + b_j \approx a_i D_i + b_i$$
$$a_j a_{ij} D_i + a_j b_{ij} + b_j \approx a_i D_i + b_i$$

By equating the coefficients of $D_i$ and the constant term, we derive the exact relationships connecting local measurements to global parameters:
1. **Scale Constraint:** $a_i = a_j a_{ij}$
2. **Shift Constraint:** $b_i = a_j b_{ij} + b_j$

## 4. Pose-Aware Correspondence
Because the camera undergoes motion, static pixel identity $(u,v)$ across frames is physically invalid. 
Pose-aware correspondence provides geometric association while preserving unresolved global depth scale. The pipeline is:
1. Frame $i$ relative depth $D_i$
2. Generate pixel ray using $K_{rect}$
3. Relative 3D point in camera-$i$ gauge
4. Transform via B2 relative camera transformation ($R_{ij}, t_{ij}$)
5. Project into frame $j$
6. Sample $D_j$ at projected location
7. Obtain cross-frame depth pair

Correspondences are rejected if they lack depth validity, fall outside image bounds, have low local confidence, or fail geometric validity checks (e.g., low-parallax thresholds).

## 5. Primary Solver Objective (Graph Constraints Only)
The primary solver enforces the graph constraints without any temporal smoothing priors to objectively measure unregularized drift. 

**Gauge Fixing:** A global relative gauge has infinite affine freedom unless anchored. We enforce:
- $a_{ref} = 1$
- $b_{ref} = 0$

**Formulation:**
Taking the logarithm of the scale constraint yields a linear equation:
$$\log a_i - \log a_j = \log a_{ij}$$

The solver operates in two decoupled linear steps:
1. **Solve for Scales:** Minimize $\sum_{edges} w_{ij} (\log a_i - \log a_j - \log a_{ij})^2$ subject to $\log a_{ref} = 0$.
2. **Solve for Shifts:** With $a_j$ known, minimize $\sum_{edges} w_{ij} (b_i - b_j - a_j b_{ij})^2$ subject to $b_{ref} = 0$.

## 6. Optional Regularized Ablation
**B5.2-A:** Pure graph consistency (Primary solver).
**B5.2-B:** Graph consistency + Explicit temporal smoothness prior.

$$L_{smooth} = \lambda_a \sum_i (\log a_{i+1} - \log a_i)^2 + \lambda_b \sum_i (b_{i+1} - b_i)^2$$

**Interpretation:** Physical camera parameters (ISP gains, focal adjustments) and model priors change continuously. The smoothness penalty acts as a regularizer asserting that the scale/shift mapping shouldn't experience instantaneous spikes between consecutive 30 FPS frames.
**Identifiability Effect:** It risks artificially conditioning an otherwise rank-deficient sequence by forcing stability, obscuring genuine regions where the gauge is truly unidentifiable. This is why it is strictly an ablation and excluded from the primary solver.

## 7. Representation Selection Criteria
Both representations ($D_{inv}$ and $D_{rel}$) will be evaluated prior to sequence alignment using a deterministic composite rule based on:
- Valid correspondence count
- Robust affine residual
- Normalized RMSE
- Pearson correlation
- Condition number
- Parameter variance
- Leave-one-edge-out stability
- Held-out edge prediction error
