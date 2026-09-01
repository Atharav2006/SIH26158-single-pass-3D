# SIH26158 — B5.2 GLOBAL RELATIVE-DEPTH GAUGE ALIGNMENT REPORT

## 1. Objective and Hypothesis
The objective of B5.2 is to establish a globally consistent numerical depth gauge across all 350 monocular depth frames ($D_i$) in the sequence, without imposing absolute metric scale. Given that MiDaS outputs depth up to an arbitrary positive scale $a_i$ and shift $b_i$, raw outputs are structurally consistent (highly correlated) but numerically incompatible across frames.

The core hypothesis is that B2 metric poses can establish accurate sparse geometric correspondences (pose-aware projection), allowing us to solve an unregularized graph optimization for the global gauge parameters $a_i, b_i$ relative to a single fixed anchor frame.

## 2. Mathematical Convention Adopted

**1. $D_i$**: The raw relative depth map of frame $i$, output by MiDaS. We evaluate two representations: $D_{inv}$ (raw inverse depth) and $D_{rel}$ (reciprocal relative depth).
**2. $D_j$**: The raw relative depth map of frame $j$.
**3. $a_{ij}, b_{ij}$**: The local edge affine transformation such that $D_j \approx a_{ij} D_i + b_{ij}$, optimized robustly via Huber loss over valid pose-aware correspondences.
**4. $a_i, b_i$**: The global frame parameters that map local depth $D_i$ to the globally aligned gauge $G(D_i)$.
**5. Gauge Fixing**: A single reference frame (e.g., Frame 1) is chosen, such that $a_{ref} = 1$ and $b_{ref} = 0$ precisely, avoiding full rank deficiency.
**6. Primary Solver Objective**: 
   - **Scale**: $\min \sum w_{ij} [\log(a_i) - \log(a_j) - \log(a_{ij})]^2$
   - **Shift**: $\min \sum w_{ij} [b_i - b_j - a_j b_{ij}]^2$
   Solved decoupled using linear sparse least-squares.
**7. Correspondence Method**: Pose-aware geometric correspondence. $D_i$ is unprojected to an arbitrary 3D scale, transformed to frame $j$ using B2 relative metrically-scaled poses (introducing minor mismatch but resolving 2D homography collapse), and projected into pixel $j$ to sample $D_j$.
**8. Optional Regularizer**: No temporal smoothness regularization is enforced in the primary solver. Only data-driven graph edge consistency dictates the solution, preserving high-frequency geometric validity.
**9. Representation Selection Criteria**: A deterministic composite score combining Pearson correlation ($r \uparrow$), normalized RMSE ($\downarrow$), condition number ($\downarrow$), and valid correspondence count ($\uparrow$), heavily favoring stable linear condition.

## 3. Results Summary
The full pipeline completed successfully over the Zurich sequence. The sparse linear solver identified the gauge rapidly, successfully scaling and shifting all 350 frames into a single, unified relative depth space ready for subsequent dense geometry reconstruction.

The B5.2 implementation is structurally complete and validated with deterministic synthetic data assuring mathematically exact gauge recovery.

B5.2 PLAN STATUS: APPROVED FOR IMPLEMENTATION
(Implementation and unit tests completed)
