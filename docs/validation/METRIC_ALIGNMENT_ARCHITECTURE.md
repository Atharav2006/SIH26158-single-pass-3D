# Metric Alignment Architecture

## 1. Overview
The SIH26158 `MetricAligner` module computes a 7-DoF absolute similarity transformation (scale, rotation, and translation) to bring a relative photogrammetric reconstruction (e.g., COLMAP output) into a metric, real-world coordinate system using independently surveyed Ground Control Points (GCPs).

## 2. Mathematics (Umeyama's Algorithm / Absolute Orientation)
The module implements the standard closed-form solution to the absolute orientation problem (Umeyama, 1991) to minimize the sum of squared errors between the source reconstruction coordinates and the target metric GCP coordinates.

Given $N$ corresponding points in the source frame $P \in \mathbb{R}^{3 \times N}$ and target frame $Q \in \mathbb{R}^{3 \times N}$:
1. **Centering**: The points are centered by subtracting their respective centroids ($\mu_P$, $\mu_Q$).
2. **Covariance**: A $3 \times 3$ cross-covariance matrix $H = \frac{1}{N} \sum_{i=1}^N (P_i - \mu_P)(Q_i - \mu_Q)^T$ is computed.
3. **SVD**: Singular Value Decomposition is applied to $H$, yielding $H = U \Sigma V^T$.
4. **Rotation**: The rotation matrix is $R = V S U^T$, where $S = \text{diag}(1, 1, \det(V U^T))$ to ensure a valid rotation (reflection prevention).
5. **Scale**: The uniform scale factor $s$ is estimated as $s = \frac{\text{tr}(S \Sigma)}{\sigma_P^2}$, where $\sigma_P^2$ is the variance of the source points.
6. **Translation**: The translation vector is $t = \mu_Q - s R \mu_P$.

The final transformed point for any source point $x$ is:
$x_{metric} = s R x + t$

## 3. Control / Checkpoint Separation
A core fail-closed requirement of SIH26158 is preventing checkpoint leakage.
- The `MetricAligner.align()` API explicitly accepts **only** control points. It does not accept the full GCP list.
- Checkpoints are held completely out of the SVD calculation. 
- After the scale, rotation, and translation are computed, the transformation is **frozen**. The caller is responsible for evaluating the independent checkpoints using the `AlignmentResult.transform()` method.

## 4. Failure Conditions (Fail-Closed Logic)
The module strictly rejects mathematically unstable or physically impossible alignments:
- `INSUFFICIENT_POINTS`: Fewer than 3 control points provided.
- `INVALID_COORDINATES`: Presence of NaN or Inf values.
- `DUPLICATE_POINTS`: Source or target points contain exact duplicates, artificially inflating the point count without adding spatial constraints.
- `COLLINEAR_DEGENERATE`: The control points lie on a line or are otherwise rank-deficient (covariance rank < 2).
- `UNSTABLE_SCALE`: The scale factor resolves to a non-positive number or the source variance is negligibly small (e.g., $< 10^{-12}$), making scale identification unstable.

## 5. Integration Flow
The module is integrated into the B6 generalized pipeline via the `MetricDepthBackend` in `src/reconstruction/reconstruction_backend.py`. The standard flow is:

1. **Relative Reconstruction**: The B6 generalized engine (via COLMAP) produces an unscaled point cloud and relative camera trajectory.
2. **GCP Observation Extraction**: Using standard metadata, the system identifies the unscaled 3D coordinates corresponding to the physical control GCPs.
3. **Similarity Estimation**: `MetricAligner.align()` is called with the relative 3D points as the source and the physical surveyed GCPs as the target.
4. **Transformation**: The estimated 7-DoF transform is applied to the full relative point cloud and camera trajectory.
5. **Validation**: Checkpoint GCPs are transformed and their RMSE is evaluated to confirm absolute scale correctness.
