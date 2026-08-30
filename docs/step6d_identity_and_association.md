# Step 6D: Image Identity and Authoritative Ground-Truth Association Policy

This document defines the identity taxonomy, association hierarchy, schema specifications, and mathematical policy for mapping drone images to ground-truth photogrammetric poses and telemetry in SIH26158.

---

## 1. Association Policy Hierarchy

To maintain complete mathematical rigor and eliminate temporal ambiguity, SIH26158 establishes an explicit 3-tier association hierarchy:

$$\text{Exact Source ID Matching} \succ \text{Explicit Dataset Relationships} \succ \text{Nearest-Neighbor Timestamp Synchronization}$$

1. **Tier 1: Exact Source ID (`EXACT_ID`)** [**Authoritative for Discrete Ground Truth**]:
   - When a dataset provides discrete ground-truth keyframe poses tied to explicit frame identifiers (e.g. `imgid` in Zurich Urban MAV `GroundTruthAGL.csv`), association **must** be performed via exact ID matching:
     $$\text{image\_id} \to \text{imgid} \equiv \text{ground\_truth\_imgid} \implies \text{Ground Truth Pose}$$
   - Guarantees 100% precision, zero temporal ambiguity, and zero artificial time residual.

2. **Tier 2: Explicit Dataset Relationships**:
   - Frame indexes explicitly defined in flight log manifests or video container packet metadata.

3. **Tier 3: Nearest-Neighbor Timestamp Synchronization (`TIMESTAMP_NEAREST`)** [**Authoritative for Continuous Telemetry**]:
   - Used for continuous, asynchronous sensor streams (GNSS at ~30 Hz, IMU at ~10 Hz, Onboard Estimator at ~50 Hz, Barometer).
   - Associates an image timestamp $t_{\text{img}}$ with the nearest telemetry timestamp $t_{\text{sensor}}$ within an enforced tolerance window $\tau$:
     $$\min_{t_{\text{sensor}}} |t_{\text{img}} - t_{\text{sensor}}| \le \tau$$

---

## 2. Image Identity Architecture

In the normalized ingestion schema, every image record explicitly maintains two identifiers:

| Identifier | Type | Scope | Definition |
| :--- | :--- | :--- | :--- |
| **`image_id`** | Integer | Internal SIH26158 | Sequential 1-based index within the processing batch ($1, 2, 3 \dots N$). |
| **`imgid`** | Integer | Dataset-Native | Native image identifier parsed from the original filename (`00001.jpg` $\to 1$) and matching `OnboardGPS.csv` / `GroundTruthAGL.csv`. |

### Updated `images.csv` Schema
```csv
image_id,imgid,filename,timestamp_seconds,width,height
1,1,00001.jpg,7.009129,1920,1080
2,2,00002.jpg,7.042462,1920,1080
...
31,31,00031.jpg,7.988182,1920,1080
```

---

## 3. Keyframe vs. Intermediate-Frame Distinction

The system enforces an explicit separation between two categories of images:

1. **Ground-Truth Keyframes (`EXACT_ID`)**:
   - Images whose native `imgid` is explicitly present in the offline photogrammetric bundle-adjustment ground truth (`GroundTruthAGL.csv`).
   - In Zurich Urban MAV, bundle adjustment was solved at **1 Hz** ($\Delta \text{imgid} = 30$ frames):
     $$\text{Keyframe imgids} \in \{1, 31, 61, 91, 121, 151, 181, 211, 241, 271, 301, 331, \dots\}$$
   - For the 350-image sample, exactly **12 keyframes** exist. All 12 have exact ground-truth poses with $\Delta t = 0.000000\text{ s}$.

2. **Intermediate Video Frames (`UNMATCHED`)**:
   - Intermediate 30 FPS video frames between the 1 Hz keyframes (e.g. `00002.jpg` to `00030.jpg`).
   - **Data Policy**: Intermediate frames are explicitly marked `matched=false`, `association_method="UNMATCHED"`. They are **never** falsely labeled as ground-truth poses.
   - Any dense trajectory populated for intermediate frames in downstream modules (e.g. via SLERP or visual odometry) must be explicitly flagged as `ESTIMATED` or `INTERPOLATED`.

---

## 4. Ground-Truth Association Output Schema

The association table is generated at [outputs/reports/zurich_mav/image_groundtruth_associations.csv](file:///d:/SIH26158-single-pass-3D/outputs/reports/zurich_mav/image_groundtruth_associations.csv):

| Column | Type | Description |
| :--- | :--- | :--- |
| `image_id` | Integer | Internal SIH26158 sequential image ID |
| `imgid` | Integer | Dataset-native image ID |
| `filename` | String | Image filename (`00001.jpg`) |
| `image_timestamp_seconds` | Float | Microsecond-accurate image timestamp |
| `ground_truth_imgid` | Optional[Int] | Matched ground-truth `imgid` or empty |
| `ground_truth_pose_timestamp_seconds` | Optional[Float] | Matched ground-truth pose timestamp or empty |
| `association_method` | String | `EXACT_ID` or `UNMATCHED` |
| `matched` | Boolean | `true` if keyframe ground truth exists, `false` otherwise |
| `delta_seconds` | Optional[Float] | Temporal difference ($0.0\text{ s}$ for exact matches) |

### Sample Records from `image_groundtruth_associations.csv`
```csv
image_id,imgid,filename,image_timestamp_seconds,ground_truth_imgid,ground_truth_pose_timestamp_seconds,association_method,matched,delta_seconds
1,1,00001.jpg,7.009129,1,7.009129,EXACT_ID,true,0.0
2,2,00002.jpg,7.042462,,,UNMATCHED,false,
...
30,30,00030.jpg,7.965103,,,UNMATCHED,false,
31,31,00031.jpg,7.988182,31,7.988182,EXACT_ID,true,0.0
32,32,00032.jpg,8.021515,,,UNMATCHED,false,
```

---

## 5. Sample Validation & Test Verification

```powershell
pytest -q
```
**Results**:
```text
..................................                                       [100%]
34 passed in 12.38s
```

* **Sample Image Count:** 350 images
* **Exact Ground-Truth Keyframe Matches:** **`12`** (`imgid = 1, 31, 61, 91, 121, 151, 181, 211, 241, 271, 301, 331`)
* **Intermediate Video Frames:** **`338`**
* **Identity Ambiguities / Duplicates:** **`0`**
* **Identity / Association Status:** **`PASS`**
