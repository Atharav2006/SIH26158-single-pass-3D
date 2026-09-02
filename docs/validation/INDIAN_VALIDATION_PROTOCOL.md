# SIH26158 Indian Dataset Validation Protocol

## 1. Why Multiple Indian Datasets Are Required
No single dataset captures the complexity of Indian scenes. We require distinct datasets to validate dense urban geometry (Delhi/Roorkee), informal settlements (slums), dense vegetation (Gujarat forest), disaster terrain (Nagaland landslide), and dynamic objects (Ahmedabad traffic).

## 2. Role of Each Dataset
Each dataset is registered with a `reconstruction_role` (e.g., `PRIMARY_3D_BENCHMARK`, `VIDEO_ONLY_TEST`, `DYNAMIC_OBJECT`). The engine adapts its pipeline validation based on this role rather than treating all inputs identically.

## 3. Metric vs Relative Validation (Fail-Closed)
Metric scale is **never** inferred simply because GPS or RTK metadata exists. The system explicitly fails closed on metric geometry unless a verified `metric_anchor_category` (like an explicit ground-truth LiDAR scan or GCP list) is present. Otherwise, the dataset defaults to `READY_FOR_RELATIVE_VALIDATION`.

## 4. Dataset Isolation
Every dataset is isolated in its own workspace (`datasets/india_validation/<dataset_id>`). Validation sessions operate independently. Merging poses, calibration, or reference geometry between datasets is strictly prohibited to prevent cross-contamination.

## 5. Permission/License Handling
Datasets marked `RESEARCH_ONLY` must include local copies of the corresponding license/consent forms prior to validation. If `license_metadata_present` is false during intake, the contract evaluates to `NOT_READY`.

## 6. No-Result-Before-Access Rule
The system strictly prevents the generation of fake or synthetic point clouds. A dataset must transition through `requested` -> `approved` -> `downloaded` -> `verified` -> `ready for evaluation` before the `EvaluationResultSchema` is populated.

## 7. Successful B6.3 Experiment
A successful validation experiment occurs when:
1. The real dataset is locally available.
2. The Intake Validator confirms structural integrity.
3. The Validation Contract classifies the dataset appropriately (Metric vs Relative).
4. The general B6 pipeline runs and populates the `EvaluationResultSchema` without ad-hoc codebase modifications.
