# SIH26158 — Indian Validation Suite: Dataset Discovery & Acquisition Plan

## Why Multiple Datasets Are Needed

No single Indian dataset can simultaneously validate:
- **3D Reconstruction geometry** (requires calibrated overlapping UAV images with camera motion)
- **Indian scene diversity** (buildings, slums, forests, disaster zones, infrastructure)
- **Dynamic object handling** (moving traffic at Indian intersections)
- **Product generalization** (arbitrary video → session → reconstruction)

Each dataset addresses a specific subset of validation requirements.

## Dataset Registry Summary

| Priority | Dataset | Location | Images | Type | Access | 3D Suitability |
|----------|---------|----------|--------|------|--------|-----------------|
| 1 | UASG 2023 Delhi | Delhi | 223 | UAV images | Request | PRIMARY_3D_BENCHMARK |
| 2 | UASG 2019 Roorkee | Roorkee | 102 | UAV images | Request | PRIMARY_3D_BENCHMARK |
| 3 | UASG 2019 Slum | Chhattisgarh | 85 | UAV images | Request | SECONDARY_3D_TEST |
| 4 | ManipalUAVid | Manipal | unknown | UAV video | Request | VIDEO_ONLY_TEST |
| 5 | UASG 2023 Forest | Gujarat | 221 | UAV images | Request | SECONDARY_3D_TEST |
| 6 | UASG 2023 Landslide | Nagaland | 141 | UAV images | Request | SECONDARY_3D_TEST |
| 7 | SkyEye | Ahmedabad | 4K video | UAV video | Public | NOT_SUITABLE_FOR_3D |
| 8 | IIIT-H Infrastructure | Hyderabad | unknown | UAV images | Request | SEMANTIC_ONLY |
| 9 | MUAAD | Manipal | unknown | UAV video | Request | NOT_SUITABLE_FOR_3D |
| 10 | DENSEWORLD-115K | Mixed/India | 115K clips | YouTube | Public | NOT_SUITABLE_FOR_3D |

## Capability Matrix (Key Findings)

### Buildings & Urban Structure
- **CONFIRMED**: IIIT-H infrastructure dataset confirms window/storey annotation
- **LIKELY**: UASG Delhi and Roorkee datasets capture dense urban rooftops, walls, and facade geometry at ~2cm GSD from 100m altitude

### Dynamic Objects (Moving Traffic)
- **CONFIRMED**: SkyEye Ahmedabad explicitly provides annotated 4K video of Indian intersection traffic including cars, motorcycles, pedestrians, and auto-rickshaws

### Vegetation & Terrain
- **CONFIRMED**: UASG 2023 Gujarat is described as "Dense Forest" 
- **LIKELY**: UASG 2023 Nagaland captures landslide terrain

### Roads & Infrastructure
- **CONFIRMED**: SkyEye captures named Ahmedabad intersections
- **LIKELY**: Urban datasets contain visible roads, poles, and wiring

## Scientific vs. Product Distinction

### A. Scientific 3D Benchmark
**UASG Delhi (Priority 1)** and **UASG Roorkee (Priority 2)**: Photogrammetric UAV image sets with known GSD, flying height, and professional drone cameras. Suitable for evaluating COLMAP pose estimation and relative depth fusion quality.

### B. Indian Scene Robustness
**UASG Slum (Priority 3)**, **Gujarat Forest (Priority 5)**, **Nagaland Landslide (Priority 6)**: Test the engine on non-standard Indian scenes (informal settlements, dense canopy, disaster terrain).

### C. Dynamic Object Test
**SkyEye Ahmedabad (Priority 7)**: 4K annotated traffic video. Tests dynamic object detection but NOT 3D reconstruction (static overhead viewpoint, no parallax).

### D. Product Demo Data
**ManipalUAVid (Priority 4)**: Video-format UAV data from Indian campus. Validates the B6 video ingestion → session → frame extraction → mode selection pipeline.

## Licensing & Access Restrictions

> [!WARNING]
> All UASG datasets are **RESEARCH_ONLY** with mandatory citation of Dr. Kamal Jain, IIT Roorkee. They explicitly prohibit commercial demonstration without permission.

> [!CAUTION]
> SkyEye's GitHub repo does not include an explicit license file. Its use in a public SIH demo must be verified with the authors first.

> [!NOTE]
> DENSEWORLD-115K is YouTube-sourced. Individual video copyrights belong to original uploaders. Not suitable for any public demo.

## Recommended Priority & Subsets

### MUST_HAVE (Total ~1 GB)
1. **UASG 2023 Delhi** — 50-image subset for initial COLMAP + MiDaS validation (~150 MB)
2. **UASG 2019 Roorkee** — Full 102 images for rapid iteration (~350 MB)

### SHOULD_HAVE (Total ~3 GB additional)
3. **UASG 2019 Slum** — 50-image subset (~80 MB)
4. **ManipalUAVid** — 5 video clips, ~60 sec total (~500 MB)
5. **SkyEye** — 2 intersection clips, ~30 sec each (~300 MB)

### OPTIONAL
6–8. Gujarat Forest, Nagaland Landslide, IIIT-H Infrastructure

### DO_NOT_NEED_YET
9–10. MUAAD, DENSEWORLD-115K

## Storage Estimates (RTX 3050 4GB VRAM)

| Subset | Disk | RAM | VRAM | COLMAP | MiDaS |
|--------|------|-----|------|--------|-------|
| Delhi 50-img | 150 MB | LOW | LOW | ~5 min | ~2 min |
| Roorkee 102-img | 350 MB | LOW | LOW | ~10 min | ~5 min |
| Slum 50-img | 80 MB | LOW | LOW | ~5 min | ~2 min |
| ManipalUAVid clips | 500 MB | MEDIUM | MEDIUM | ~15 min | ~10 min |

## Next Acquisition Steps
1. Send formal request emails to `info.uasg2023@iitr.ac.in` (Items 01, 02, 03) and `info_uasg2019@iitr.ac.in` (Items 01, 03)
2. Fill ManipalUAVid Google Form on GitHub
3. Verify SkyEye license with repository maintainer
4. Upon receipt, place datasets into `D:\SIH26158\datasets\india_validation\<dataset_id>\`
5. Run B6 generalized pipeline on each dataset independently
