# Step 7B: Zurich Urban MAV 350-Image Dataset Manifest

This document records the verification, structural integrity inspection, and physical manifest for the full 350-image sequence from the **Zurich Urban MAV Dataset (AGZ Subset)** prepared for the Classical COLMAP Structure-from-Motion Baseline (B0).

---

## 1. Executive Summary

| Verification Property | Requirement | Inspected Value | Status |
| :--- | :--- | :--- | :---: |
| **Source Directory** | `D:\SIH26158\datasets\zurich_mav\AGZ_subset\MAV Images` | `D:\SIH26158\datasets\zurich_mav\AGZ_subset\MAV Images` | **PASS** |
| **Total Images** | Exactly 350 | **350 physical JPEG files** | **PASS** |
| **Native ID Range (`imgid`)** | 1 to 350 | **`imgid ∈ [1, 350]`** | **PASS** |
| **Filename Naming Schema** | `00001.jpg` to `00350.jpg` | 100% 5-digit zero-padded | **PASS** |
| **Resolution Consistency** | $1920 \times 1080$ | **350 / 350 (100.0%)** | **PASS** |
| **Image Readability** | All files uncorrupted & decodable | **350 / 350 (100.0%)** | **PASS** |
| **Metadata Consistency** | `images.csv` $\leftrightarrow$ Disk | Identical IDs, timestamps & filenames | **PASS** |
| **Total Disk Footprint** | Complete sample volume | **`112.35 MB`** (117,805,956 bytes) | **PASS** |
| **Manifest Status** | Pre-reconstruction readiness | **`B0 IMAGE MANIFEST: PASS`** | **PASS** |

---

## 2. File Size & Storage Distribution

* **Total Dataset Size**: **`117,805,956 bytes`** ($112.35\text{ MB}$)
* **Minimum File Size**: **`264,422 bytes`** ($258.2\text{ KB}$, `00001.jpg`)
* **Maximum File Size**: **`387,020 bytes`** ($377.9\text{ KB}$, `00346.jpg`)
* **Mean File Size**: **`336,588.45 bytes`** ($328.7\text{ KB}$)
* **Median File Size**: **`339,235.00 bytes`** ($331.3\text{ KB}$)
* **Standard Deviation**: **`24,196.88 bytes`**

---

## 3. Image Dimensions & Resolution Distribution

| Resolution | Format | File Count | Percentage |
| :---: | :---: | :---: | :---: |
| **$1920 \times 1080$** (Full HD) | JPEG (`image/jpeg`) | **350** | **100.0%** |
| *Other / Corrupted* | - | **0** | **0.0%** |

---

## 4. Manifest Sample (First & Last Records)

| `image_id` | `imgid` | `filename` | `timestamp_seconds` | `width` | `height` | `file_size_bytes` | `readable` |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **1** | 1 | `00001.jpg` | `0.000000` | 1920 | 1080 | 264,422 | `true` |
| **2** | 2 | `00002.jpg` | `0.033333` | 1920 | 1080 | 288,574 | `true` |
| **3** | 3 | `00003.jpg` | `0.066667` | 1920 | 1080 | 291,142 | `true` |
| ... | ... | ... | ... | ... | ... | ... | ... |
| **348** | 348 | `00348.jpg` | `11.566667` | 1920 | 1080 | 382,914 | `true` |
| **349** | 349 | `00349.jpg` | `11.600000` | 1920 | 1080 | 385,410 | `true` |
| **350** | 350 | `00350.jpg` | `11.633333` | 1920 | 1080 | 384,102 | `true` |

---

## 5. Machine-Readable Artifacts

* **CSV Manifest**: [outputs/reports/zurich_mav/b0_image_manifest.csv](file:///d:/SIH26158-single-pass-3D/outputs/reports/zurich_mav/b0_image_manifest.csv)
* **JSON Manifest**: [outputs/reports/zurich_mav/b0_image_manifest.json](file:///d:/SIH26158-single-pass-3D/outputs/reports/zurich_mav/b0_image_manifest.json)
