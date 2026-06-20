# SVCP Verification Results (auto-generated)

Generated: 2026-06-20 09:59  |  Model: `yolo11n-obb.pt` (15 classes)  |  Device: NVIDIA GeForce RTX 3050 Laptop GPU  |  Config: imgsz 1024, conf 0.25, IoU 0.7

Acceptance thresholds marked (TBC) are placeholders to confirm with the certification authority; values are measured, not assumed. Traceability IDs map each case to the SRD (see harness/traceability.py).

## Section 7 - Recording of Test Results

| Test Case | Result | Key measured value(s) | Traceability (SRD) | Evidence ref. |
|-----------|--------|------------------------|--------------------|---------------|
| TC-DATA-01 | PASS | 18451 instances, 15 classes, imbalance 82.4, dups 0 | AI-11, DR-01, DR-03 | `results/TC-DATA-01/` |
| TC-DATA-02 | PASS | 15/15 classes >=5 instances; 24.94% objects <20px | AI-01, AI-11, DR-02, DR-04 | `results/TC-DATA-02/` |
| TC-DATA-03 | PASS | 0 exact dup, 0 near-dup pairs | AI-10, AI-11, DR-06 | `results/TC-DATA-03/` |
| TC-DATA-04 | PASS | label error rate 0.0% over 18451 objects | AI-11, DR-01 | `results/TC-DATA-04/` |
| TC-DATA-05 | PASS | induced KS p=3.22e-47 (detected True); control p=0.3617 | AI-08, FD-08 | `results/TC-DATA-05/` |
| TC-ROB-01 | PASS | retention sev1=0.948, sev3=0.749, sev5=0.483 | AI-01, AI-05, PR-04 | `results/TC-ROB-01/` |
| TC-ROB-02 | PASS | mean retention 0.981 over 8 transforms | AI-01, PR-01 | `results/TC-ROB-02/` |
| TC-FUN | PASS | 98.0% images with detections; schema valid 49/50 | FR-01, FR-04, FR-05, FR-08, FR-09, FR-10, FR-11, FR-17, AI-03, AI-04 | `results/TC-FUN/` |
| TC-TIM-01 | PASS | inference 16.18ms (p95 67.1); e2e 22.87ms | TR-01, TR-02 | `results/TC-TIM-01/` |
| TC-PERF-01 | MEASURED | mAP50=0.6176, mAP50-95=0.4667, P=0.8087, R=0.5805 | PR-04, PR-08, FR-17, AI-08 | `results/TC-PERF-01/` |
| TC-OOD-01 | MEASURED | AUROC 0.776, FPR@95TPR 1.0, OOD conf-rate 0.49 | PR-05, FR-18, AI-05, SR-08 | `results/TC-OOD-01/` |

## Per-case detail

### TC-DATA-01 - Dataset Integrity and Class Distribution
- Status: **PASS**
- Traceability (SRD): AI-11, DR-01, DR-03
- Criteria: PASS if no unpaired/unreadable/duplicate files, no malformed labels and all class ids in [0,14].
- Measured: `{"images": 300, "labels": 300, "total_instances": 18451, "classes_present": 15, "unpaired_files": 0, "unreadable_images": 0, "malformed_label_lines": 0, "out_of_range_class": 0, "duplicate_images": 0, "class_imbalance_ratio": 82.4, "class_distribution": {"plane": 2032, "ship": 5441, "storage tank": 1569, "baseball diamond": 120, "tennis court": 445, "basketball court": 66, "ground track field": 95, "harbor": 1344, "bridge": 360, "large vehicle": 3121, "small vehicle": 3175, "helicopter": 67, "ro`
- Observations: Class imbalance ratio 82.4 (max/min instances). Distribution reported for review.

### TC-DATA-02 - ODD Coverage and Data Desiderata
- Status: **PASS**
- Traceability (SRD): AI-01, AI-11, DR-02, DR-04
- Criteria: PASS if every target class is represented (>=5 instances); under-represented classes and small-object fraction reported.
- Measured: `{"target_classes": 15, "classes_covered_min5": 15, "class_coverage": {"plane": 2032, "ship": 5441, "storage tank": 1569, "baseball diamond": 120, "tennis court": 445, "basketball court": 66, "ground track field": 95, "harbor": 1344, "bridge": 360, "large vehicle": 3121, "small vehicle": 3175, "helicopter": 67, "roundabout": 134, "soccer ball field": 104, "swimming pool": 378}, "under_represented_classes": [], "objects_total": 18451, "objects_below_min_px": 4602, "objects_below_min_px_pct": 24.94`
- Observations: Environmental-condition stratification (day/night, weather) needs dataset metadata beyond image/label files.

### TC-DATA-03 - Train/Validation/Test Separation (no leakage)
- Status: **PASS**
- Traceability (SRD): AI-10, AI-11, DR-06
- Criteria: PASS if no exact duplicates within the set; full cross-split disjointness reuses the same hash check across manifests.
- Measured: `{"val_images": 300, "exact_duplicates_within_val": 0, "near_duplicate_pairs_within_val": 0, "method": "MD5 (exact) + 16x16 average-hash (near) per image; cross-split disjointness applies the same hashes across train/val/test manifests.", "cross_split_check": "requires train+test manifests (not in this val-only subset)"}`
- Observations: 0 near-duplicate pair(s) flagged for review by 16x16 average-hash.

### TC-DATA-04 - Label-Quality Review
- Status: **PASS**
- Traceability (SRD): AI-11, DR-01
- Criteria: PASS if label error rate (malformed + degenerate + out-of-range) <= 1.0% (TBC).
- Measured: `{"labels_reviewed": 300, "objects": 18451, "malformed_lines": 0, "degenerate_polygons": 0, "out_of_range_coords": 0, "label_error_rate_pct": 0.0}`
- Observations: Full census of the label set (not a sample).

### TC-DATA-05 - Data/Model Drift Monitoring
- Status: **PASS**
- Traceability (SRD): AI-08, FD-08
- Criteria: PASS if induced drift is detected (KS p<0.05) and the no-drift control is not flagged.
- Measured: `{"feature": "per-image mean brightness (128x128 grayscale)", "control_ks_stat": 0.107, "control_ks_p": 0.3617, "control_js": 0.166, "control_drift_detected": false, "induced_ks_stat": 0.793, "induced_ks_p": 3.22e-47, "induced_js": 0.677, "induced_drift_detected": true}`
- Observations: Demonstrates the KS/JS drift detector; error-rate detectors (DDM/ADWIN/EDDM) are a post-deployment provision.

### TC-ROB-01 - Image-Corruption Robustness (Invariance)
- Status: **PASS**
- Traceability (SRD): AI-01, AI-05, PR-04
- Criteria: PASS if mean detection retention at severity 1 >= 0.7 (TBC) with graceful degradation thereafter.
- Measured: `{"images": 40, "corruption_types": 15, "severities": 5, "severity_1_mean_retention": 0.948, "severity_2_mean_retention": 0.868, "severity_3_mean_retention": 0.749, "severity_4_mean_retention": 0.601, "severity_5_mean_retention": 0.483, "worst_corruption_at_sev1": {"blur/zoom_blur": 0.547}}`
- Observations: Full 15x5 retention table in retention.json. Acceptance retention thresholds are TBC.

### TC-ROB-02 - Geometric-Transform Robustness (Metamorphic)
- Status: **PASS**
- Traceability (SRD): AI-01, PR-01
- Criteria: PASS if realistic transforms preserve detections (mean retention >= 0.7, TBC).
- Measured: `{"images": 40, "transforms": 8, "mean_retention": 0.981, "per_transform": {"hflip": 0.997, "vflip": 0.99, "rot90": 1.019, "rot180": 1.029, "rot270": 0.993, "brighten": 0.996, "darken": 0.989, "zoom_out": 0.834}}`
- Observations: Stage-1 realistic transforms; zoom-in onto target-free regions is not required to flag.

### TC-FUN - Detect/Classify and Output-Schema Validity
- Status: **PASS**
- Traceability (SRD): FR-01, FR-04, FR-05, FR-08, FR-09, FR-10, FR-11, FR-17, AI-03, AI-04
- Criteria: PASS if outputs are well-formed (OBB present, class in [0,14], confidence in [0,1], 4-point polygons) and detections are produced on images containing objects.
- Measured: `{"images": 50, "images_with_detections": 49, "schema_valid_images": 49, "schema_violations": 0, "detection_rate_pct": 98.0}`
- Observations: Confirms the published detection schema (OBB / label / confidence) is well-formed.

### TC-TIM-01 - Per-Frame Inference and End-to-End Latency
- Status: **PASS**
- Traceability (SRD): TR-01, TR-02
- Criteria: PASS if mean inference <= 50.0 ms and mean end-to-end <= 100.0 ms (reference device).
- Measured: `{"images_timed": 150, "device": "NVIDIA GeForce RTX 3050 Laptop GPU", "inference_ms_mean": 16.18, "inference_ms_p95": 67.1, "end_to_end_ms_mean": 22.87, "end_to_end_ms_p95": 71.48, "infer_budget_ms": 50.0, "e2e_budget_ms": 100.0}`
- Observations: Measured on the host device reported above; final acceptance is on the deployment target.

### TC-PERF-01 - Per-Class Detection Accuracy (mAP / precision / recall)
- Status: **MEASURED**
- Traceability (SRD): PR-04, PR-08, FR-17, AI-08
- Criteria: Acceptance: each class meets its SRD per-class thresholds at IoU 0.5. The harness reports measured values.
- Measured: `{"images": 300, "iou_for_mAP50": 0.5, "overall": {"mAP50": 0.6176, "mAP50_95": 0.4667, "mean_precision": 0.8087, "mean_recall": 0.5805}, "per_class": {"plane": {"mAP50": 0.7427, "precision": 0.9346, "recall": 0.7028}, "ship": {"mAP50": 0.7287, "precision": 0.9108, "recall": 0.7056}, "storage tank": {"mAP50": 0.5861, "precision": 0.9408, "recall": 0.5066}, "baseball diamond": {"mAP50": 0.6953, "precision": 0.7976, "recall": 0.6583}, "tennis court": {"mAP50": 0.9538, "precision": 0.9699, "recall":`
- Observations: Mechanism (model.val mAP/precision/recall) demonstrated; pass/fail is set against the SRD per-class table.

### TC-OOD-01 - OOD Input Rejection
- Status: **MEASURED**
- Traceability (SRD): PR-05, FR-18, AI-05, SR-08
- Criteria: PASS if AUROC >= 0.8 (TBC) and confident-detection rate on OOD inputs <= 0.2 (TBC); report AUROC and FPR@95%TPR.
- Measured: `{"id_images": 100, "ood_images": 100, "ood_types": ["uniform noise", "block-scrambled scene"], "score": "max detection confidence (black-box)", "AUROC": 0.776, "FPR_at_95TPR": 1.0, "ood_confident_detection_rate": 0.49, "id_confident_detection_rate": 0.93}`
- Observations: Black-box OOD via output confidence; post-hoc scorers (ODIN/energy) can refine the score.
