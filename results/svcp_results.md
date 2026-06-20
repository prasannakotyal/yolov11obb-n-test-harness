# SVCP Verification - Measured Evidence (auto-generated)

Generated: 2026-06-20 11:02  |  Model: `yolo11n-obb.pt` (15 classes)  |  Device: NVIDIA GeForce RTX 3050 Laptop GPU  |  Config: imgsz 1024, conf 0.25, IoU 0.7

Measured values only; pass/fail verdicts are produced by the atomic test suite (`pytest`). Each case maps to the SRD via `harness/traceability.py`.

| Test Case | Name | Key measured value(s) | Traceability (SRD) | Evidence ref. |
|-----------|------|------------------------|--------------------|---------------|
| TC-DATA-01 | Dataset Integrity and Class Distribution | 18451 instances, 15 classes, imbalance 82.4, dups 0 | AI-11, DR-01, DR-03 | `results/TC-DATA-01/` |
| TC-DATA-02 | ODD Coverage and Data Desiderata | 15/15 classes >=5 instances; 24.94% objects <20px | AI-01, AI-11, DR-02, DR-04 | `results/TC-DATA-02/` |
| TC-DATA-03 | Train/Validation/Test Separation | 0 exact dup, 0 near-dup pairs | AI-10, AI-11, DR-06 | `results/TC-DATA-03/` |
| TC-DATA-04 | Label-Quality Review | label error rate 0.0% over 18451 objects | AI-11, DR-01 | `results/TC-DATA-04/` |
| TC-DATA-05 | Data/Model Drift Monitoring | induced KS p=3.22e-47 (detected True); control p=0.3617 | AI-08, FD-08 | `results/TC-DATA-05/` |
| TC-ROB-01 | Image-Corruption Robustness | retention sev1=0.946, sev3=0.748, sev5=0.483 | AI-01, AI-05, PR-04 | `results/TC-ROB-01/` |
| TC-ROB-02 | Geometric-Transform Robustness | mean retention 0.981 over 8 transforms | AI-01, PR-01 | `results/TC-ROB-02/` |
| TC-FUN | Detect/Classify and Output-Schema Validity | 98.0% images with detections; schema valid 49/50 | FR-01, FR-04, FR-05, FR-08, FR-09, FR-10, FR-11, FR-17, AI-03, AI-04 | `results/TC-FUN/` |
| TC-TIM-01 | Per-Frame Inference and End-to-End Latency | inference 15.02ms (p95 62.7); e2e 21.66ms | TR-01, TR-02 | `results/TC-TIM-01/` |
| TC-PERF-01 | Per-Class Detection Accuracy | mAP50=0.6176, mAP50-95=0.4667, P=0.8087, R=0.5805 | PR-04, PR-08, FR-17, AI-08 | `results/TC-PERF-01/` |
| TC-OOD-01 | OOD Input Rejection | AUROC 0.776, FPR@95TPR 1.0, OOD conf-rate 0.49 | PR-05, FR-18, AI-05, SR-08 | `results/TC-OOD-01/` |
