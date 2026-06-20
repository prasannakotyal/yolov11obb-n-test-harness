#!/usr/bin/env python3
"""Generate the measured-evidence report for the SVCP verification cases.

Loads the model once, runs each case's measure() function, and writes:
  - results/<TC-ID>/evidence.json : the full measured values for that case.
  - results/svcp_results.{csv,md}  : the measured-value summary + SRD traceability.

Pass/fail verdicts are decided by the atomic test suite (`pytest`); this report
records the measured numbers and how each case maps to the SRD.

Usage:  python run_harness.py
"""

from __future__ import annotations

import csv
import datetime
import json

from harness import common as H
from harness import config as C
from harness import data_tests, model_tests, ood, robustness, traceability


def measure_all(model) -> list[tuple[str, str, dict]]:
    """Run every case and return (tc_id, name, measured) in report order."""
    cases = [
        ("TC-DATA-01", "Dataset Integrity and Class Distribution", data_tests.tc_data_01),
        ("TC-DATA-02", "ODD Coverage and Data Desiderata", data_tests.tc_data_02),
        ("TC-DATA-03", "Train/Validation/Test Separation", data_tests.tc_data_03),
        ("TC-DATA-04", "Label-Quality Review", data_tests.tc_data_04),
        ("TC-DATA-05", "Data/Model Drift Monitoring", data_tests.tc_data_05),
        ("TC-ROB-01", "Image-Corruption Robustness", lambda: robustness.tc_rob_01(model)),
        ("TC-ROB-02", "Geometric-Transform Robustness", lambda: robustness.tc_rob_02(model)),
        ("TC-FUN", "Detect/Classify and Output-Schema Validity", lambda: model_tests.tc_fun(model)),
        ("TC-TIM-01", "Per-Frame Inference and End-to-End Latency", lambda: model_tests.tc_tim_01(model)),
        ("TC-PERF-01", "Per-Class Detection Accuracy", lambda: model_tests.tc_perf_01(model)),
        ("TC-OOD-01", "OOD Input Rejection", lambda: ood.tc_ood_01(model)),
    ]
    rows = []
    for tc_id, name, fn in cases:
        measured = fn()
        rows.append((tc_id, name, measured))
        print(f"  [done] {tc_id}  {name}")
    return rows


def key_metrics(tc_id: str, m: dict) -> str:
    """One-line summary of a case's measured values for the recording table."""
    if tc_id == "TC-DATA-01":
        return f"{m['total_instances']} instances, {m['classes_present']} classes, imbalance {m['class_imbalance_ratio']}, dups {m['duplicate_images']}"
    if tc_id == "TC-DATA-02":
        return f"{m['classes_covered_min5']}/{m['target_classes']} classes >=5 instances; {m['objects_below_min_px_pct']}% objects <{C.MIN_OBJECT_PX}px"
    if tc_id == "TC-DATA-03":
        return f"{m['exact_duplicates_within_val']} exact dup, {m['near_duplicate_pairs_within_val']} near-dup pairs"
    if tc_id == "TC-DATA-04":
        return f"label error rate {m['label_error_rate_pct']}% over {m['objects']} objects"
    if tc_id == "TC-DATA-05":
        return (
            f"induced KS p={m['induced_ks_p']} (detected {m['induced_drift_detected']}); control p={m['control_ks_p']}"
        )
    if tc_id == "TC-ROB-01":
        s = m["severity_mean_retention"]
        return f"retention sev1={s[1]}, sev3={s[3]}, sev5={s[5]}"
    if tc_id == "TC-ROB-02":
        return f"mean retention {m['mean_retention']} over {m['transforms']} transforms"
    if tc_id == "TC-PERF-01":
        o = m["overall"]
        return f"mAP50={o['mAP50']}, mAP50-95={o['mAP50_95']}, P={o['mean_precision']}, R={o['mean_recall']}"
    if tc_id == "TC-TIM-01":
        return f"inference {m['inference_ms_mean']}ms (p95 {m['inference_ms_p95']}); e2e {m['end_to_end_ms_mean']}ms"
    if tc_id == "TC-FUN":
        return (
            f"{m['detection_rate_pct']}% images with detections; schema valid {m['schema_valid_images']}/{m['images']}"
        )
    if tc_id == "TC-OOD-01":
        return f"AUROC {m['AUROC']}, FPR@95TPR {m['FPR_at_95TPR']}, OOD conf-rate {m['ood_confident_detection_rate']}"
    return json.dumps(m)[:120]


def write_reports(rows: list[tuple[str, str, dict]], header: str) -> None:
    for tc_id, _, measured in rows:
        H.save_json(measured, C.RESULTS / tc_id / "evidence.json")

    with open(C.RESULTS / "svcp_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Test Case", "Name", "Key measured value(s)", "Traceability (SRD)", "Evidence ref."])
        for tc_id, name, measured in rows:
            w.writerow(
                [
                    tc_id,
                    name,
                    key_metrics(tc_id, measured),
                    ", ".join(traceability.trace_for(tc_id)),
                    f"results/{tc_id}/",
                ]
            )

    md = [
        "# SVCP Verification - Measured Evidence (auto-generated)",
        "",
        header,
        "",
        "Measured values only; pass/fail verdicts are produced by the atomic test suite (`pytest`). "
        "Each case maps to the SRD via `harness/traceability.py`.",
        "",
        "| Test Case | Name | Key measured value(s) | Traceability (SRD) | Evidence ref. |",
        "|-----------|------|------------------------|--------------------|---------------|",
    ]
    for tc_id, name, measured in rows:
        md.append(
            f"| {tc_id} | {name} | {key_metrics(tc_id, measured)} | {', '.join(traceability.trace_for(tc_id))} | `results/{tc_id}/` |"
        )
    (C.RESULTS / "svcp_results.md").write_text("\n".join(md) + "\n")


def main() -> None:
    model = H.load_model()
    device = H.device_name()
    print("=" * 70)
    print(f"SVCP verification harness  |  model: {C.MODEL_PATH.name}  |  device: {device}")
    print(f"config: imgsz {C.IMGSZ}  conf {C.CONF}  iou {C.IOU}")
    print("=" * 70)

    rows = measure_all(model)
    header = (
        f"Generated: {datetime.datetime.now():%Y-%m-%d %H:%M}  |  Model: `{C.MODEL_PATH.name}` "
        f"({C.NUM_CLASSES} classes)  |  Device: {device}  |  Config: imgsz {C.IMGSZ}, conf {C.CONF}, IoU {C.IOU}"
    )
    write_reports(rows, header)

    print("\n" + "=" * 70)
    print(f"wrote {len(rows)} cases to {C.RESULTS / 'svcp_results.md'}")
    print("run `pytest` for the atomic pass/fail verdicts mapped to each TC and SRD requirement.")
    print("=" * 70)


if __name__ == "__main__":
    main()
