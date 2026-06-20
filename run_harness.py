#!/usr/bin/env python3
"""Run the verification harness end-to-end and emit SVCP-format results.

Outputs (under results/):
  - svcp_results.csv / svcp_results.md : the Section-7 "Recording of Test Results" table.
  - summary.json                       : full per-test-case records.
  - <TC-ID>/result.json + evidence     : per-case evidence.

Usage:  python run_harness.py
"""

from __future__ import annotations

import csv
import datetime
import json

from harness import common as H
from harness import config as C
from harness import data_tests, model_tests, ood, robustness


def key_metrics(row: dict) -> str:
    """One-line summary of a case's measured values for the recording table."""
    m, tc = row["measured"], row["tc_id"]
    try:
        if tc == "TC-DATA-01":
            return f"{m['total_instances']} instances, {m['classes_present']} classes, imbalance {m['class_imbalance_ratio']}, dups {m['duplicate_images']}"
        if tc == "TC-DATA-02":
            return f"{m['classes_covered_min5']}/{m['target_classes']} classes >=5 instances; {m['objects_below_min_px_pct']}% objects <{C.MIN_OBJECT_PX}px"
        if tc == "TC-DATA-03":
            return f"{m['exact_duplicates_within_val']} exact dup, {m['near_duplicate_pairs_within_val']} near-dup pairs"
        if tc == "TC-DATA-04":
            return f"label error rate {m['label_error_rate_pct']}% over {m['objects']} objects"
        if tc == "TC-DATA-05":
            return f"induced KS p={m['induced_ks_p']} (detected {m['induced_drift_detected']}); control p={m['control_ks_p']}"
        if tc == "TC-ROB-01":
            return f"retention sev1={m['severity_1_mean_retention']}, sev3={m['severity_3_mean_retention']}, sev5={m['severity_5_mean_retention']}"
        if tc == "TC-ROB-02":
            return f"mean retention {m['mean_retention']} over {m['transforms']} transforms"
        if tc == "TC-PERF-01":
            o = m["overall"]
            return f"mAP50={o['mAP50']}, mAP50-95={o['mAP50_95']}, P={o['mean_precision']}, R={o['mean_recall']}"
        if tc == "TC-TIM-01":
            return f"inference {m['inference_ms_mean']}ms (p95 {m['inference_ms_p95']}); e2e {m['end_to_end_ms_mean']}ms"
        if tc == "TC-FUN":
            return f"{m['detection_rate_pct']}% images with detections; schema valid {m['schema_valid_images']}/{m['images']}"
        if tc == "TC-OOD-01":
            return f"AUROC {m['AUROC']}, FPR@95TPR {m['FPR_at_95TPR']}, OOD conf-rate {m['ood_confident_detection_rate']}"
    except KeyError:
        pass
    return json.dumps(m)[:120]


def write_csv(rows: list[dict], path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "Test Case",
                "Name",
                "Result",
                "Key measured value(s)",
                "Run date",
                "Evidence ref.",
                "Traceability (SRD)",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r["tc_id"],
                    r["name"],
                    r["status"],
                    key_metrics(r),
                    r["run_date"],
                    f"results/{r['tc_id']}/",
                    ", ".join(r["trace"]),
                ]
            )


def write_markdown(rows: list[dict], path, header: str) -> None:
    md = [
        "# SVCP Verification Results (auto-generated)",
        "",
        header,
        "",
        "Acceptance thresholds marked (TBC) are placeholders to confirm with the certification authority; "
        "values are measured, not assumed. Traceability IDs map each case to the SRD (see harness/traceability.py).",
        "",
        "## Section 7 - Recording of Test Results",
        "",
        "| Test Case | Result | Key measured value(s) | Traceability (SRD) | Evidence ref. |",
        "|-----------|--------|------------------------|--------------------|---------------|",
    ]
    for r in rows:
        md.append(
            f"| {r['tc_id']} | {r['status']} | {key_metrics(r)} | {', '.join(r['trace'])} | `results/{r['tc_id']}/` |"
        )
    md += ["", "## Per-case detail", ""]
    for r in rows:
        md += [
            f"### {r['tc_id']} - {r['name']}",
            f"- Status: **{r['status']}**",
            f"- Traceability (SRD): {', '.join(r['trace'])}",
            f"- Criteria: {r['criteria']}",
            f"- Measured: `{json.dumps(r['measured'])[:500]}`",
            f"- Observations: {r['observations']}",
            "",
        ]
    path.write_text("\n".join(md))


def main() -> None:
    device = H.device_name()
    print("=" * 70)
    print(
        f"SVCP verification harness  |  model: {C.MODEL_PATH.name}  |  device: {device}"
    )
    print(f"config: imgsz {C.IMGSZ}  conf {C.CONF}  iou {C.IOU}")
    print("=" * 70)

    data_tests.run_all()
    robustness.run_all()
    model_tests.run_all()
    ood.run_all()

    rows = H.results()
    header = (
        f"Generated: {datetime.datetime.now():%Y-%m-%d %H:%M}  |  Model: `{C.MODEL_PATH.name}` "
        f"({C.NUM_CLASSES} classes)  |  Device: {device}  |  Config: imgsz {C.IMGSZ}, conf {C.CONF}, IoU {C.IOU}"
    )
    write_csv(rows, C.RESULTS / "svcp_results.csv")
    write_markdown(rows, C.RESULTS / "svcp_results.md", header)
    H.save_json(rows, C.RESULTS / "summary.json")

    print("\n" + "=" * 70)
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print(
        "SUMMARY:",
        "  ".join(f"{k}={v}" for k, v in sorted(counts.items())),
        f"  (total {len(rows)})",
    )
    print("Results:", C.RESULTS / "svcp_results.csv")
    print("=" * 70)


if __name__ == "__main__":
    main()
