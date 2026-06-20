"""Black-box model-validation cases: TC-PERF-01 accuracy, TC-TIM-01 latency, TC-FUN functional."""

from __future__ import annotations

import numpy as np

from . import common as H
from . import config as C
from . import metrics as M


def tc_perf_01(batch: int = 4):
    """Per-class detection accuracy via Ultralytics model.val() (ground-truth labels)."""
    model = H.load_model()
    result = model.val(
        data=str(H.ensure_dataset_yaml()),
        imgsz=C.IMGSZ,
        conf=0.001,
        iou=C.IOU,
        batch=batch,
        device=C.DEVICE,
        plots=False,
        verbose=False,
        save_json=False,
    )
    box = result.box
    evaluated = list(box.ap_class_index)
    targets = C.CLASSES_OF_INTEREST or sorted(evaluated)

    per_class = {}
    for cid in targets:
        name = C.CLASS_NAMES.get(cid, str(cid))
        if cid in evaluated:
            j = evaluated.index(cid)
            per_class[name] = {
                "mAP50": round(float(box.ap50[j]), 4),
                "precision": round(float(box.p[j]), 4),
                "recall": round(float(box.r[j]), 4),
            }
        else:
            per_class[name] = "no instances in dataset"

    measured = {
        "images": len(H.val_images()),
        "iou_for_mAP50": 0.5,
        "overall": {
            "mAP50": round(float(box.map50), 4),
            "mAP50_95": round(float(box.map), 4),
            "mean_precision": round(float(box.mp), 4),
            "mean_recall": round(float(box.mr), 4),
        },
        "per_class": per_class,
    }
    H.save_json(measured, C.RESULTS / "TC-PERF-01" / "accuracy.json")
    H.record(
        "TC-PERF-01",
        "Per-Class Detection Accuracy (mAP / precision / recall)",
        "MEASURED",
        "Acceptance: each class meets its SRD per-class thresholds at IoU 0.5. The harness reports measured values.",
        measured,
        "Mechanism (model.val mAP/precision/recall) demonstrated; pass/fail is set against the SRD per-class table.",
        ["TC-PERF-01/accuracy.json"],
    )
    return measured


def tc_tim_01():
    """Per-frame inference and end-to-end latency (Ultralytics pre+inference+post timing)."""
    model = H.load_model()
    paths = H.val_images(C.TIM_IMAGES)
    for p in paths[:10]:
        H.predict(model, str(p))  # warm-up

    inference_ms, end_to_end_ms = [], []
    for p in paths:
        speed = H.predict(model, str(p)).speed  # ms: preprocess, inference, postprocess
        inference_ms.append(float(speed["inference"]))
        end_to_end_ms.append(
            float(speed["preprocess"] + speed["inference"] + speed["postprocess"])
        )
    inference_ms, end_to_end_ms = np.array(inference_ms), np.array(end_to_end_ms)

    measured = {
        "images_timed": len(paths),
        "device": H.device_name(),
        "inference_ms_mean": round(float(inference_ms.mean()), 2),
        "inference_ms_p95": round(float(np.percentile(inference_ms, 95)), 2),
        "end_to_end_ms_mean": round(float(end_to_end_ms.mean()), 2),
        "end_to_end_ms_p95": round(float(np.percentile(end_to_end_ms, 95)), 2),
        "infer_budget_ms": C.INFER_BUDGET_MS,
        "e2e_budget_ms": C.E2E_BUDGET_MS,
    }
    ok = (
        measured["inference_ms_mean"] <= C.INFER_BUDGET_MS
        and measured["end_to_end_ms_mean"] <= C.E2E_BUDGET_MS
    )
    H.record(
        "TC-TIM-01",
        "Per-Frame Inference and End-to-End Latency",
        "PASS" if ok else "MEASURED",
        f"PASS if mean inference <= {C.INFER_BUDGET_MS} ms and mean end-to-end <= {C.E2E_BUDGET_MS} ms (reference device).",
        measured,
        "Measured on the host device reported above; final acceptance is on the deployment target.",
    )
    return measured


def tc_fun():
    """Functional detect/classify and output-schema validity."""
    model = H.load_model()
    paths = H.val_images(C.FUN_IMAGES)
    with_detections, schema_valid, violations = 0, 0, 0
    for p in paths:
        obb = H.predict(model, str(p)).obb
        if obb is None:
            violations += 1
            continue
        if obb.conf is not None and len(obb.conf) > 0:
            with_detections += 1
            if M.valid_obb_schema(
                obb.conf.cpu().numpy(),
                obb.cls.cpu().numpy(),
                obb.xyxyxyxy.cpu().numpy(),
                C.NUM_CLASSES,
            ):
                schema_valid += 1
            else:
                violations += 1

    measured = {
        "images": len(paths),
        "images_with_detections": with_detections,
        "schema_valid_images": schema_valid,
        "schema_violations": violations,
        "detection_rate_pct": round(100 * with_detections / len(paths), 1),
    }
    H.record(
        "TC-FUN",
        "Detect/Classify and Output-Schema Validity",
        "PASS" if violations == 0 and with_detections > 0 else "FAIL",
        f"PASS if outputs are well-formed (OBB present, class in [0,{C.NUM_CLASSES - 1}], confidence in [0,1], "
        "4-point polygons) and detections are produced on images containing objects.",
        measured,
        "Confirms the published detection schema (OBB / label / confidence) is well-formed.",
    )
    return measured


def run_all():
    print("[model black-box validation]")
    tc_fun()
    tc_tim_01()
    tc_perf_01()
