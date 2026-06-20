"""Black-box model-validation cases: TC-PERF-01 accuracy, TC-TIM-01 latency, TC-FUN functional."""

from __future__ import annotations

import numpy as np

from . import common as H
from . import config as C
from . import metrics as M


def tc_perf_01(model, batch: int = 4) -> dict:
    """Per-class detection accuracy via Ultralytics model.val() (ground-truth labels)."""
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

    return {
        "images": len(H.val_images()),
        "iou_for_mAP50": 0.5,
        "classes_evaluated": len(evaluated),
        "overall": {
            "mAP50": round(float(box.map50), 4),
            "mAP50_95": round(float(box.map), 4),
            "mean_precision": round(float(box.mp), 4),
            "mean_recall": round(float(box.mr), 4),
        },
        "per_class": per_class,
    }


def tc_tim_01(model) -> dict:
    """Per-frame inference and end-to-end latency (Ultralytics pre+inference+post timing)."""
    paths = H.val_images(C.TIM_IMAGES)
    for p in paths[:10]:
        H.predict(model, str(p))  # warm-up

    inference_ms, end_to_end_ms = [], []
    for p in paths:
        speed = H.predict(model, str(p)).speed  # ms: preprocess, inference, postprocess
        inference_ms.append(float(speed["inference"]))
        end_to_end_ms.append(float(speed["preprocess"] + speed["inference"] + speed["postprocess"]))
    inference_ms, end_to_end_ms = np.array(inference_ms), np.array(end_to_end_ms)

    return {
        "images_timed": len(paths),
        "device": H.device_name(),
        "inference_ms_mean": round(float(inference_ms.mean()), 2),
        "inference_ms_p95": round(float(np.percentile(inference_ms, 95)), 2),
        "end_to_end_ms_mean": round(float(end_to_end_ms.mean()), 2),
        "end_to_end_ms_p95": round(float(np.percentile(end_to_end_ms, 95)), 2),
        "infer_budget_ms": C.INFER_BUDGET_MS,
        "e2e_budget_ms": C.E2E_BUDGET_MS,
    }


def tc_fun(model) -> dict:
    """Functional detect/classify and output-schema validity."""
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
                obb.conf.cpu().numpy(), obb.cls.cpu().numpy(), obb.xyxyxyxy.cpu().numpy(), C.NUM_CLASSES
            ):
                schema_valid += 1
            else:
                violations += 1

    return {
        "images": len(paths),
        "images_with_detections": with_detections,
        "schema_valid_images": schema_valid,
        "schema_violations": violations,
        "detection_rate_pct": round(100 * with_detections / len(paths), 1),
    }
