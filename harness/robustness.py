"""Black-box robustness cases: TC-ROB-01 image corruptions, TC-ROB-02 geometric.

Metric = detection retention (confident detections under perturbation vs clean):
invariance at low severity, graceful degradation as severity rises. White-box
gradient attacks (FGSM/PGD) are out of scope at Level 1 (input/output only).
"""

from __future__ import annotations

import cv2
import numpy as np

from . import common as H
from . import config as C
from . import corruptions as X
from . import metrics as M


def _prep(path) -> np.ndarray:
    return cv2.resize(
        H.read_rgb(path), (C.STD_IMG, C.STD_IMG), interpolation=cv2.INTER_AREA
    )


def _baseline_detections(model, images) -> int:
    return max(sum(H.dets_of(H.predict(model, im))["n"] for im in images), 1)


def tc_rob_01():
    """Image-corruption robustness (invariance / graceful degradation)."""
    np.random.seed(C.SEED)
    model = H.load_model()
    cleans = [_prep(p) for p in H.val_images(C.ROB_IMAGES)]
    baseline = _baseline_detections(model, cleans)

    table, per_severity = {}, {s: [] for s in X.SEVERITIES}
    for group, funcs in X.CORRUPTIONS.items():
        for name, fn in funcs.items():
            row = {}
            for s in X.SEVERITIES:
                total = sum(
                    H.dets_of(H.predict(model, X.apply_corruption(im, fn, s)))["n"]
                    for im in cleans
                )
                row[s] = M.retention(total, baseline)
                per_severity[s].append(row[s])
            table[f"{group}/{name}"] = row

    summary = {
        f"severity_{s}_mean_retention": round(float(np.mean(v)), 3)
        for s, v in per_severity.items()
    }
    worst = min(((k, r[1]) for k, r in table.items()), key=lambda kv: kv[1])
    H.save_json(
        {
            "baseline_total_detections": baseline,
            "per_corruption_retention": table,
            "severity_summary": summary,
        },
        C.RESULTS / "TC-ROB-01" / "retention.json",
    )
    sev1 = summary["severity_1_mean_retention"]
    H.record(
        "TC-ROB-01",
        "Image-Corruption Robustness (Invariance)",
        "PASS" if sev1 >= C.ROB_MIN_RETENTION else "MEASURED",
        f"PASS if mean detection retention at severity 1 >= {C.ROB_MIN_RETENTION} (TBC) with graceful degradation thereafter.",
        {
            "images": C.ROB_IMAGES,
            "corruption_types": sum(len(v) for v in X.CORRUPTIONS.values()),
            "severities": len(X.SEVERITIES),
            **summary,
            "worst_corruption_at_sev1": {worst[0]: worst[1]},
        },
        "Full 15x5 retention table in retention.json. Acceptance retention thresholds are TBC.",
        ["TC-ROB-01/retention.json"],
    )
    return summary


def tc_rob_02():
    """Geometric-transform robustness (metamorphic): realistic transforms preserve detections."""
    np.random.seed(C.SEED)
    model = H.load_model()
    cleans = [_prep(p) for p in H.val_images(C.ROB_IMAGES)]
    baseline = _baseline_detections(model, cleans)

    table = {}
    for name, fn in X.GEOMETRIC.items():
        total = sum(
            H.dets_of(H.predict(model, X.apply_geometric(im, fn)))["n"] for im in cleans
        )
        table[name] = M.retention(total, baseline)
    mean_ret = round(float(np.mean(list(table.values()))), 3)
    H.save_json(
        {"baseline_total": baseline, "per_transform_retention": table},
        C.RESULTS / "TC-ROB-02" / "geometric.json",
    )
    H.record(
        "TC-ROB-02",
        "Geometric-Transform Robustness (Metamorphic)",
        "PASS" if mean_ret >= C.ROB_MIN_RETENTION else "MEASURED",
        f"PASS if realistic transforms preserve detections (mean retention >= {C.ROB_MIN_RETENTION}, TBC).",
        {
            "images": C.ROB_IMAGES,
            "transforms": len(X.GEOMETRIC),
            "mean_retention": mean_ret,
            "per_transform": table,
        },
        "Stage-1 realistic transforms; zoom-in onto target-free regions is not required to flag.",
        ["TC-ROB-02/geometric.json"],
    )
    return table


def run_all():
    print("[robustness verification]")
    tc_rob_01()
    tc_rob_02()
