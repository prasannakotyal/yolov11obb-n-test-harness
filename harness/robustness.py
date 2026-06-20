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
    return cv2.resize(H.read_rgb(path), (C.STD_IMG, C.STD_IMG), interpolation=cv2.INTER_AREA)


def _baseline_detections(model, images) -> int:
    return max(sum(H.dets_of(H.predict(model, im))["n"] for im in images), 1)


def tc_rob_01(model) -> dict:
    """Image-corruption robustness (invariance / graceful degradation)."""
    np.random.seed(C.SEED)
    cleans = [_prep(p) for p in H.val_images(C.ROB_IMAGES)]
    baseline = _baseline_detections(model, cleans)

    per_corruption, per_severity = {}, {s: [] for s in X.SEVERITIES}
    for group, funcs in X.CORRUPTIONS.items():
        for name, fn in funcs.items():
            row = {}
            for s in X.SEVERITIES:
                total = sum(H.dets_of(H.predict(model, X.apply_corruption(im, fn, s)))["n"] for im in cleans)
                row[s] = M.retention(total, baseline)
                per_severity[s].append(row[s])
            per_corruption[f"{group}/{name}"] = row

    return {
        "images": C.ROB_IMAGES,
        "corruption_types": sum(len(v) for v in X.CORRUPTIONS.values()),
        "severities": len(X.SEVERITIES),
        "baseline_total_detections": baseline,
        "severity_mean_retention": {s: round(float(np.mean(v)), 3) for s, v in per_severity.items()},
        "per_corruption_retention": per_corruption,
    }


def tc_rob_02(model) -> dict:
    """Geometric-transform robustness (metamorphic): realistic transforms preserve detections."""
    np.random.seed(C.SEED)
    cleans = [_prep(p) for p in H.val_images(C.ROB_IMAGES)]
    baseline = _baseline_detections(model, cleans)

    per_transform = {}
    for name, fn in X.GEOMETRIC.items():
        total = sum(H.dets_of(H.predict(model, X.apply_geometric(im, fn)))["n"] for im in cleans)
        per_transform[name] = M.retention(total, baseline)
    return {
        "images": C.ROB_IMAGES,
        "transforms": len(X.GEOMETRIC),
        "baseline_total_detections": baseline,
        "mean_retention": round(float(np.mean(list(per_transform.values()))), 3),
        "per_transform_retention": per_transform,
    }
