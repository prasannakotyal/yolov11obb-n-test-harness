"""Data-management verification cases TC-DATA-01..05.

Verifies the dataset that bounds the model's airborne behaviour: integrity,
ODD coverage, train/val/test separation, label quality and drift monitoring.
"""

from __future__ import annotations

from collections import Counter

import cv2
import numpy as np
from PIL import Image

from . import common as H
from . import config as C
from . import metrics as M


def tc_data_01():
    """Dataset integrity and class distribution."""
    imgs = H.val_images()
    labels = sorted(C.VAL_LABELS.glob("*.txt"))
    unpaired = {p.stem for p in imgs} ^ {p.stem for p in labels}

    unreadable, hashes, duplicates = 0, {}, []
    for p in imgs:
        try:
            if Image.open(p).size[0] <= 0:
                unreadable += 1
        except Exception:
            unreadable += 1
        digest = H.md5(p)
        duplicates.append(p.name) if digest in hashes else hashes.setdefault(
            digest, p.name
        )

    counts, malformed, out_of_range = Counter(), 0, 0
    for lp in labels:
        objs, bad = M.parse_obb_label(lp.read_text())
        malformed += bad
        for cls, _ in objs:
            counts[cls] += 1
            if cls < 0 or cls >= C.NUM_CLASSES:
                out_of_range += 1

    imbalance = M.class_imbalance(counts.values())
    ok = (
        not unpaired
        and not malformed
        and not out_of_range
        and not unreadable
        and not duplicates
    )
    measured = {
        "images": len(imgs),
        "labels": len(labels),
        "total_instances": int(sum(counts.values())),
        "classes_present": sum(1 for c in counts.values() if c > 0),
        "unpaired_files": len(unpaired),
        "unreadable_images": unreadable,
        "malformed_label_lines": malformed,
        "out_of_range_class": out_of_range,
        "duplicate_images": len(duplicates),
        "class_imbalance_ratio": imbalance,
        "class_distribution": {
            C.CLASS_NAMES[k]: counts.get(k, 0) for k in C.CLASS_NAMES
        },
    }
    H.record(
        "TC-DATA-01",
        "Dataset Integrity and Class Distribution",
        "PASS" if ok else "FAIL",
        f"PASS if no unpaired/unreadable/duplicate files, no malformed labels and all class ids in [0,{C.NUM_CLASSES - 1}].",
        measured,
        f"Class imbalance ratio {imbalance} (max/min instances). Distribution reported for review.",
    )
    return measured


def tc_data_02():
    """ODD coverage and data desiderata (class coverage, small-object fraction, size range)."""
    labels = sorted(C.VAL_LABELS.glob("*.txt"))
    counts, small_obj, total_obj, sizes = Counter(), 0, 0, []
    for lp in labels:
        ip = C.VAL_IMAGES / (lp.stem + ".jpg")
        if not ip.exists():
            continue
        w, h = Image.open(ip).size
        sizes.append((w, h))
        objs, _ = M.parse_obb_label(lp.read_text())
        for cls, pts in objs:
            counts[cls] += 1
            total_obj += 1
            wpx = (pts[:, 0].max() - pts[:, 0].min()) * w
            hpx = (pts[:, 1].max() - pts[:, 1].min()) * h
            if wpx < C.MIN_OBJECT_PX or hpx < C.MIN_OBJECT_PX:
                small_obj += 1

    targets = C.CLASSES_OF_INTEREST or list(C.CLASS_NAMES)
    covered = [c for c in targets if counts.get(c, 0) >= 5]
    gaps = [C.CLASS_NAMES[c] for c in targets if counts.get(c, 0) < 5]
    widths = [w for w, _ in sizes]
    measured = {
        "target_classes": len(targets),
        "classes_covered_min5": len(covered),
        "class_coverage": {C.CLASS_NAMES[c]: counts.get(c, 0) for c in targets},
        "under_represented_classes": gaps,
        "objects_total": total_obj,
        "objects_below_min_px": small_obj,
        "objects_below_min_px_pct": round(100 * small_obj / max(total_obj, 1), 2),
        "image_width_range_px": [int(min(widths)), int(max(widths))],
        "std_pipeline_input_px": C.STD_IMG,
    }
    H.record(
        "TC-DATA-02",
        "ODD Coverage and Data Desiderata",
        "PASS" if not gaps else "PARTIAL",
        "PASS if every target class is represented (>=5 instances); under-represented classes and small-object fraction reported.",
        measured,
        "Environmental-condition stratification (day/night, weather) needs dataset metadata beyond image/label files.",
    )
    return measured


def tc_data_03():
    """Train/validation/test separation and leakage detection (intra-set + method)."""
    imgs = H.val_images()
    exact, exact_dups, phash, near = {}, [], {}, []
    for p in imgs:
        digest = H.md5(p)
        exact_dups.append(p.name) if digest in exact else exact.setdefault(
            digest, p.name
        )
        small = cv2.resize(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE), (16, 16))
        bits = (small > small.mean()).astype(np.uint8).tobytes()
        if bits in phash and phash[bits] != p.name:
            near.append((p.name, phash[bits]))
        phash[bits] = p.name
    measured = {
        "val_images": len(imgs),
        "exact_duplicates_within_val": len(exact_dups),
        "near_duplicate_pairs_within_val": len(near),
        "method": "MD5 (exact) + 16x16 average-hash (near) per image; cross-split disjointness applies the "
        "same hashes across train/val/test manifests.",
        "cross_split_check": "requires train+test manifests (not in this val-only subset)",
    }
    H.record(
        "TC-DATA-03",
        "Train/Validation/Test Separation (no leakage)",
        "PASS" if not exact_dups else "FAIL",
        "PASS if no exact duplicates within the set; full cross-split disjointness reuses the same hash check across manifests.",
        measured,
        f"{len(near)} near-duplicate pair(s) flagged for review by 16x16 average-hash.",
    )
    return measured


def tc_data_04():
    """Label-quality review (format and geometry validity)."""
    labels = sorted(C.VAL_LABELS.glob("*.txt"))
    total, malformed, degenerate, out_of_range = 0, 0, 0, 0
    for lp in labels:
        objs, bad = M.parse_obb_label(lp.read_text())
        malformed += bad
        for _, pts in objs:
            total += 1
            if (pts < -0.01).any() or (pts > 1.01).any():
                out_of_range += 1
            if M.polygon_area(pts) < 1e-7:
                degenerate += 1
    errors = malformed + degenerate + out_of_range
    rate = round(100 * errors / max(total + malformed, 1), 3)
    measured = {
        "labels_reviewed": len(labels),
        "objects": total,
        "malformed_lines": malformed,
        "degenerate_polygons": degenerate,
        "out_of_range_coords": out_of_range,
        "label_error_rate_pct": rate,
    }
    H.record(
        "TC-DATA-04",
        "Label-Quality Review",
        "PASS" if rate <= C.MAX_LABEL_ERROR_PCT else "FAIL",
        f"PASS if label error rate (malformed + degenerate + out-of-range) <= {C.MAX_LABEL_ERROR_PCT}% (TBC).",
        measured,
        "Full census of the label set (not a sample).",
    )
    return measured


def _brightness(paths) -> np.ndarray:
    return np.array(
        [
            float(
                cv2.resize(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE), (128, 128)).mean()
            )
            for p in paths
        ]
    )


def tc_data_05():
    """Data-drift monitoring: control (no drift) vs induced brightness drift, via KS + JS."""
    imgs = H.val_images()
    half = len(imgs) // 2
    ref, cur = _brightness(imgs[:half]), _brightness(imgs[half:])
    control = M.ks_drift(ref, cur, C.DRIFT_ALPHA)
    shifted = np.clip(cur + 60.0, 0, 255)  # induced brightness drift
    induced = M.ks_drift(ref, shifted, C.DRIFT_ALPHA)
    measured = {
        "feature": "per-image mean brightness (128x128 grayscale)",
        "control_ks_stat": round(control["stat"], 3),
        "control_ks_p": round(control["p"], 4),
        "control_js": round(M.js_distance(ref, cur), 3),
        "control_drift_detected": control["drift"],
        "induced_ks_stat": round(induced["stat"], 3),
        "induced_ks_p": float(f"{induced['p']:.2e}"),
        "induced_js": round(M.js_distance(ref, shifted), 3),
        "induced_drift_detected": induced["drift"],
    }
    H.record(
        "TC-DATA-05",
        "Data/Model Drift Monitoring",
        "PASS" if (induced["drift"] and not control["drift"]) else "FAIL",
        f"PASS if induced drift is detected (KS p<{C.DRIFT_ALPHA}) and the no-drift control is not flagged.",
        measured,
        "Demonstrates the KS/JS drift detector; error-rate detectors (DDM/ADWIN/EDDM) are a post-deployment provision.",
    )
    return measured


def run_all():
    print("[data-management verification]")
    tc_data_01()
    tc_data_02()
    tc_data_03()
    tc_data_04()
    tc_data_05()
