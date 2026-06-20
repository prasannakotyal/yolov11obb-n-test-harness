"""Pure verification metrics: no model, no GPU, no file I/O.

Every function here is deterministic given its inputs, which is what makes the
methodology unit-testable (see tests/test_metrics.py). The case modules call
these on real model outputs; the tests call them on known-answer inputs.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.spatial.distance import jensenshannon


def parse_obb_label(text: str):
    """Parse DOTA oriented-bounding-box label text.

    Each valid line is ``<cls> x1 y1 x2 y2 x3 y3 x4 y4`` (9 whitespace-separated
    tokens). Returns ``(objects, n_malformed)`` where objects is a list of
    ``(cls: int, pts: ndarray[4, 2])`` and n_malformed counts unparseable lines.
    """
    objects, malformed = [], 0
    for line in text.strip().splitlines():
        tok = line.split()
        if len(tok) != 9:
            malformed += 1
            continue
        try:
            cls = int(tok[0])
            coords = [float(x) for x in tok[1:]]
        except ValueError:
            malformed += 1
            continue
        objects.append((cls, np.asarray(coords, dtype=float).reshape(4, 2)))
    return objects, malformed


def polygon_area(pts) -> float:
    """Shoelace area of a polygon given as an (N, 2) point array."""
    pts = np.asarray(pts, dtype=float)
    x, y = pts[:, 0], pts[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def class_imbalance(counts) -> float | None:
    """Ratio of the most- to least-frequent present class. None if no instances."""
    present = [c for c in counts if c > 0]
    if not present:
        return None
    return round(max(present) / min(present), 1)


def auroc(scores_id, scores_ood) -> float:
    """Area under the ROC for separating in-distribution from OOD by score.

    In-distribution is the positive class. Computed rank-based (Mann-Whitney U),
    so no threshold sweep is needed. Higher score is expected to mean "more ID".
    """
    s_id = np.asarray(scores_id, dtype=float)
    s_ood = np.asarray(scores_ood, dtype=float)
    n1, n0 = len(s_id), len(s_ood)
    assert n1 and n0, "need at least one ID and one OOD score"
    ranks = stats.rankdata(np.concatenate([s_id, s_ood]))
    return float((ranks[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def fpr_at_tpr(scores_id, scores_ood, tpr: float = 0.95) -> float:
    """OOD false-positive rate at a threshold that keeps `tpr` of ID scores."""
    s_id = np.asarray(scores_id, dtype=float)
    s_ood = np.asarray(scores_ood, dtype=float)
    threshold = np.percentile(s_id, 100 * (1 - tpr))
    return float((s_ood >= threshold).mean())


def ks_drift(reference, current, alpha: float = 0.05) -> dict:
    """Two-sample Kolmogorov-Smirnov drift test between two 1-D samples."""
    ks = stats.ks_2samp(
        np.asarray(reference, dtype=float), np.asarray(current, dtype=float)
    )
    return {
        "stat": float(ks.statistic),
        "p": float(ks.pvalue),
        "drift": bool(ks.pvalue < alpha),
    }


def js_distance(a, b, bins: int = 32, lo: float = 0.0, hi: float = 255.0) -> float:
    """Jensen-Shannon distance between the histograms of two samples (0 = identical)."""
    edges = np.linspace(lo, hi, bins + 1)
    pa = np.histogram(a, edges)[0] + 1e-9
    pb = np.histogram(b, edges)[0] + 1e-9
    return float(jensenshannon(pa, pb))


def retention(perturbed_count: int, baseline_count: int) -> float:
    """Detection retention = detections under perturbation / clean-baseline detections."""
    return round(perturbed_count / max(baseline_count, 1), 3)


def valid_obb_schema(conf, cls, poly, num_classes: int) -> bool:
    """True if a detection batch is well-formed.

    Confidences in [0, 1], class ids in [0, num_classes), polygons shaped (N, 4, 2).
    """
    conf = np.asarray(conf, dtype=float)
    cls = np.asarray(cls, dtype=int)
    poly = np.asarray(poly)
    return bool(
        (conf >= 0).all()
        and (conf <= 1).all()
        and (cls >= 0).all()
        and (cls < num_classes).all()
        and poly.shape[1:] == (4, 2)
    )
