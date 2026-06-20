"""Known-answer tests for the pure verification metrics."""

import numpy as np
import pytest

from harness import metrics as M


# ---------------------------------------------------------------- parse_obb_label
def test_parse_obb_label_valid():
    text = "0 0 0 1 0 1 1 0 1\n3 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2"
    objects, malformed = M.parse_obb_label(text)
    assert malformed == 0
    assert len(objects) == 2
    cls, pts = objects[0]
    assert cls == 0
    assert pts.shape == (4, 2)


def test_parse_obb_label_counts_malformed():
    text = "0 0 0 1 0 1 1 0 1\n0 1 2 3\nbad line here now\n2 a b c d e f g h"
    objects, malformed = M.parse_obb_label(text)
    assert len(objects) == 1  # only the first line is well-formed
    assert malformed == 3  # wrong token count, wrong token count, non-numeric


def test_parse_obb_label_empty():
    assert M.parse_obb_label("   \n  ") == ([], 0)


# ---------------------------------------------------------------- polygon_area
def test_polygon_area_unit_square():
    assert M.polygon_area([[0, 0], [1, 0], [1, 1], [0, 1]]) == pytest.approx(1.0)


def test_polygon_area_degenerate_is_zero():
    assert M.polygon_area([[0, 0], [0, 0], [0, 0], [0, 0]]) == pytest.approx(0.0)


# ---------------------------------------------------------------- class_imbalance
def test_class_imbalance_ratio():
    assert M.class_imbalance([10, 5, 1]) == 10.0


def test_class_imbalance_ignores_absent_classes():
    assert M.class_imbalance([0, 4, 0]) == 1.0


def test_class_imbalance_none_when_empty():
    assert M.class_imbalance([]) is None
    assert M.class_imbalance([0, 0]) is None


# ---------------------------------------------------------------- auroc
def test_auroc_perfect_separation():
    assert M.auroc([0.9, 0.8, 0.95], [0.1, 0.2, 0.05]) == pytest.approx(1.0)


def test_auroc_reversed_separation():
    assert M.auroc([0.1, 0.2], [0.9, 0.8]) == pytest.approx(0.0)


def test_auroc_identical_is_half():
    assert M.auroc([0.5, 0.5], [0.5, 0.5]) == pytest.approx(0.5)


# ---------------------------------------------------------------- fpr_at_tpr
def test_fpr_at_tpr_no_false_positives():
    id_scores = np.linspace(0.5, 1.0, 100)
    assert M.fpr_at_tpr(id_scores, np.zeros(50), tpr=0.95) == pytest.approx(0.0)


def test_fpr_at_tpr_all_false_positives():
    id_scores = np.linspace(0.5, 1.0, 100)
    assert M.fpr_at_tpr(id_scores, np.ones(50), tpr=0.95) == pytest.approx(1.0)


# ---------------------------------------------------------------- ks_drift
def test_ks_drift_detects_shift():
    rng = np.random.default_rng(0)
    out = M.ks_drift(rng.normal(0, 1, 500), rng.normal(5, 1, 500), alpha=0.05)
    assert out["drift"] is True
    assert out["p"] < 0.05


def test_ks_drift_no_false_alarm_on_same_distribution():
    rng = np.random.default_rng(1)
    sample = rng.normal(0, 1, 500)
    out = M.ks_drift(sample, sample, alpha=0.05)
    assert out["drift"] is False


# ---------------------------------------------------------------- js_distance
def test_js_distance_identical_is_zero():
    rng = np.random.default_rng(0)
    sample = rng.uniform(0, 255, 1000)
    assert M.js_distance(sample, sample) == pytest.approx(0.0, abs=1e-6)


def test_js_distance_disjoint_is_large():
    assert M.js_distance(np.full(500, 10.0), np.full(500, 240.0)) > 0.5


# ---------------------------------------------------------------- retention
def test_retention_basic():
    assert M.retention(8, 10) == 0.8
    assert M.retention(0, 10) == 0.0


def test_retention_floors_zero_baseline():
    assert M.retention(5, 0) == 5.0  # baseline floored to 1 to avoid division by zero


# ---------------------------------------------------------------- valid_obb_schema
def _poly(n):
    return np.zeros((n, 4, 2))


def test_valid_obb_schema_accepts_well_formed():
    assert M.valid_obb_schema([0.5, 0.9], [0, 3], _poly(2), num_classes=15) is True


def test_valid_obb_schema_rejects_confidence_out_of_range():
    assert M.valid_obb_schema([1.5], [0], _poly(1), num_classes=15) is False


def test_valid_obb_schema_rejects_class_out_of_range():
    assert M.valid_obb_schema([0.5], [15], _poly(1), num_classes=15) is False


def test_valid_obb_schema_rejects_wrong_polygon_shape():
    assert M.valid_obb_schema([0.5], [0], np.zeros((1, 3, 2)), num_classes=15) is False
