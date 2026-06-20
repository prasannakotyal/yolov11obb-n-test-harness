"""TC-PERF-01 - Per-class detection accuracy (mAP / precision / recall).

Atomic checks that model.val() produces valid metrics overall and for every class
(one test per class). The per-class acceptance against the SRD table is TBC; the
measured values are recorded as evidence and validated for sanity here.
"""

import pytest

from harness import common as H
from harness import config as C

pytestmark = pytest.mark.tc("TC-PERF-01")

CLASS_NAMES = list(C.CLASS_NAMES.values())


def test_all_classes_evaluated(perf01):
    assert perf01["classes_evaluated"] == C.NUM_CLASSES


def test_overall_map50_is_valid(perf01):
    assert 0.0 < perf01["overall"]["mAP50"] <= 1.0


def test_map50_95_not_above_map50(perf01):
    overall = perf01["overall"]
    assert overall["mAP50_95"] <= overall["mAP50"]


def test_mean_precision_in_range(perf01):
    assert 0.0 <= perf01["overall"]["mean_precision"] <= 1.0


def test_mean_recall_in_range(perf01):
    assert 0.0 <= perf01["overall"]["mean_recall"] <= 1.0


@pytest.mark.parametrize("class_name", CLASS_NAMES)
def test_per_class_metrics_in_range(perf01, class_name):
    metrics = perf01["per_class"][class_name]
    assert isinstance(metrics, dict), f"{class_name} was not evaluated"
    for value in metrics.values():
        assert 0.0 <= value <= 1.0


def test_dataset_yaml_is_portable(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "DATASET_YAML", tmp_path / "dataset.yaml")
    text = H.ensure_dataset_yaml().read_text()
    assert f"path: {C.DATA}" in text  # absolute path for this machine, not a baked-in one
    assert "val: images/val" in text
    assert "0: plane" in text
