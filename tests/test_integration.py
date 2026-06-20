"""Integration tests: dataset-YAML generation (fast) and one model case (slow)."""

import pytest

from harness import common as H
from harness import config as C


def test_ensure_dataset_yaml_is_portable(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "DATASET_YAML", tmp_path / "dataset.yaml")
    path = H.ensure_dataset_yaml()
    text = path.read_text()
    assert path.exists()
    assert f"path: {C.DATA}" in text  # absolute path computed for this machine
    assert "val: images/val" in text
    assert "0: plane" in text  # class names sourced from config


@pytest.mark.slow
def test_functional_case_end_to_end(tmp_path, monkeypatch):
    if not C.MODEL_PATH.exists():
        pytest.skip("model under test not present")
    if not (C.VAL_IMAGES.exists() and any(C.VAL_IMAGES.glob("*"))):
        pytest.skip("validation images not present")
    import torch

    if isinstance(C.DEVICE, int) and not torch.cuda.is_available():
        pytest.skip("configured CUDA device not available")

    monkeypatch.setattr(C, "RESULTS", tmp_path)
    monkeypatch.setattr(C, "FUN_IMAGES", 3)
    from harness import model_tests

    measured = model_tests.tc_fun()
    assert measured["images"] == 3
    assert "detection_rate_pct" in measured

    assert (tmp_path / "TC-FUN" / "result.json").exists()
    row = H.results()[-1]
    assert row["tc_id"] == "TC-FUN"
    assert row["trace"], "case must map to at least one SRD requirement"
    assert row["status"] in {"PASS", "FAIL"}
