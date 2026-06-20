"""TC-FUN - Detect/classify and output-schema validity.

Atomic checks that the model produces well-formed detections on real frames, plus
known-answer checks of the schema validator.
"""

import numpy as np
import pytest

from harness import metrics as M

pytestmark = pytest.mark.tc("TC-FUN")


# --- schema validity on real model outputs ---
def test_detections_are_produced(fun01):
    assert fun01["images_with_detections"] > 0


def test_no_schema_violations(fun01):
    assert fun01["schema_violations"] == 0


def test_every_detection_image_is_schema_valid(fun01):
    assert fun01["schema_valid_images"] == fun01["images_with_detections"]


def test_detection_rate_positive(fun01):
    assert fun01["detection_rate_pct"] > 0


# --- methodology: the validator accepts/rejects the right shapes ---
def test_validator_accepts_well_formed_batch():
    assert M.valid_obb_schema([0.5, 0.9], [0, 3], np.zeros((2, 4, 2)), num_classes=15) is True


def test_validator_rejects_confidence_out_of_range():
    assert M.valid_obb_schema([1.5], [0], np.zeros((1, 4, 2)), num_classes=15) is False


def test_validator_rejects_class_out_of_range():
    assert M.valid_obb_schema([0.5], [15], np.zeros((1, 4, 2)), num_classes=15) is False


def test_validator_rejects_wrong_polygon_shape():
    assert M.valid_obb_schema([0.5], [0], np.zeros((1, 3, 2)), num_classes=15) is False
