"""TC-ROB-02 - Geometric-transform robustness (metamorphic).

Atomic checks that realistic transforms preserve detections, plus known-answer
checks of the transforms themselves.
"""

import numpy as np
import pytest

from harness import config as C
from harness import corruptions as X

pytestmark = pytest.mark.tc("TC-ROB-02")

TRANSFORMS = list(X.GEOMETRIC.items())
TRANSFORM_NAMES = list(X.GEOMETRIC)


# --- retention on the real model + data ---
def test_baseline_has_detections(rob02):
    assert rob02["baseline_total_detections"] > 0


def test_all_transforms_measured(rob02):
    assert len(rob02["per_transform_retention"]) == len(X.GEOMETRIC)


def test_mean_retention_meets_threshold(rob02):
    assert rob02["mean_retention"] >= C.ROB_MIN_RETENTION


@pytest.mark.parametrize("name", TRANSFORM_NAMES)
def test_transform_retention_recorded(rob02, name):
    assert rob02["per_transform_retention"][name] >= 0.0


# --- methodology: each transform returns a valid image ---
@pytest.mark.parametrize("name,fn", TRANSFORMS, ids=TRANSFORM_NAMES)
def test_transform_returns_contiguous_uint8_image(name, fn, rgb_image):
    out = X.apply_geometric(rgb_image, fn)
    assert out.dtype == np.uint8
    assert out.ndim == 3
    assert out.flags["C_CONTIGUOUS"]


def test_hflip_is_an_involution(rgb_image):
    once = X.apply_geometric(rgb_image, X.GEOMETRIC["hflip"])
    twice = X.apply_geometric(once, X.GEOMETRIC["hflip"])
    assert np.array_equal(twice, rgb_image)
