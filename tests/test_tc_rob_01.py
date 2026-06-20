"""TC-ROB-01 - Image-corruption robustness (invariance / graceful degradation).

Atomic checks on the real retention sweep plus, for every corruption, a known-answer
check that it produces a valid same-shape image and is deterministic under a seed.
"""

import numpy as np
import pytest

from harness import config as C
from harness import corruptions as X
from harness import metrics as M

pytestmark = pytest.mark.tc("TC-ROB-01")

CORRUPTIONS = [(f"{g}/{n}", fn) for g, fns in X.CORRUPTIONS.items() for n, fn in fns.items()]
CORRUPTION_NAMES = [name for name, _ in CORRUPTIONS]


# --- retention sweep on the real model + data ---
def test_baseline_has_detections(rob01):
    assert rob01["baseline_total_detections"] > 0


def test_suite_covers_15_corruptions(rob01):
    assert rob01["corruption_types"] == 15
    assert len(rob01["per_corruption_retention"]) == 15


def test_severity1_retention_meets_threshold(rob01):
    assert rob01["severity_mean_retention"][1] >= C.ROB_MIN_RETENTION


def test_retention_degrades_with_severity(rob01):
    mean = rob01["severity_mean_retention"]
    assert mean[1] >= mean[5]


@pytest.mark.parametrize("name", CORRUPTION_NAMES)
def test_corruption_retention_recorded_for_all_severities(rob01, name):
    row = rob01["per_corruption_retention"][name]
    assert sorted(row) == list(X.SEVERITIES)
    assert all(value >= 0.0 for value in row.values())


# --- methodology: each corruption returns a valid same-shape image ---
@pytest.mark.parametrize("name,fn", CORRUPTIONS, ids=CORRUPTION_NAMES)
@pytest.mark.parametrize("severity", [1, 5])
def test_corruption_preserves_shape_dtype_range(name, fn, severity, rgb_image):
    np.random.seed(0)
    out = X.apply_corruption(rgb_image, fn, severity)
    assert out.shape == rgb_image.shape
    assert out.dtype == np.uint8
    assert out.min() >= 0 and out.max() <= 255


def test_corruption_is_deterministic_under_seed(rgb_image):
    np.random.seed(123)
    first = X.apply_corruption(rgb_image, X.gaussian_noise, 3)
    np.random.seed(123)
    second = X.apply_corruption(rgb_image, X.gaussian_noise, 3)
    assert np.array_equal(first, second)


def test_retention_metric_matches_definition():
    assert M.retention(8, 10) == 0.8
    assert M.retention(0, 10) == 0.0
