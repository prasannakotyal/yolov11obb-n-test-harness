"""TC-DATA-02 - ODD coverage and data desiderata.

Atomic checks of per-class coverage (one test per class), small-object fraction
and image-size range over the real dataset.
"""

import pytest

from harness import config as C

pytestmark = pytest.mark.tc("TC-DATA-02")

CLASS_NAMES = list(C.CLASS_NAMES.values())


def test_objects_total_positive(data02):
    assert data02["objects_total"] > 0


def test_no_under_represented_classes(data02):
    assert data02["under_represented_classes"] == []


@pytest.mark.parametrize("class_name", CLASS_NAMES)
def test_class_has_minimum_instances(data02, class_name):
    assert data02["class_coverage"][class_name] >= 5


def test_small_object_fraction_in_range(data02):
    assert 0.0 <= data02["objects_below_min_px_pct"] <= 100.0


def test_image_width_range_is_ascending(data02):
    lo, hi = data02["image_width_range_px"]
    assert 0 < lo <= hi


def test_standard_input_size_recorded(data02):
    assert data02["std_pipeline_input_px"] == C.STD_IMG
