"""TC-DATA-04 - Label-quality review.

Atomic checks over every object (format + polygon geometry) plus known-answer
checks of the geometry/parsing helpers the review relies on.
"""

import numpy as np
import pytest

from harness import config as C
from harness import metrics as M

pytestmark = pytest.mark.tc("TC-DATA-04")


# --- review of the real label set ---
def test_objects_reviewed_positive(data04):
    assert data04["objects"] > 0


def test_no_malformed_lines(data04):
    assert data04["malformed_lines"] == 0


def test_no_degenerate_polygons(data04):
    assert data04["degenerate_polygons"] == 0


def test_no_out_of_range_coords(data04):
    assert data04["out_of_range_coords"] == 0


def test_label_error_rate_within_bound(data04):
    assert data04["label_error_rate_pct"] <= C.MAX_LABEL_ERROR_PCT


# --- methodology: the helpers compute the expected values ---
def test_polygon_area_unit_square():
    assert M.polygon_area([[0, 0], [1, 0], [1, 1], [0, 1]]) == pytest.approx(1.0)


def test_polygon_area_degenerate_is_zero():
    assert M.polygon_area(np.zeros((4, 2))) == pytest.approx(0.0)


def test_parse_label_reads_valid_object():
    objects, malformed = M.parse_obb_label("0 0 0 1 0 1 1 0 1")
    assert malformed == 0
    assert objects[0][0] == 0
    assert objects[0][1].shape == (4, 2)


def test_parse_label_counts_malformed():
    objects, malformed = M.parse_obb_label("0 0 0 1 0 1 1 0 1\nbad line\n1 2 3")
    assert len(objects) == 1
    assert malformed == 2
