"""TC-DATA-03 - Train/validation/test separation (no leakage).

Atomic checks for exact (MD5) and near (16x16 average-hash) duplicates within the set.
"""

import pytest

pytestmark = pytest.mark.tc("TC-DATA-03")


def test_val_set_nonempty(data03):
    assert data03["val_images"] > 0


def test_no_exact_duplicates(data03):
    assert data03["exact_duplicates_within_val"] == 0


def test_no_near_duplicates(data03):
    assert data03["near_duplicate_pairs_within_val"] == 0
