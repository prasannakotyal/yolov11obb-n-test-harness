"""TC-DATA-01 - Dataset integrity and class distribution.

Atomic checks over the real dataset: pairing, readability, MD5 duplicates,
malformed/out-of-range labels, class presence and imbalance.
"""

import pytest

from harness import config as C

pytestmark = pytest.mark.tc("TC-DATA-01")


def test_images_present(data01):
    assert data01["images"] > 0


def test_labels_present(data01):
    assert data01["labels"] > 0


def test_every_image_has_a_label(data01):
    assert data01["unpaired_files"] == 0


def test_all_images_readable(data01):
    assert data01["unreadable_images"] == 0


def test_no_duplicate_images(data01):
    assert data01["duplicate_images"] == 0


def test_no_malformed_label_lines(data01):
    assert data01["malformed_label_lines"] == 0


def test_all_class_ids_in_range(data01):
    assert data01["out_of_range_class"] == 0


def test_all_classes_present(data01):
    assert data01["classes_present"] == C.NUM_CLASSES


def test_total_instances_positive(data01):
    assert data01["total_instances"] > 0


def test_class_imbalance_is_measured(data01):
    assert data01["class_imbalance_ratio"] is not None
    assert data01["class_imbalance_ratio"] >= 1.0
