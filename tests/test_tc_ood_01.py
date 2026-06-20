"""TC-OOD-01 - OOD input rejection.

Atomic checks that the max-confidence score separates in-distribution frames from
OOD inputs, plus known-answer checks of the AUROC / FPR statistics. The TBC
acceptance AUROC is reported as evidence; here we assert the separation is real
(better than chance, and ID more confident than OOD).
"""

import pytest

from harness import metrics as M

pytestmark = pytest.mark.tc("TC-OOD-01")


# --- separation on the real model ---
def test_auroc_in_unit_range(ood01):
    assert 0.0 <= ood01["AUROC"] <= 1.0


def test_auroc_better_than_chance(ood01):
    assert ood01["AUROC"] > 0.5


def test_fpr_in_unit_range(ood01):
    assert 0.0 <= ood01["FPR_at_95TPR"] <= 1.0


def test_id_more_confident_than_ood(ood01):
    assert ood01["id_confident_detection_rate"] > ood01["ood_confident_detection_rate"]


# --- methodology: the statistics behave as expected on known scores ---
def test_auroc_is_one_for_perfect_separation():
    assert M.auroc([0.9, 0.8, 0.95], [0.1, 0.2, 0.05]) == pytest.approx(1.0)


def test_auroc_is_zero_for_reversed_separation():
    assert M.auroc([0.1, 0.2], [0.9, 0.8]) == pytest.approx(0.0)


def test_auroc_is_half_for_identical_scores():
    assert M.auroc([0.5, 0.5], [0.5, 0.5]) == pytest.approx(0.5)


def test_fpr_zero_when_ood_below_threshold():
    assert M.fpr_at_tpr([0.6, 0.7, 0.8, 0.9, 1.0], [0.0, 0.1], tpr=0.95) == pytest.approx(0.0)


def test_fpr_one_when_ood_above_threshold():
    assert M.fpr_at_tpr([0.6, 0.7, 0.8, 0.9, 1.0], [1.0, 1.0], tpr=0.95) == pytest.approx(1.0)
