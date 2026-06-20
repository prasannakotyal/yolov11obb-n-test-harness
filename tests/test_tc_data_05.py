"""TC-DATA-05 - Data/model drift monitoring.

Atomic checks that the KS/JS monitor flags an induced brightness drift and not the
no-drift control, plus known-answer checks of the drift statistics.
"""

import numpy as np
import pytest

from harness import config as C
from harness import metrics as M

pytestmark = pytest.mark.tc("TC-DATA-05")


# --- drift on the real dataset ---
def test_induced_drift_detected(data05):
    assert data05["induced_drift_detected"] is True


def test_control_not_flagged(data05):
    assert data05["control_drift_detected"] is False


def test_induced_ks_below_alpha(data05):
    assert data05["induced_ks_p"] < C.DRIFT_ALPHA


def test_control_ks_above_alpha(data05):
    assert data05["control_ks_p"] >= C.DRIFT_ALPHA


def test_induced_js_exceeds_control_js(data05):
    assert data05["induced_js"] > data05["control_js"]


# --- methodology: the statistics behave as expected on known inputs ---
def test_ks_detects_a_clear_shift():
    rng = np.random.default_rng(0)
    assert M.ks_drift(rng.normal(0, 1, 500), rng.normal(5, 1, 500))["drift"] is True


def test_ks_no_false_alarm_on_same_distribution():
    rng = np.random.default_rng(1)
    sample = rng.normal(0, 1, 500)
    assert M.ks_drift(sample, sample)["drift"] is False


def test_js_distance_zero_for_identical():
    sample = np.random.default_rng(0).uniform(0, 255, 1000)
    assert M.js_distance(sample, sample) == pytest.approx(0.0, abs=1e-6)


def test_js_distance_large_for_disjoint():
    assert M.js_distance(np.full(500, 10.0), np.full(500, 240.0)) > 0.5
