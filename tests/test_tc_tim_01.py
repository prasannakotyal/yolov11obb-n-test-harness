"""TC-TIM-01 - Per-frame inference and end-to-end latency.

Atomic checks of the mean latency budgets and the timing invariants. p95 is
recorded as evidence; on reference hardware it may exceed the budget that binds on
the deployment target, so it is not asserted against the budget here.
"""

import pytest

from harness import config as C

pytestmark = pytest.mark.tc("TC-TIM-01")


def test_all_frames_timed(tim01):
    assert tim01["images_timed"] == C.TIM_IMAGES


def test_mean_inference_within_budget(tim01):
    assert tim01["inference_ms_mean"] <= C.INFER_BUDGET_MS


def test_mean_end_to_end_within_budget(tim01):
    assert tim01["end_to_end_ms_mean"] <= C.E2E_BUDGET_MS


def test_end_to_end_not_faster_than_inference(tim01):
    assert tim01["end_to_end_ms_mean"] >= tim01["inference_ms_mean"]


def test_p95_not_below_mean(tim01):
    assert tim01["inference_ms_p95"] >= tim01["inference_ms_mean"]
