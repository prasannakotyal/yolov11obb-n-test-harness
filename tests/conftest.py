"""Shared fixtures.

The model and dataset are always required: the fixtures load them directly, so a
missing model or dataset makes the suite error (there is nothing to skip). Each
case is measured once per session and its result is shared by the atomic tests for
that test case.
"""

import numpy as np
import pytest

from harness import common as H
from harness import data_tests, model_tests, ood, robustness


@pytest.fixture(scope="session")
def model():
    return H.load_model()


# --- one measured result per case, computed once and shared by that case's tests ---
@pytest.fixture(scope="session")
def data01():
    return data_tests.tc_data_01()


@pytest.fixture(scope="session")
def data02():
    return data_tests.tc_data_02()


@pytest.fixture(scope="session")
def data03():
    return data_tests.tc_data_03()


@pytest.fixture(scope="session")
def data04():
    return data_tests.tc_data_04()


@pytest.fixture(scope="session")
def data05():
    return data_tests.tc_data_05()


@pytest.fixture(scope="session")
def rob01(model):
    return robustness.tc_rob_01(model)


@pytest.fixture(scope="session")
def rob02(model):
    return robustness.tc_rob_02(model)


@pytest.fixture(scope="session")
def ood01(model):
    return ood.tc_ood_01(model)


@pytest.fixture(scope="session")
def perf01(model):
    return model_tests.tc_perf_01(model)


@pytest.fixture(scope="session")
def tim01(model):
    return model_tests.tc_tim_01(model)


@pytest.fixture(scope="session")
def fun01(model):
    return model_tests.tc_fun(model)


@pytest.fixture
def rgb_image():
    """A small deterministic uint8 RGB image for transform tests."""
    return (np.random.default_rng(0).random((64, 64, 3)) * 255).astype(np.uint8)
