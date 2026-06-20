"""Shared pytest configuration.

Unit tests cover the pure verification logic (no model, no GPU). The single
integration test is marked `slow` and skips itself when the model or dataset is
absent. Run only the fast suite with:  pytest -m "not slow"
"""

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: end-to-end test that loads the model under test"
    )


@pytest.fixture
def rgb_image():
    """A small deterministic uint8 RGB image for transform tests."""
    import numpy as np

    return (np.random.default_rng(0).random((64, 64, 3)) * 255).astype(np.uint8)
