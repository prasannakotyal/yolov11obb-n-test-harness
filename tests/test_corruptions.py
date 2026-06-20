"""Invariants for the corruption and geometric-transform suite.

These do not need the model: a corruption must return a same-shape uint8 image
in range, and must be deterministic once the RNG is seeded.
"""

import numpy as np
import pytest

from harness import corruptions as X

ALL_CORRUPTIONS = [
    (f"{group}/{name}", fn)
    for group, fns in X.CORRUPTIONS.items()
    for name, fn in fns.items()
]


def test_suite_has_15_corruptions():
    assert len(ALL_CORRUPTIONS) == 15


@pytest.mark.parametrize(
    "name,fn", ALL_CORRUPTIONS, ids=[n for n, _ in ALL_CORRUPTIONS]
)
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


@pytest.mark.parametrize("name,fn", list(X.GEOMETRIC.items()))
def test_geometric_returns_contiguous_uint8_image(name, fn, rgb_image):
    out = X.apply_geometric(rgb_image, fn)
    assert out.dtype == np.uint8
    assert out.ndim == 3
    assert out.flags["C_CONTIGUOUS"]


def test_hflip_is_an_involution(rgb_image):
    once = X.apply_geometric(rgb_image, X.GEOMETRIC["hflip"])
    twice = X.apply_geometric(once, X.GEOMETRIC["hflip"])
    assert np.array_equal(twice, rgb_image)
