"""Image corruptions and geometric transforms for the robustness cases.

A self-contained suite of 15 corruption types over 4 families (noise, blur,
weather, digital), each at 5 increasing severities, plus realistic geometric
transforms. Every function maps an image to an image of the same shape, so it
is testable without a model (see tests/test_corruptions.py). Functions that use
randomness read the global numpy RNG; seed it for reproducibility.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.util import random_noise

SEVERITIES = (1, 2, 3, 4, 5)


def _disk(radius: int) -> np.ndarray:
    size = int(radius) * 2 + 1
    yy, xx = np.ogrid[:size, :size]
    kernel = ((xx - radius) ** 2 + (yy - radius) ** 2 <= radius**2).astype(np.float32)
    return kernel / kernel.sum()


# ---- corruption functions: take a float32 RGB image + severity, return a float array ----
def gaussian_noise(x, s):
    sigma = [0.04, 0.06, 0.10, 0.16, 0.26][s - 1]
    return np.clip(x / 255.0 + np.random.normal(0, sigma, x.shape), 0, 1) * 255


def shot_noise(x, s):
    lam = [60, 25, 12, 5, 3][s - 1]
    return np.clip(np.random.poisson(np.clip(x / 255.0, 0, 1) * lam) / lam, 0, 1) * 255


def impulse_noise(x, s):
    amount = [0.01, 0.02, 0.04, 0.07, 0.10][s - 1]
    return np.clip(random_noise(x / 255.0, mode="s&p", amount=amount), 0, 1) * 255


def defocus_blur(x, s):
    return cv2.filter2D(x.astype(np.float32), -1, _disk([1, 2, 3, 4, 5][s - 1]))


def glass_blur(x, s):
    sigma = [0.7, 0.9, 1.2, 1.6, 2.0][s - 1]
    f = gaussian_filter(x.astype(np.float32), sigma=(sigma, sigma, 0))
    for _ in range(2):
        dx, dy = np.random.randint(-2, 3, 2)
        f = gaussian_filter(np.roll(np.roll(f, dx, 0), dy, 1), sigma=(sigma, sigma, 0))
    return np.clip(f, 0, 255)


def motion_blur(x, s):
    length = [5, 8, 12, 16, 20][s - 1]
    kernel = np.zeros((length, length), np.float32)
    kernel[length // 2, :] = 1.0 / length
    rot = cv2.getRotationMatrix2D(
        (length / 2, length / 2), np.random.uniform(0, 180), 1
    )
    kernel = cv2.warpAffine(kernel, rot, (length, length))
    kernel /= kernel.sum() + 1e-8
    return cv2.filter2D(x.astype(np.float32), -1, kernel)


def zoom_blur(x, s):
    zooms = np.arange(1, [1.06, 1.11, 1.16, 1.21, 1.26][s - 1], 0.02)
    h, w = x.shape[:2]
    acc = x.astype(np.float32).copy()
    for z in zooms:
        zoomed = cv2.resize(x, (int(w * z), int(h * z)))
        top, left = (zoomed.shape[0] - h) // 2, (zoomed.shape[1] - w) // 2
        acc += zoomed[top : top + h, left : left + w].astype(np.float32)
    return np.clip(acc / (len(zooms) + 1), 0, 255)


def fog(x, s):
    c = [0.2, 0.35, 0.5, 0.65, 0.8][s - 1]
    haze = gaussian_filter(np.random.rand(*x.shape[:2]).astype(np.float32), 64)[
        ..., None
    ]
    haze = 0.5 + 0.5 * (haze - haze.min()) / (np.ptp(haze) + 1e-8)
    return np.clip(x.astype(np.float32) * (1 - c) + c * 255 * haze, 0, 255)


def frost(x, s):
    a, b = [(0.9, 0.3), (0.8, 0.4), (0.7, 0.5), (0.65, 0.6), (0.6, 0.7)][s - 1]
    texture = (
        gaussian_filter(np.random.rand(*x.shape).astype(np.float32), (2, 2, 0)) * 255
    )
    return np.clip(a * x.astype(np.float32) + b * texture, 0, 255)


def snow(x, s):
    amount = [0.01, 0.02, 0.035, 0.05, 0.07][s - 1]
    layer = (np.random.rand(*x.shape[:2]) < amount).astype(np.float32)
    kernel = np.zeros((9, 9), np.float32)
    kernel[4, :] = 1 / 9
    rot = cv2.getRotationMatrix2D((4, 4), np.random.uniform(-30, 30), 1)
    layer = cv2.filter2D(layer, -1, cv2.warpAffine(kernel, rot, (9, 9)))[..., None]
    return np.clip(x.astype(np.float32) + layer * 255, 0, 255)


def brightness(x, s):
    return np.clip(
        x.astype(np.float32) + [0.1, 0.2, 0.3, 0.4, 0.5][s - 1] * 255, 0, 255
    )


def contrast(x, s):
    c = [0.75, 0.6, 0.5, 0.4, 0.3][s - 1]
    mean = x.astype(np.float32).mean(axis=(0, 1), keepdims=True)
    return np.clip(mean + (x - mean) * c, 0, 255)


def elastic_transform(x, s):
    sigma, alpha = [(6, 8), (6, 16), (5, 24), (5, 32), (4, 40)][s - 1]
    h, w = x.shape[:2]
    dx = gaussian_filter(np.random.rand(h, w) * 2 - 1, sigma) * alpha
    dy = gaussian_filter(np.random.rand(h, w) * 2 - 1, sigma) * alpha
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    map_x = (xx + dx).astype(np.float32)
    map_y = (yy + dy).astype(np.float32)
    return cv2.remap(x, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def pixelate(x, s):
    f = [0.6, 0.5, 0.4, 0.3, 0.25][s - 1]
    h, w = x.shape[:2]
    small = cv2.resize(
        x, (max(1, int(w * f)), max(1, int(h * f))), interpolation=cv2.INTER_LINEAR
    )
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def jpeg_compression(x, s):
    quality = [80, 65, 50, 38, 25][s - 1]
    _, enc = cv2.imencode(
        ".jpg", x[:, :, ::-1].astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, quality]
    )
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)[:, :, ::-1]


CORRUPTIONS = {
    "noise": {
        "gaussian_noise": gaussian_noise,
        "shot_noise": shot_noise,
        "impulse_noise": impulse_noise,
    },
    "blur": {
        "defocus_blur": defocus_blur,
        "glass_blur": glass_blur,
        "motion_blur": motion_blur,
        "zoom_blur": zoom_blur,
    },
    "weather": {"fog": fog, "frost": frost, "snow": snow, "brightness": brightness},
    "digital": {
        "contrast": contrast,
        "elastic_transform": elastic_transform,
        "pixelate": pixelate,
        "jpeg_compression": jpeg_compression,
    },
}


def apply_corruption(image_uint8, fn, severity: int) -> np.ndarray:
    """Apply a corruption function and clamp back to a uint8 RGB image."""
    return np.clip(fn(image_uint8.astype(np.float32), severity), 0, 255).astype(
        np.uint8
    )


# ---- realistic geometric transforms (metamorphic): take and return a uint8 image ----
def _zoom_out(x):
    pad = int(x.shape[0] * 0.15)
    shrunk = cv2.resize(x, (int(x.shape[1] * 0.7), int(x.shape[0] * 0.7)))
    return cv2.copyMakeBorder(shrunk, pad, pad, pad, pad, cv2.BORDER_CONSTANT)


GEOMETRIC = {
    "hflip": lambda x: x[:, ::-1],
    "vflip": lambda x: x[::-1, :],
    "rot90": lambda x: np.rot90(x, 1),
    "rot180": lambda x: np.rot90(x, 2),
    "rot270": lambda x: np.rot90(x, 3),
    "brighten": lambda x: np.clip(x.astype(np.float32) + 40, 0, 255).astype(np.uint8),
    "darken": lambda x: np.clip(x.astype(np.float32) - 40, 0, 255).astype(np.uint8),
    "zoom_out": _zoom_out,
}


def apply_geometric(image_uint8, fn) -> np.ndarray:
    """Apply a geometric transform and return a contiguous uint8 image."""
    return np.ascontiguousarray(fn(image_uint8)).astype(np.uint8)
