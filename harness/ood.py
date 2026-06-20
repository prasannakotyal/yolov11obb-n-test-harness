"""Out-of-distribution verification (black-box): TC-OOD-01 OOD input rejection.

OOD score = maximum detection confidence, a pure input/output signal. In-distribution
frames should score high (confident detections); OOD inputs (uniform noise,
block-scrambled scenes) should score low. Reported as AUROC, FPR@95%TPR and the
rate of confident target declarations on OOD inputs.
"""

from __future__ import annotations

import cv2
import numpy as np

from . import common as H
from . import config as C
from . import metrics as M


def _scramble(img, k: int = 8) -> np.ndarray:
    """Shuffle an image into a k x k grid of blocks (destroys global structure)."""
    h, w = img.shape[:2]
    bh, bw = h // k, w // k
    blocks = [img[i * bh : (i + 1) * bh, j * bw : (j + 1) * bw] for i in range(k) for j in range(k)]
    np.random.shuffle(blocks)
    return np.vstack([np.hstack(blocks[i * k : (i + 1) * k]) for i in range(k)])


def _resize(path) -> np.ndarray:
    return cv2.resize(H.read_rgb(path), (C.STD_IMG, C.STD_IMG), interpolation=cv2.INTER_AREA)


def tc_ood_01(model) -> dict:
    """OOD input rejection via max-confidence score."""
    np.random.seed(C.SEED)
    id_imgs = [_resize(p) for p in H.val_images(C.OOD_IMAGES)]
    n_ood = C.OOD_IMAGES // 2
    ood_imgs = [(np.random.rand(C.STD_IMG, C.STD_IMG, 3) * 255).astype(np.uint8) for _ in range(n_ood)]
    ood_imgs += [_scramble(im) for im in id_imgs[:n_ood]]

    s_id = np.array([H.dets_of(H.predict(model, im))["max_conf"] for im in id_imgs])
    s_ood = np.array([H.dets_of(H.predict(model, im))["max_conf"] for im in ood_imgs])
    return {
        "id_images": len(id_imgs),
        "ood_images": len(ood_imgs),
        "ood_types": ["uniform noise", "block-scrambled scene"],
        "score": "max detection confidence (black-box)",
        "AUROC": round(M.auroc(s_id, s_ood), 3),
        "FPR_at_95TPR": round(M.fpr_at_tpr(s_id, s_ood, 0.95), 3),
        "ood_confident_detection_rate": round(float((s_ood >= C.CONF).mean()), 3),
        "id_confident_detection_rate": round(float((s_id >= C.CONF).mean()), 3),
    }
