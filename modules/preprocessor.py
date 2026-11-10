"""CLAHE contrast enhancement, resize, normalise -> model tensor."""
import cv2
import numpy as np
import torch

from config import settings as cfg


def apply_clahe(bgr):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def to_tensor(bgr, size=None):
    """BGR uint8 image -> (1, 3, size, size) float tensor."""
    size = size or cfg.IMAGE_SIZE
    if cfg.USE_CLAHE:
        bgr = apply_clahe(bgr)
    rgb = cv2.cvtColor(cv2.resize(bgr, (size, size)), cv2.COLOR_BGR2RGB)
    rgb = rgb.astype(np.float32) / 255.0
    rgb = (rgb - np.array(cfg.MEAN, np.float32)) / np.array(cfg.STD, np.float32)
    return torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0)
