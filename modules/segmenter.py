"""Turn the heatmap into tumor regions (contours, boxes, area %).

For pixel-accurate outlines, swap this for a U-Net (segmentation_models_pytorch)
trained on BraTS / LiTS and return the same region dicts.
"""
import cv2
import numpy as np

from config import settings as cfg


def locate_regions(cam, image_shape):
    h, w = image_shape[:2]
    cam_full = cv2.resize(cam, (w, h), interpolation=cv2.INTER_CUBIC).clip(0, 1)
    mask = (cam_full >= cfg.CAM_THRESHOLD).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regions = []
    for c in contours:
        area_pct = 100 * cv2.contourArea(c) / (h * w)
        if area_pct < cfg.MIN_REGION_AREA_PCT:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        regions.append({"bbox": [x, y, bw, bh], "area_pct": round(float(area_pct), 2), "contour": c})
    regions.sort(key=lambda r: -r["area_pct"])
    return cam_full, mask, regions
