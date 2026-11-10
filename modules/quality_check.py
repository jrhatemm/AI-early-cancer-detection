"""Image quality gate: reject unusable scans, warn about poor ones."""
import cv2

from config import settings as cfg


def check(bgr):
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    mean, contrast = float(gray.mean()), float(gray.std())

    errors, warnings = [], []
    if min(h, w) < cfg.MIN_IMAGE_SIDE:
        errors.append(f"image too small ({w}x{h}px, minimum side {cfg.MIN_IMAGE_SIDE}px)")
    if contrast < 2:
        errors.append("image is blank or has no contrast")
    if sharp < cfg.BLUR_THRESHOLD:
        warnings.append("image looks blurry, results may be unreliable")
    if mean < cfg.DARK_MEAN:
        warnings.append("image is very dark")
    if mean > cfg.BRIGHT_MEAN:
        warnings.append("image is overexposed")
    return {"ok": not errors, "errors": errors, "warnings": warnings,
            "sharpness": round(sharp, 1), "brightness": round(mean, 1), "contrast": round(contrast, 1)}
