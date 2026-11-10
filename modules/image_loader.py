"""Load medical images from disk or raw bytes (PNG/JPG/TIFF/DICOM).

16-bit scans (common for X-ray / CT TIFF and DICOM) are windowed to 8-bit
using the 1st-99th percentile instead of being crushed by a plain cast.
"""
import logging
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)
EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".dcm"}


def to_uint8(px):
    """Window any numeric array to 0-255 using percentiles."""
    if px.dtype == np.uint8:
        return px
    px = px.astype(np.float32)
    lo, hi = np.percentile(px, (1, 99))
    if hi <= lo:
        lo, hi = float(px.min()), float(px.max())
    return (255 * (px.clip(lo, hi) - lo) / (hi - lo + 1e-8)).astype(np.uint8)


def to_bgr(img):
    img = to_uint8(img)
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def _load_dicom(path):
    import pydicom
    ds = pydicom.dcmread(str(path))
    px = ds.pixel_array.astype(np.float32)
    if px.ndim == 3 and px.shape[-1] not in (3, 4):   # multi-frame: take middle slice
        px = px[px.shape[0] // 2]
    px = px * float(getattr(ds, "RescaleSlope", 1)) + float(getattr(ds, "RescaleIntercept", 0))
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        px = px.max() - px                             # MONOCHROME1 is inverted
    return to_bgr(px)


def load_image(path):
    """Return a BGR uint8 image. Patient metadata in DICOM files is never read."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".dcm":
        return _load_dicom(path)
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return to_bgr(img)


def decode_bytes(data: bytes):
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Uploaded file is not a valid image")
    return to_bgr(img)


def iter_images(folder):
    for p in sorted(Path(folder).rglob("*")):
        if p.suffix.lower() in EXTS:
            yield p
