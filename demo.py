"""Demo mode: runs the full pipeline on a synthetic scan.

No dataset, no trained weights and no internet needed. Because the model head is
untrained, the risk score is random - this only proves the pipeline works.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))   # run from any folder
from config import settings as cfg  # noqa: E402

cfg.PRETRAINED_BACKBONE = False       # skip ImageNet download for the demo
cfg.LOG_HISTORY = False               # demo runs are not real scans
cfg.setup_logging()

from modules import pipeline, report_generator  # noqa: E402


def synthetic_scan(size=512, seed=7):
    rng = np.random.default_rng(seed)
    img = rng.normal(90, 18, (size, size)).clip(0, 255).astype(np.float32)
    yy, xx = np.mgrid[:size, :size]
    for cx, cy, r in [(300, 220, 38), (180, 340, 22)]:       # fake "tumors"
        img += 120 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r ** 2))
    img = cv2.GaussianBlur(img.clip(0, 255).astype(np.uint8), (5, 5), 0)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


if __name__ == "__main__":
    cancer = sys.argv[1] if len(sys.argv) > 1 else cfg.DEFAULT_CANCER
    result, overlay = pipeline.analyze(synthetic_scan(), cancer, source="synthetic")
    report_generator.save(result, overlay, stem=f"demo_{cancer}")
    print(report_generator.build_report(result))
