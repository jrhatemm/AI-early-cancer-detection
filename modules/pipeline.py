"""Full analysis pipeline shared by the CLI, demo and API (thread-safe)."""
import logging
import threading
import time
from datetime import datetime

import torch

from config import settings as cfg
from modules import history, preprocessor, quality_check, risk_assessor, segmenter, visualizer
from modules.classifier import Classifier
from modules.gradcam import GradCAM

log = logging.getLogger(__name__)
_models = {}
_lock = threading.Lock()      # Grad-CAM hooks live on the shared model, so analyses run one at a time


class ImageQualityError(ValueError):
    """The image is unusable (too small, blank...)."""


def get_classifier(cancer):
    if cancer not in cfg.CANCER_TYPES:
        raise ValueError(f"Unsupported cancer type: {cancer}")
    if cancer not in _models:
        _models[cancer] = Classifier(cancer)
    return _models[cancer]


def analyze(bgr, cancer=cfg.DEFAULT_CANCER, source="memory"):
    quality = quality_check.check(bgr)
    if quality["errors"]:
        raise ImageQualityError("; ".join(quality["errors"]))

    t0 = time.time()
    with _lock:
        clf = get_classifier(cancer)
        x = preprocessor.to_tensor(bgr, cfg.INPUT_SIZE.get(clf.arch, cfg.IMAGE_SIZE)).to(clf.device)

        engine = GradCAM(clf.model, clf.layer)
        try:
            cam, probs = engine(x)
        finally:
            engine.close()          # hooks must be removed before the no_grad TTA pass

        if cfg.USE_TTA:             # average with the mirrored image for a steadier score
            with torch.no_grad():
                p2 = torch.softmax(clf.model(x.flip(3)) / cfg.TEMPERATURE, 1)[0].cpu().numpy()
            probs = (probs + p2) / 2

    cam_full, _mask, regions = segmenter.locate_regions(cam, bgr.shape)
    assessment = risk_assessor.assess(float(probs[1]), probs)
    if assessment["level"] == "Low":
        regions = []                # Grad-CAM is min-max normalised, so a normal scan still has a "hottest" spot
    overlay = visualizer.render(bgr, cam_full, regions, assessment)

    result = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "source": str(source),
        "cancer": cancer,
        "modality": cfg.CANCER_TYPES[cancer]["modality"],
        "model": clf.arch,
        "trained": clf.trained,
        "assessment": assessment,
        "regions": [{k: v for k, v in r.items() if k != "contour"} for r in regions],
        "quality": quality,
        "tta": cfg.USE_TTA,
        "seconds": round(time.time() - t0, 2),
    }
    if cfg.LOG_HISTORY:
        try:
            history.log_scan(result)
        except Exception:           # history must never break an analysis
            log.exception("Could not write scan history")
    log.info("%s -> %s risk (%.1f%%) in %.2fs", cancer, assessment["level"],
             100 * assessment["score"], result["seconds"])
    return result, overlay
