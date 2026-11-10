"""AI model zoo: ResNet-50, DenseNet-121, EfficientNet-B4 with a benign/malignant head."""
import logging

import torch
import torch.nn as nn
from torchvision import models

from config import settings as cfg

log = logging.getLogger(__name__)


def build_model(arch, pretrained=True):
    if arch == "efficientnet_b4":
        m = models.efficientnet_b4(weights="IMAGENET1K_V1" if pretrained else None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, 2)
    elif arch == "densenet121":
        m = models.densenet121(weights="IMAGENET1K_V1" if pretrained else None)
        m.classifier = nn.Linear(m.classifier.in_features, 2)
    else:
        m = models.resnet50(weights="IMAGENET1K_V2" if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, 2)
    return m


def target_layer(model):
    """Last convolutional block, used by Grad-CAM."""
    if hasattr(model, "layer4"):
        return model.layer4
    if hasattr(model.features, "denseblock4"):
        return model.features.denseblock4
    return model.features[-1]


class Classifier:
    CLASSES = ("benign", "malignant")

    def __init__(self, cancer):
        spec = cfg.CANCER_TYPES[cancer]
        self.cancer, self.arch = cancer, spec["arch"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        path = cfg.WEIGHTS_DIR / f"{cancer}.pt"
        self.trained = path.exists()
        self.model = build_model(self.arch, pretrained=cfg.PRETRAINED_BACKBONE and not self.trained)
        if self.trained:
            state = torch.load(path, map_location=self.device)
            if isinstance(state, dict) and "state_dict" in state:       # full checkpoint
                state = state["state_dict"]
            state = {k.removeprefix("module."): v for k, v in state.items()}   # DataParallel
            self.model.load_state_dict(state)
            log.info("Loaded trained weights: %s", path)
        else:
            log.warning("No weights at %s - head is untrained, results are NOT meaningful", path)
        self.model.to(self.device).eval()

    @property
    def layer(self):
        return target_layer(self.model)
