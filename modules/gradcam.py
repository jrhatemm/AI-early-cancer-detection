"""Grad-CAM: explains which image regions drove the 'malignant' score."""
import numpy as np
import torch

from config import settings as cfg


class GradCAM:
    def __init__(self, model, layer, class_idx=1):
        self.model, self.class_idx = model, class_idx
        self.acts = self.grads = None
        self._handle = layer.register_forward_hook(self._on_forward)

    def _on_forward(self, module, inputs, output):
        self.acts = output
        output.register_hook(lambda g: setattr(self, "grads", g))

    def __call__(self, x):
        """Return (cam in 0..1 at feature-map size, class probabilities)."""
        with torch.enable_grad():
            logits = self.model(x)
            self.model.zero_grad()
            logits[0, self.class_idx].backward()
        weights = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.acts).sum(1))[0].detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        probs = torch.softmax(logits / cfg.TEMPERATURE, 1)[0].detach().cpu().numpy()
        return cam.astype(np.float32), probs

    def close(self):
        self._handle.remove()
