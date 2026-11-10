"""End-to-end test, needs PyTorch (skipped automatically if it is not installed)."""
import pytest

pytest.importorskip("torch")
pytest.importorskip("torchvision")

from config import settings as cfg  # noqa: E402


def test_demo_pipeline_runs_for_every_cancer(monkeypatch):
    monkeypatch.setattr(cfg, "PRETRAINED_BACKBONE", False)
    monkeypatch.setattr(cfg, "LOG_HISTORY", False)
    import demo
    from modules import pipeline
    for cancer in cfg.CANCER_TYPES:
        result, overlay = pipeline.analyze(demo.synthetic_scan(), cancer, "synthetic")
        assert 0 <= result["assessment"]["score"] <= 1
        assert overlay.shape[:2] == (512, 512)
        assert result["trained"] is False
