"""Tests that run without PyTorch:  pytest -q"""
import json

import cv2
import numpy as np
import pytest

from config import settings as cfg
from modules import (history, image_loader, metrics, quality_check, report_generator,
                     risk_assessor, segmenter, visualizer)


def scan(size=300, blob=True):
    rng = np.random.default_rng(1)
    img = rng.normal(100, 25, (size, size)).clip(0, 255).astype(np.uint8)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    if blob:
        cv2.circle(img, (150, 150), 30, 230, -1)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


# --- metrics ---
def test_auc_perfect_and_random():
    assert metrics.roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert metrics.roc_auc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) == 0.5      # ties
    assert np.isnan(metrics.roc_auc([1, 1], [0.2, 0.9]))                    # one class only


def test_summary_and_threshold():
    y, s = [0, 0, 0, 1, 1, 1], [0.1, 0.2, 0.6, 0.4, 0.8, 0.9]
    m = metrics.summary(y, s, 0.5)
    assert (m["tp"], m["fp"], m["tn"], m["fn"]) == (2, 1, 2, 1)
    assert m["sensitivity"] == pytest.approx(2 / 3)
    assert 0.4 <= metrics.youden_threshold(y, s) <= 0.8


# --- risk ---
@pytest.mark.parametrize("p,level", [(0.1, "Low"), (0.5, "Moderate"), (0.9, "High")])
def test_risk_levels(p, level):
    assert risk_assessor.assess(p, [1 - p, p])["level"] == level


def test_confidence_is_low_when_unsure():
    assert risk_assessor.assess(0.5, [0.5, 0.5])["confidence"] < 0.01
    assert risk_assessor.assess(0.99, [0.01, 0.99])["confidence"] > 0.8


# --- quality gate ---
def test_quality_rejects_tiny_and_blank():
    assert quality_check.check(np.zeros((50, 50, 3), np.uint8))["errors"]
    assert quality_check.check(np.full((300, 300, 3), 128, np.uint8))["errors"]


def test_quality_warns_blur_but_accepts():
    q = quality_check.check(cv2.GaussianBlur(scan(), (31, 31), 0))
    assert q["ok"] and any("blurry" in w for w in q["warnings"])
    assert quality_check.check(scan())["ok"]


# --- loading ---
def test_16bit_image_is_windowed_not_crushed(tmp_path):
    px = (np.linspace(2000, 30000, 256 * 256).reshape(256, 256)).astype(np.uint16)
    cv2.imwrite(str(tmp_path / "x.png"), px)
    img = image_loader.load_image(tmp_path / "x.png")
    assert img.dtype == np.uint8 and img.shape == (256, 256, 3) and img.max() > 200 and img.min() < 30


def test_bad_bytes_raise():
    with pytest.raises(ValueError):
        image_loader.decode_bytes(b"not an image")


# --- segmenter + visualizer ---
def fake_cam():
    cam = np.zeros((7, 7), np.float32)
    cam[2:5, 2:5] = 1.0
    return cam


def test_segmenter_finds_region():
    _, mask, regions = segmenter.locate_regions(fake_cam(), (300, 300, 3))
    assert mask.sum() > 0 and len(regions) >= 1 and regions[0]["area_pct"] > cfg.MIN_REGION_AREA_PCT


@pytest.mark.parametrize("level", ["Low", "Moderate", "High"])
def test_visualizer_keeps_shape(level):
    img = scan()
    cam_full, _, regions = segmenter.locate_regions(fake_cam(), img.shape)
    a = risk_assessor.assess({"Low": 0.1, "Moderate": 0.5, "High": 0.9}[level], [0.5, 0.5])
    out = visualizer.render(img, cam_full, [] if level == "Low" else regions, a)
    assert out.shape == img.shape


# --- reports (the dotted filename bug) + history ---
def result(source="scan.v2.final.png"):
    a = risk_assessor.assess(0.8, [0.2, 0.8])
    return {"timestamp": "2026-01-01T00:00:00", "source": source, "cancer": "lung", "modality": "X-ray",
            "model": "resnet50", "trained": False, "assessment": a, "regions": [], "quality": {"warnings": []}}


def test_report_saves_dotted_names(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "OUTPUT_DIR", tmp_path)
    base = report_generator.save(result(), scan(), "scan.v2.final")
    for ext in (".json", ".txt", "_overlay.png"):
        assert (tmp_path / (base.split("/")[-1] + ext)).exists()
    assert json.loads(open(base + ".json").read())["cancer"] == "lung"
    assert "NOT meaningful" in open(base + ".txt").read()
    assert report_generator.save_summary([result(), result()]).exists()


def test_history_stores_hash_not_filename(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "HISTORY_DB", tmp_path / "h.db")
    history.log_scan(result("John_Smith_chest.png"))
    row = history.recent()[0]
    assert "John" not in json.dumps(row) and len(row["source_hash"]) == 12
    assert history.stats()["lung"]["High"] == 1
