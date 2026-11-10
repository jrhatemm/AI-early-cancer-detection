"""Write a text report, JSON result and overlay image for each scan (+ batch CSV summary)."""
import csv
import json
import logging
import re
from datetime import datetime
from pathlib import Path

import cv2

from config import settings as cfg

log = logging.getLogger(__name__)
DISCLAIMER = "Research prototype. Not a medical device. Not for diagnosis."


def build_report(result):
    a, q = result["assessment"], result.get("quality", {})
    lines = [
        "SCAN REPORT",
        f"Time        : {result['timestamp']}",
        f"Source      : {result['source']}",
        f"Cancer type : {result['cancer']} ({result['modality']})",
        f"Model       : {result['model']} (trained weights: {result['trained']})",
        f"Risk        : {a['level']}  (score {a['score']:.1%}, confidence {a['confidence']:.1%})",
        f"Regions     : {len(result['regions'])}",
    ]
    for i, r in enumerate(result["regions"], 1):
        lines.append(f"  #{i} bbox={r['bbox']} area={r['area_pct']}% of image")
    for w in q.get("warnings", []):
        lines.append(f"Quality     : WARNING - {w}")
    if not result["trained"]:
        lines.append("WARNING     : model has no trained weights, this result is NOT meaningful")
    lines += [f"Advice      : {a['advice']}", "", DISCLAIMER]
    return "\n".join(lines)


def save(result, overlay, stem):
    """Save .json, .txt and _overlay.png. Returns the base path (no extension)."""
    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^\w\-]+", "_", str(stem))          # dots in names broke Path.with_suffix
    base = str(cfg.OUTPUT_DIR / f"{safe}_{datetime.now():%Y%m%d_%H%M%S_%f}")
    Path(base + ".json").write_text(json.dumps(result, indent=2))
    Path(base + ".txt").write_text(build_report(result))
    cv2.imwrite(base + "_overlay.png", overlay)
    log.info("Saved report to %s.*", base)
    return base


def save_summary(results, name="summary"):
    """One CSV row per scan - useful after a folder batch run."""
    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = cfg.OUTPUT_DIR / f"{name}_{datetime.now():%Y%m%d_%H%M%S}.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", "cancer", "level", "score", "confidence", "regions", "trained"])
        for r in results:
            a = r["assessment"]
            w.writerow([Path(r["source"]).name, r["cancer"], a["level"], a["score"],
                        a["confidence"], len(r["regions"]), r["trained"]])
    return path
