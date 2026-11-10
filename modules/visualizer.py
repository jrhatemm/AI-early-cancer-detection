"""OpenCV overlay: heatmap + tumor outlines + risk banner (scales with image size)."""
import cv2
import numpy as np

from config import settings as cfg

COLORS = {"Low": (80, 170, 40), "Moderate": (0, 165, 255), "High": (40, 40, 220)}  # BGR
FONT = cv2.FONT_HERSHEY_SIMPLEX


def overlay_heatmap(bgr, cam_full, alpha=cfg.HEATMAP_ALPHA):
    colored = cv2.applyColorMap((cam_full * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(bgr, 1 - alpha, colored, alpha, 0)


def draw_regions(img, regions, color, scale=1.0):
    t = max(1, int(round(2 * scale)))
    for i, r in enumerate(regions, 1):
        cv2.drawContours(img, [r["contour"]], -1, color, t)
        x, y, w, h = r["bbox"]
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), max(1, t - 1))
        cv2.putText(img, f"#{i} {r['area_pct']}%", (x, max(y - 6, int(16 * scale))),
                    FONT, 0.5 * scale, (255, 255, 255), max(1, t - 1), cv2.LINE_AA)
    return img


def render(bgr, cam_full, regions, assessment):
    scale = max(bgr.shape[1] / 640, 0.6)
    color = COLORS[assessment["level"]]
    # Low risk: show the clean scan, a heatmap on a normal scan would be misleading.
    out = bgr.copy() if assessment["level"] == "Low" else overlay_heatmap(bgr, cam_full)
    out = draw_regions(out, regions, color, scale)

    bh = int(36 * scale)
    cv2.rectangle(out, (0, 0), (out.shape[1], bh), color, -1)
    text = f"{assessment['level']} risk | score {assessment['score']:.0%} | {len(regions)} region(s)"
    cv2.putText(out, text, (int(10 * scale), int(bh * 0.68)), FONT, 0.65 * scale,
                (255, 255, 255), max(1, int(2 * scale)), cv2.LINE_AA)
    cv2.putText(out, "Research prototype - not for diagnosis", (int(10 * scale), out.shape[0] - int(10 * scale)),
                FONT, 0.45 * scale, (255, 255, 255), max(1, int(scale)), cv2.LINE_AA)
    return out
