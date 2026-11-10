"""Convert model probabilities into a risk level, confidence and advice."""
import math

from config import settings as cfg

ADVICE = {
    "Low": "No strong signs detected. Continue routine screening.",
    "Moderate": "Some suspicious features. Review by a radiologist and consider follow-up imaging.",
    "High": "Highly suspicious regions. Urgent specialist review and biopsy evaluation recommended.",
}


def assess(p_malignant, probs):
    entropy = -sum(p * math.log(p + 1e-12) for p in probs) / math.log(len(probs))
    if p_malignant < cfg.RISK_LOW:
        level = "Low"
    elif p_malignant >= cfg.RISK_HIGH:
        level = "High"
    else:
        level = "Moderate"
    return {
        "score": round(float(p_malignant), 4),
        "level": level,
        "confidence": round(1 - entropy, 4),   # 0 = model is unsure, 1 = very sure
        "advice": ADVICE[level],
    }
