"""All tunable parameters for Oncoscan live here."""
import logging
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = BASE_DIR / "weights"
OUTPUT_DIR = BASE_DIR / "outputs"
LOG_FILE = BASE_DIR / "logs" / "system.log"

# --- Pre-processing ---
IMAGE_SIZE = 224
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
USE_CLAHE = True                 # contrast enhancement for X-ray / CT
PRETRAINED_BACKBONE = True       # download ImageNet weights if no trained .pt exists

# --- Supported cancers: architecture + imaging modality ---
CANCER_TYPES = {
    "lung":       {"arch": "densenet121",     "modality": "Chest X-ray / CT"},
    "breast":     {"arch": "resnet50",        "modality": "Mammogram"},
    "skin":       {"arch": "efficientnet_b4", "modality": "Dermoscopy"},
    "brain":      {"arch": "resnet50",        "modality": "MRI"},
    "colorectal": {"arch": "efficientnet_b4", "modality": "Histopathology"},
    "prostate":   {"arch": "resnet50",        "modality": "MRI"},
    "liver":      {"arch": "resnet50",        "modality": "CT scan"},
    "cervical":   {"arch": "efficientnet_b4", "modality": "Pap smear"},
}
DEFAULT_CANCER = "lung"

# --- Risk thresholds (probability of malignancy) ---
RISK_LOW = 0.35
RISK_HIGH = 0.70

# --- Tumor localisation from the Grad-CAM heatmap ---
CAM_THRESHOLD = 0.6              # heat above this counts as a suspicious region
MIN_REGION_AREA_PCT = 0.5        # ignore regions smaller than 0.5% of the image
HEATMAP_ALPHA = 0.45

# --- Inference ---
TEMPERATURE = 1.0                # >1 softens over-confident scores (calibrate on validation data)
USE_TTA = True                   # average original + horizontally flipped prediction
INPUT_SIZE = {"efficientnet_b4": 380}   # per-architecture input size, others use IMAGE_SIZE

# --- Image quality gate ---
MIN_IMAGE_SIDE = 128             # reject images smaller than this (pixels)
BLUR_THRESHOLD = 40.0            # Laplacian variance below this = warning
DARK_MEAN, BRIGHT_MEAN = 25, 230 # mean intensity outside this range = warning

# --- Training ---
TRAIN_EPOCHS = 15
TRAIN_BATCH = 16
TRAIN_LR = 1e-4
EARLY_STOP_PATIENCE = 5

# --- History (anonymised: only a hash of the filename is stored) ---
LOG_HISTORY = True
HISTORY_DB = BASE_DIR / "logs" / "history.db"

# --- API ---
MAX_UPLOAD_MB = 20
CORS_ORIGINS = ["*"]             # restrict to your website's domain in production
API_HOST = "0.0.0.0"
API_PORT = 8000


def setup_logging(level=logging.INFO):
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
    )
