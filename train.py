"""Train a model:  python train.py --cancer skin --data ./data/isic --epochs 15"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import settings as cfg  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Fine-tune a cancer classifier")
    ap.add_argument("--cancer", required=True, choices=list(cfg.CANCER_TYPES))
    ap.add_argument("--data", required=True, help="folder with train/ and val/ sub-folders")
    ap.add_argument("--epochs", type=int, default=cfg.TRAIN_EPOCHS)
    ap.add_argument("--batch", type=int, default=cfg.TRAIN_BATCH)
    ap.add_argument("--lr", type=float, default=cfg.TRAIN_LR)
    a = ap.parse_args()
    cfg.setup_logging()
    from modules import trainer
    trainer.train(a.cancer, a.data, a.epochs, a.batch, a.lr)


if __name__ == "__main__":
    main()
