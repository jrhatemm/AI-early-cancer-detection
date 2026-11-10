# AI early tumor detection (research prototype)

Not a medical device. Not for diagnosis.

## Quick start
    pip install -r requirements.txt
    python demo.py                              # pipeline test on a synthetic scan
    python main.py --image scan.png --cancer lung
    python main.py --folder ./scans --cancer breast   # batch + CSV summary
    python main.py --serve                      # REST API: POST /api/analyze

## Make the results real
The models start untrained. Fine-tune each cancer type on a labelled dataset:

    data/train/benign/  data/train/malignant/  data/val/benign/  data/val/malignant/
    python train.py --cancer skin --data ./data --epochs 15

Best weights (by validation AUC) are saved to `weights/<cancer>.pt`, with a suggested
risk threshold in `weights/<cancer>.json`. Use it to set `RISK_HIGH` in `config/settings.py`.

## Tests
    pytest -q
