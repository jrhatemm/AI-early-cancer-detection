"""Entry point.

  python main.py --image scan.png --cancer lung
  python main.py --folder ./scans --cancer breast      # batch + CSV summary
  python main.py --serve                               # REST API on :8000
  python main.py --history                             # last 20 analyses
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import settings as cfg  # noqa: E402

log = logging.getLogger("cancer-detection")


def run_cli(args):
    from modules import image_loader, pipeline, report_generator
    paths = [Path(args.image)] if args.image else list(image_loader.iter_images(args.folder))
    if not paths:
        log.error("No images found")
        return
    results = []
    for p in paths:
        try:
            result, overlay = pipeline.analyze(image_loader.load_image(p), args.cancer, source=p)
        except (ValueError, FileNotFoundError) as e:      # one bad file must not stop a batch
            log.error("Skipping %s: %s", p, e)
            continue
        report_generator.save(result, overlay, stem=p.stem)
        results.append(result)
        print(report_generator.build_report(result), "\n")
    if len(results) > 1:
        print("Summary CSV:", report_generator.save_summary(results))


def main():
    ap = argparse.ArgumentParser(description="AI early tumor detection")
    ap.add_argument("--image")
    ap.add_argument("--folder")
    ap.add_argument("--cancer", default=cfg.DEFAULT_CANCER, choices=list(cfg.CANCER_TYPES))
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--history", action="store_true")
    args = ap.parse_args()

    cfg.setup_logging()
    if args.serve:
        import uvicorn
        uvicorn.run("modules.api_server:app", host=cfg.API_HOST, port=cfg.API_PORT)
    elif args.history:
        from modules import history
        for row in history.recent():
            print(row)
    elif args.image or args.folder:
        run_cli(args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
