from __future__ import annotations

import argparse
import json
import sys

from .engine import WatermarkRemover


def _common_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--mask", help="optional full-frame grayscale watermark mask")
    parser.add_argument("--sample-count", type=int, default=18)
    parser.add_argument("--min-confidence", type=float, default=0.55)
    parser.add_argument("--debug-dir")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="watermark-remover",
        description="Provider-agnostic transparent video overlay remover",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser("analyze", help="detect and estimate an overlay")
    analyze.add_argument("input")
    analyze.add_argument("--report")
    _common_options(analyze)

    remove = subparsers.add_parser("remove", help="remove an overlay from a video")
    remove.add_argument("input")
    remove.add_argument("output")
    remove.add_argument("--report")
    remove.add_argument("--crf", type=int, default=18)
    remove.add_argument("--preset", default="medium")
    _common_options(remove)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    engine = WatermarkRemover(
        sample_count=args.sample_count,
        min_confidence=args.min_confidence,
    )

    try:
        if args.command == "analyze":
            result = engine.analyze(
                args.input,
                mask_path=args.mask,
                debug_dir=args.debug_dir,
            )
        else:
            result = engine.remove(
                args.input,
                args.output,
                mask_path=args.mask,
                debug_dir=args.debug_dir,
                crf=args.crf,
                preset=args.preset,
            )

        if args.report:
            engine.save_report(result.report, args.report)
        print(json.dumps(result.report, indent=2))
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
