"""Command line entry: python -m sktime.benchmarking.feature_extraction."""

import argparse
import json
import sys
from pathlib import Path

from sktime.benchmarking.feature_extraction._data import DATASETS, load_dataset
from sktime.benchmarking.feature_extraction._engine import (
    extract_features,
    json_safe,
    run_benchmark,
    write_json,
)
from sktime.benchmarking.feature_extraction._methods import PRESETS, list_methods


def main(argv=None):
    """Run list, extract or benchmark; errors never masquerade as success."""
    parser = argparse.ArgumentParser(
        description="Lecture 2 feature extraction and reproducible runtime benchmark"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "list", help="List methods, parameters, presets and offline datasets"
    )
    for name in ("extract", "benchmark"):
        sub = commands.add_parser(name)
        sub.add_argument(
            "--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS)
        )
        sub.add_argument("--preset", choices=list(PRESETS), default=None)
        sub.add_argument("--methods", nargs="+", default=None)
        sub.add_argument(
            "--config",
            type=Path,
            help=(
                "JSON: preset, methods, parameters, preprocessing, "
                "seed, window_length, stride"
            ),
        )
        sub.add_argument(
            "--out",
            type=Path,
            required=True,
            help="New output directory; existing directories are never overwritten",
        )
        sub.add_argument("--seed", type=int, default=None)
        sub.add_argument("--window-length", type=int, default=None)
        sub.add_argument("--stride", type=int, default=None)
        if name == "benchmark":
            sub.add_argument("--repeats", type=int, default=3)
            sub.add_argument("--warmups", type=int, default=1)
    args = parser.parse_args(argv)
    if args.command == "list":
        print(
            json.dumps(
                {"datasets": DATASETS, "presets": PRESETS, "methods": list_methods()},
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    try:
        config = (
            json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
        )
        allowed = {
            "preset",
            "methods",
            "parameters",
            "preprocessing",
            "seed",
            "window_length",
            "stride",
        }
        if not isinstance(config, dict) or set(config) - allowed:
            raise ValueError(f"config keys must be a subset of {sorted(allowed)}")
        for key in ("preset", "methods", "seed", "window_length", "stride"):
            value = getattr(args, key)
            if value is not None:
                config[key] = value
        if args.out.exists():
            raise ValueError(f"Output directory already exists: {args.out}")
        if args.command == "benchmark":
            result = run_benchmark(
                datasets=args.datasets,
                output_dir=args.out,
                repeats=args.repeats,
                warmups=args.warmups,
                progress=lambda s: print(s, file=sys.stderr, flush=True),
                **config,
            )
            failed = any(
                r["status"] in ("failed", "blocked")
                or r.get("status_counts", {}).get("failed", 0)
                or r.get("status_counts", {}).get("blocked", 0)
                or not r.get("repeat_consistent", True)
                for r in result["summary"]
            )
        else:
            args.out.mkdir(parents=True)
            result = {}
            failed = False
            for dataset in args.datasets:
                print(dataset, file=sys.stderr, flush=True)
                extracted = extract_features(
                    load_dataset(dataset), output_dir=args.out / dataset, **config
                )
                counts = extracted.diagnostics.status.value_counts().to_dict()
                result[dataset] = {
                    "samples": len(extracted.features),
                    "scalar_columns": extracted.features.shape[1],
                    "status_counts": counts,
                }
                failed |= bool(counts.get("failed") or counts.get("blocked"))
        write_json(
            args.out / "command.json",
            {
                "argv": [
                    "python",
                    "-m",
                    "sktime.benchmarking.feature_extraction",
                    *(sys.argv[1:] if argv is None else argv),
                ],
                "config": config,
            },
        )
        print(
            json.dumps(
                json_safe(
                    {
                        "output_dir": str(args.out),
                        "status": "completed_with_errors" if failed else "completed",
                        "results": result.get("summary", result),
                    }
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 1 if failed else 0
    except (ValueError, TypeError, OSError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
