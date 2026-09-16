import argparse
import asyncio
import os
import sys

from config import ConfigError, active_models, load_config
from report import PER_FILE_FIELDS, SUMMARY_FIELDS, per_file, read_jsonl, summarize, write_csv
from runner import build_holes, load_samples, rebuild_holes, run, write_samples


def parse_args(argv):
    parser = argparse.ArgumentParser(description="FIM / code-completion evaluation harness")
    parser.add_argument("-path", "--path", required=True)
    parser.add_argument("-samples", "--samples", type=int, default=100)
    parser.add_argument("-config", "--config", required=True)
    parser.add_argument("-seed", "--seed", type=int, default=854)
    parser.add_argument("-jsonl", "--jsonl", default="results.jsonl")
    parser.add_argument("-csv", "--csv", default="summary.csv")
    parser.add_argument("-per-file-csv", "--per-file-csv", default="per_file.csv")
    parser.add_argument("-workers", "--workers", type=int, default=4)
    parser.add_argument("-timeout", "--timeout", type=float, default=60)
    parser.add_argument("-span-lines", "--span-lines", type=int, nargs=2, default=[1, 20],
                        metavar=("MIN", "MAX"))
    parser.add_argument("-cut", "--cut", choices=["mixed", "lines", "block"], default="mixed")
    parser.add_argument("-sampling", "--sampling", choices=["balanced", "random"], default="balanced")
    parser.add_argument("-min-file-lines", "--min-file-lines", type=int, default=30)
    parser.add_argument("-languages", "--languages", default="c,go,python,javascript")
    parser.add_argument("-write-samples", "--write-samples", default=None)
    parser.add_argument("-samples-file", "--samples-file", default=None)
    parser.add_argument("-no-stream", "--no-stream", action="store_true")
    parser.add_argument("-fresh", "--fresh", action="store_true",
                        help="overwrite results and summaries instead of appending to previous runs")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    languages = [l.strip().lower() for l in args.languages.split(",") if l.strip()]
    if args.span_lines[0] > args.span_lines[1]:
        print("span-lines MIN must be <= MAX", file=sys.stderr)
        return 2
    try:
        all_models = load_config(args.config)
    except ConfigError as e:
        print(f"config error: {e}", file=sys.stderr)
        return 2
    models = active_models(all_models)
    for model in all_models:
        if not model.is_active():
            print(f"deactivated, skipped: {model.name}")
    if not models:
        print("no active models in config", file=sys.stderr)
        return 2
    for model in models:
        if model.timeout is None:
            model.timeout = args.timeout

    if args.samples_file:
        samples = load_samples(args.samples_file)
        holes = rebuild_holes(samples)
    else:
        holes = build_holes(
            args.path, languages, args.samples, args.seed,
            args.min_file_lines, args.span_lines[0], args.span_lines[1], args.cut,
            args.sampling,
        )
        if args.write_samples:
            write_samples(args.write_samples, holes)
    if not holes:
        print("no usable holes found", file=sys.stderr)
        return 1
    previous = []
    if not args.fresh and os.path.exists(args.jsonl):
        previous = read_jsonl(args.jsonl)
        print(f"appending to {args.jsonl} ({len(previous)} rows from previous runs, -fresh to start over)")
    rows = asyncio.run(run(args, models, holes, args.jsonl, append=not args.fresh))
    all_rows = previous + rows
    write_csv(args.csv, summarize(all_rows), SUMMARY_FIELDS)
    write_csv(args.per_file_csv, per_file(all_rows), PER_FILE_FIELDS)
    print(f"wrote {len(rows)} new rows to {args.jsonl} ({len(all_rows)} rows total)")
    print(f"summaries over all {len(all_rows)} rows -> {args.csv}, {args.per_file_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
