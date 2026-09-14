import argparse
import asyncio
import sys

from config import ConfigError, load_config
from report import per_file, summarize, write_csv, write_jsonl
from runner import build_holes, load_samples, rebuild_holes, run, write_samples


def parse_args(argv):
    parser = argparse.ArgumentParser(description="FIM / code-completion evaluation harness")
    parser.add_argument("-path", "--path", required=True)
    parser.add_argument("-samples", "--samples", type=int, default=100)
    parser.add_argument("-config", "--config", required=True)
    parser.add_argument("-seed", "--seed", type=int, default=42)
    parser.add_argument("-jsonl", "--jsonl", default="results.jsonl")
    parser.add_argument("-csv", "--csv", default="summary.csv")
    parser.add_argument("-per-file-csv", "--per-file-csv", default="per_file.csv")
    parser.add_argument("-workers", "--workers", type=int, default=4)
    parser.add_argument("-timeout", "--timeout", type=float, default=60)
    parser.add_argument("-span-lines", "--span-lines", type=int, nargs=2, default=[1, 20],
                        metavar=("MIN", "MAX"))
    parser.add_argument("-cut", "--cut", choices=["mixed", "lines", "block"], default="mixed")
    parser.add_argument("-min-file-lines", "--min-file-lines", type=int, default=30)
    parser.add_argument("-languages", "--languages", default="c,go,python,javascript")
    parser.add_argument("-write-samples", "--write-samples", default=None)
    parser.add_argument("-samples-file", "--samples-file", default=None)
    parser.add_argument("-no-stream", "--no-stream", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    languages = [l.strip().lower() for l in args.languages.split(",") if l.strip()]
    try:
        models = load_config(args.config)
    except ConfigError as e:
        print(f"config error: {e}", file=sys.stderr)
        return 2
    if args.samples_file:
        samples = load_samples(args.samples_file)
        holes = rebuild_holes(samples)
    else:
        holes = build_holes(
            args.path, languages, args.samples, args.seed,
            args.min_file_lines, args.span_lines[0], args.span_lines[1], args.cut,
        )
        if args.write_samples:
            write_samples(args.write_samples, holes)
    if not holes:
        print("no usable holes found", file=sys.stderr)
        return 1
    rows = asyncio.run(run(args, models, holes, args.jsonl))
    write_jsonl(args.jsonl, rows)
    write_csv(args.csv, summarize(rows), ["model", "mode", "samples", "ok", "unsupported", "errors",
                                          "error_rate", "exact_match", "exact_match_norm",
                                          "avg_similarity_char", "avg_similarity_token",
                                          "syntax_valid", "avg_latency_ms", "median_latency_ms",
                                          "p95_latency_ms", "avg_ttft_ms", "total_prompt_tokens",
                                          "total_completion_tokens", "total_cost_usd"])
    write_csv(args.per_file_csv, per_file(rows), ["model", "mode", "file", "language", "samples",
                                                  "exact_match", "avg_similarity_char",
                                                  "syntax_valid", "errors", "cost_usd"])
    print(f"wrote {len(rows)} rows to {args.jsonl}, {args.csv}, {args.per_file_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
