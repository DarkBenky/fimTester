import argparse
import asyncio
import os
import sys

from config import ConfigError, load_config
from judge import judge_rows, row_key
from providers import Provider
from report import (PER_FILE_FIELDS, SUMMARY_FIELDS, per_file, read_jsonl,
                    summarize, write_csv, write_jsonl)


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Rebuild the summary CSVs from a results JSONL, judging unscored rows first with -judge")
    parser.add_argument("-jsonl", "--jsonl", default="results.jsonl")
    parser.add_argument("-csv", "--csv", default="summary.csv")
    parser.add_argument("-per-file-csv", "--per-file-csv", default="per_file.csv")
    parser.add_argument("-judge", "--judge", action="store_true",
                        help="score unscored ok rows first, reusing scores already in the judged file")
    parser.add_argument("-judge-config", "--judge-config", default="judge.config.json")
    parser.add_argument("-judged-jsonl", "--judged-jsonl", default=None)
    parser.add_argument("-judged-csv", "--judged-csv", default="summary_judged.csv")
    parser.add_argument("-workers", "--workers", type=int, default=4)
    return parser.parse_args(argv)


def carry_over_scores(rows, judged_path):
    if not os.path.exists(judged_path):
        return 0
    previous = {}
    for old in read_jsonl(judged_path):
        if old.get("judge_score") is not None:
            previous[row_key(old)] = (old["judge_score"], old.get("judge_notes"))
    reused = 0
    for row in rows:
        if row["status"] == "ok" and row["judge_score"] is None:
            score = previous.get(row_key(row))
            if score is not None:
                row["judge_score"], row["judge_notes"] = score
                reused += 1
    return reused


def judge_unscored(rows, config_path, workers):
    judges = load_config(config_path)
    if len(judges) != 1:
        raise ConfigError("judge config must contain exactly one model")
    targets = [r for r in rows if r["status"] == "ok" and r["judge_score"] is None]
    if not targets:
        return 0

    async def go():
        provider = Provider(judges[0])
        try:
            return await judge_rows(targets, provider, workers)
        finally:
            await provider.close()

    asyncio.run(go())
    return len(targets)


def main(argv=None):
    args = parse_args(argv)
    if not os.path.exists(args.jsonl):
        print(f"jsonl not found: {args.jsonl}", file=sys.stderr)
        return 2
    rows = read_jsonl(args.jsonl)
    if not rows:
        print(f"no rows in {args.jsonl}", file=sys.stderr)
        return 1
    for row in rows:
        row.setdefault("judge_score", None)
        row.setdefault("judge_notes", None)
    outputs = [args.csv, args.per_file_csv]
    if args.judge:
        judged_path = args.judged_jsonl or args.jsonl.replace(".jsonl", "_judged.jsonl")
        reused = carry_over_scores(rows, judged_path)
        try:
            judged = judge_unscored(rows, args.judge_config, args.workers)
        except ConfigError as e:
            print(f"judge config error: {e}", file=sys.stderr)
            return 2
        write_jsonl(judged_path, rows)
        outputs.append(args.judged_csv)
        print(f"judged {judged} new rows, reused {reused} -> {judged_path}")
    write_csv(args.csv, summarize(rows), SUMMARY_FIELDS)
    write_csv(args.per_file_csv, per_file(rows), PER_FILE_FIELDS)
    if args.judge:
        write_csv(args.judged_csv, summarize(rows), SUMMARY_FIELDS)
    runs = sorted({str(row.get("run_id")) for row in rows})
    models = sorted({row["model"] for row in rows})
    print(f"read {len(rows)} rows ({len(runs)} runs, {len(models)} models) from {args.jsonl}")
    print(f"wrote {', '.join(outputs)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
