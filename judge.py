import argparse
import asyncio
import json
import os
import re
import sys
from collections import defaultdict

from config import ConfigError, load_config
from providers import Provider, ProviderError
from report import SUMMARY_FIELDS, read_jsonl, summarize, write_csv

JUDGE_SYSTEM = (
    "You are an expert code reviewer grading a code completion against the original "
    "missing code. Reply with ONLY one JSON object: {\"score\": <integer 0 to 10>, "
    "\"notes\": \"<one short sentence saying what is right or wrong>\"}. "
    "Score 10 means identical or functionally equivalent, 0 means unrelated or empty."
)

SCORE_RE = re.compile(r'"score"\s*:\s*(\d+)')
NOTES_RE = re.compile(r'"notes"\s*:\s*"([^"]*)"')

MAX_TEXT = 3000


def judge_prompt(row):
    removed = row["removed_text"][:MAX_TEXT]
    completion = (row["completion_norm"] or "")[:MAX_TEXT]
    if not completion.strip():
        completion = "(empty)"
    user = (
        f"Language: {row['language']}\n"
        f"File: {row['file']} (lines {row['start_line']}-{row['end_line']})\n\n"
        f"REFERENCE (original removed code):\n{removed}\n\n"
        f"CANDIDATE (model completion):\n{completion}"
    )
    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": user},
    ]


def parse_score(text):
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        data = json.loads(cleaned)
        score = data.get("score")
        notes = data.get("notes")
        if isinstance(score, (int, float)):
            return min(10, max(0, int(score))), str(notes or "")
    except Exception:
        pass
    score_match = SCORE_RE.search(text)
    notes_match = NOTES_RE.search(text)
    if score_match:
        score = min(10, max(0, int(score_match.group(1))))
        notes = notes_match.group(1) if notes_match else ""
        return score, notes
    return None, None


def row_key(row):
    return (row.get("run_id"), row["model"], row["mode"], row["file"],
            row["start_line"], row["end_line"])


async def judge_rows(rows, provider, workers):
    semaphore = asyncio.Semaphore(workers)

    async def judge_one(row):
        async with semaphore:
            try:
                answer = await provider.raw_chat(judge_prompt(row))
                score, notes = parse_score(answer)
                row["judge_score"] = score
                row["judge_notes"] = notes if score is not None else f"unparsed: {answer[:200]}"
            except ProviderError as e:
                row["judge_notes"] = f"judge error: {e.kind}"
        return row

    tasks = [asyncio.create_task(judge_one(r)) for r in rows]
    out = []
    done = 0
    for coro in asyncio.as_completed(tasks):
        out.append(await coro)
        done += 1
        print(f"[judge {done}/{len(rows)}]")
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description="LLM judge: score FIM completions 0-10")
    parser.add_argument("-results", "--results", required=True)
    parser.add_argument("-config", "--config", required=True)
    parser.add_argument("-workers", "--workers", type=int, default=4)
    parser.add_argument("-out", "--out", default=None)
    parser.add_argument("-csv", "--csv", default=None)
    parser.add_argument("-limit", "--limit", type=int, default=None)
    parser.add_argument("-reuse", "--reuse", action="store_true",
                        help="carry over judge scores from the previous output file and only judge new rows")
    args = parser.parse_args(argv)
    try:
        judges = load_config(args.config)
    except ConfigError as e:
        print(f"config error: {e}", file=sys.stderr)
        return 2
    if len(judges) != 1:
        print("judge config must contain exactly one model", file=sys.stderr)
        return 2
    rows = read_jsonl(args.results)
    for row in rows:
        row.setdefault("judge_score", None)
        row.setdefault("judge_notes", None)
    out_path = args.out or args.results.replace(".jsonl", "_judged.jsonl")
    reused = 0
    if args.reuse and os.path.exists(out_path):
        previous = {}
        for old in read_jsonl(out_path):
            if old.get("judge_score") is not None:
                previous[row_key(old)] = (old["judge_score"], old.get("judge_notes"))
        for row in rows:
            key = row_key(row)
            if row["status"] == "ok" and row["judge_score"] is None and key in previous:
                row["judge_score"], row["judge_notes"] = previous[key]
                reused += 1
    targets = [r for r in rows if r["status"] == "ok" and r["judge_score"] is None]
    if args.limit:
        targets = targets[: args.limit]

    async def run_judge():
        provider = Provider(judges[0])
        try:
            return await judge_rows(targets, provider, args.workers)
        finally:
            await provider.close()

    judged = asyncio.run(run_judge())
    by_index = {id(r): r for r in judged}
    for row in rows:
        update = by_index.get(id(row))
        if update is not None:
            row["judge_score"] = update["judge_score"]
            row["judge_notes"] = update["judge_notes"]
    with open(out_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    csv_path = args.csv or "summary_judged.csv"
    write_csv(csv_path, summarize(rows), SUMMARY_FIELDS)
    print(f"judged {len(targets)} new rows, reused {reused} -> {out_path}")
    print(f"summary over all {len(rows)} rows -> {csv_path}")
    scores = defaultdict(list)
    for r in rows:
        if r["status"] == "ok" and r["judge_score"] is not None:
            scores[(r["model"], r["mode"])].append(r["judge_score"])
    for key in sorted(scores):
        values = scores[key]
        print(f"  {key[0]:26s} {key[1]:5s} avg {sum(values)/len(values):.1f}/10 ({len(values)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
