import csv
import json
import statistics
import sys

JSONL_FIELDS = [
    "run_id", "timestamp", "seed", "model", "mode", "file", "language",
    "start_line", "end_line", "cut", "removed_text", "completion_raw",
    "completion_norm", "prompt_tokens", "completion_tokens", "context_truncated",
    "latency_ms", "ttft_ms", "cost_usd", "exact_match_raw", "exact_match_norm",
    "similarity_char", "similarity_token", "syntax_valid_fragment",
    "syntax_valid_merged", "status", "error", "attempts",
    "judge_score", "judge_notes",
]

SUMMARY_FIELDS = [
    "model", "mode", "samples", "ok", "unsupported", "errors", "error_rate",
    "exact_match", "exact_match_norm", "avg_similarity_char", "avg_similarity_token",
    "syntax_valid", "avg_latency_ms", "median_latency_ms", "p95_latency_ms",
    "avg_ttft_ms", "total_prompt_tokens", "total_completion_tokens", "total_cost_usd",
    "avg_judge_score",
]

PER_FILE_FIELDS = [
    "model", "mode", "file", "language", "samples", "exact_match",
    "avg_similarity_char", "syntax_valid", "errors", "cost_usd",
]


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for number, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"warning: skipping corrupt line {number} in {path}", file=sys.stderr)
    return rows


def _mean(values):
    return sum(values) / len(values) if values else None


def _p95(values):
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(0.95 * len(ordered)) - 1))
    return ordered[index]


def summarize(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row["model"], row["mode"]), []).append(row)
    out = []
    for (model, mode), group in sorted(groups.items()):
        ok = [r for r in group if r["status"] == "ok"]
        unsupported = [r for r in group if r["status"] == "unsupported"]
        errors = [r for r in group if r["status"] == "error"]
        latencies = [r["latency_ms"] for r in ok if r["latency_ms"] is not None]
        ttfts = [r["ttft_ms"] for r in ok if r["ttft_ms"] is not None]
        costs = [r["cost_usd"] for r in ok if r["cost_usd"] is not None]
        judge = [r["judge_score"] for r in ok if r.get("judge_score") is not None]
        out.append({
            "model": model,
            "mode": mode,
            "samples": len(group),
            "ok": len(ok),
            "unsupported": len(unsupported),
            "errors": len(errors),
            "error_rate": len(errors) / len(group) if group else 0.0,
            "exact_match": _mean([r["exact_match_raw"] for r in ok]),
            "exact_match_norm": _mean([r["exact_match_norm"] for r in ok]),
            "avg_similarity_char": _mean([r["similarity_char"] for r in ok]),
            "avg_similarity_token": _mean([r["similarity_token"] for r in ok]),
            "syntax_valid": _mean([r["syntax_valid_merged"] for r in ok]),
            "avg_latency_ms": _mean(latencies),
            "median_latency_ms": statistics.median(latencies) if latencies else None,
            "p95_latency_ms": _p95(latencies),
            "avg_ttft_ms": _mean(ttfts),
            "total_prompt_tokens": sum(r["prompt_tokens"] or 0 for r in ok),
            "total_completion_tokens": sum(r["completion_tokens"] or 0 for r in ok),
            "total_cost_usd": sum(costs) if costs else None,
            "avg_judge_score": _mean(judge) if judge else None,
        })
    return out


def per_file(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row["model"], row["mode"], row["file"]), []).append(row)
    out = []
    for (model, mode, file), group in sorted(groups.items()):
        ok = [r for r in group if r["status"] == "ok"]
        errors = [r for r in group if r["status"] == "error"]
        costs = [r["cost_usd"] for r in ok if r["cost_usd"] is not None]
        out.append({
            "model": model,
            "mode": mode,
            "file": file,
            "language": group[0]["language"],
            "samples": len(group),
            "exact_match": _mean([r["exact_match_raw"] for r in ok]),
            "avg_similarity_char": _mean([r["similarity_char"] for r in ok]),
            "syntax_valid": _mean([r["syntax_valid_merged"] for r in ok]),
            "errors": len(errors),
            "cost_usd": sum(costs) if costs else None,
        })
    return out


def write_csv(path, rows, fields):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
