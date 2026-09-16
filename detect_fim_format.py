import argparse
import asyncio
import copy
import json
import os
import shutil
import sys

from config import ConfigError, active_models, load_config
from fim_formats import FORMATS, LEAK_TOKENS, applicable_formats, format_names
from judge import judge_rows
from metrics import check_syntax, normalize, postprocess, similarity
from providers import Provider, ProviderError
from report import write_csv, write_jsonl
from runner import build_holes

SUMMARY_FIELDS = [
    "model", "format", "requests", "avg_similarity_char", "avg_similarity_token",
    "syntax_valid_rate", "leak_rate", "composite_score",
]

JUDGE_MARGIN = 0.10


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Detect the best FIM prompt format per model")
    parser.add_argument("-config", "--config", required=True)
    parser.add_argument("-path", "--path", default="testcorpus")
    parser.add_argument("-languages", "--languages", default="c,go,python,javascript")
    parser.add_argument("-samples-per-format", "--samples-per-format", type=int, default=3)
    parser.add_argument("-workers", "--workers", type=int, default=4)
    parser.add_argument("-seed", "--seed", type=int, default=42)
    parser.add_argument("-timeout", "--timeout", type=float, default=60)
    parser.add_argument("-max-tokens", "--max-tokens", type=int, default=None)
    parser.add_argument("-formats", "--formats", default=None)
    parser.add_argument("-models", "--models", default=None)
    parser.add_argument("-out-jsonl", "--out-jsonl", default="fim_format_results.jsonl")
    parser.add_argument("-out-csv", "--out-csv", default="fim_format_summary.csv")
    parser.add_argument("-judge", "--judge", action="store_true")
    parser.add_argument("-judge-config", "--judge-config", default="judge.config.json")
    parser.add_argument("-apply", "--apply", action="store_true")
    return parser.parse_args(argv)


def split_csv(value):
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def mean(values):
    return sum(values) / len(values) if values else None


def show(value, digits=3):
    return "-" if value is None else f"{value:.{digits}f}"


def base_row(model, fmt, hole):
    return {
        "model": model.name,
        "format": fmt["name"],
        "protocol": fmt["protocol"],
        "file": hole.file,
        "language": hole.language,
        "start_line": hole.start_line,
        "end_line": hole.end_line,
        "removed_text": hole.removed_text,
        "completion_raw": None,
        "completion_norm": None,
        "similarity_char": None,
        "similarity_token": None,
        "syntax_valid_fragment": None,
        "syntax_valid_merged": None,
        "leaked": None,
        "composite": 0.0,
        "status": "ok",
        "error": None,
    }


def row_composite(row, token_sim, merged):
    if not (row["completion_norm"] or "").strip():
        return 0.0
    return 0.5 * token_sim + 0.3 * (1.0 if merged else 0.0) + 0.2 * (0.0 if row["leaked"] else 1.0)


async def probe_one(provider, model, fmt, hole):
    row = base_row(model, fmt, hole)
    try:
        result = await provider.fim(hole.prefix, hole.suffix, False)
    except ProviderError as e:
        row["status"] = "error"
        row["error"] = f"{e.kind}: {e.message}"
        return row
    except Exception as e:
        row["status"] = "error"
        row["error"] = f"unexpected {type(e).__name__}: {e}"
        return row
    raw = result["text"]
    row["completion_raw"] = raw
    normalized = postprocess(raw, hole.language, hole.indent)
    row["completion_norm"] = normalized
    char_sim, token_sim = similarity(normalized, normalize(hole.removed_text))
    row["similarity_char"] = char_sim
    row["similarity_token"] = token_sim
    fragment, merged = check_syntax(hole.language, normalized, hole.prefix, hole.suffix)
    row["syntax_valid_fragment"] = fragment
    row["syntax_valid_merged"] = merged
    row["leaked"] = any(token in raw for token in LEAK_TOKENS)
    row["composite"] = row_composite(row, token_sim, merged)
    return row


async def probe_all(models, formats_by_model, holes, workers, jsonl_path):
    model_sems = {model.name: asyncio.Semaphore(workers) for model in models}
    device_sems = {}
    for model in models:
        if model.device_type and model.device_type not in device_sems:
            device_sems[model.device_type] = asyncio.Semaphore(1)
    pairs = [(model, fmt) for model in models for fmt in formats_by_model[model.name]]
    total = len(pairs) * len(holes)
    progress = {"done": 0}
    rows = []

    with open(jsonl_path, "w", encoding="utf-8") as handle:

        async def run_pair(model, fmt):
            cfg = copy.copy(model)
            cfg.fim_protocol = fmt["protocol"]
            cfg.fim_template = fmt["template"]
            provider = Provider(cfg)
            device = device_sems.get(model.device_type)

            async def run_hole(hole):
                async with model_sems[model.name]:
                    if device is None:
                        return await probe_one(provider, model, fmt, hole)
                    async with device:
                        return await probe_one(provider, model, fmt, hole)

            try:
                pending = [asyncio.create_task(run_hole(hole)) for hole in holes]
                for coro in asyncio.as_completed(pending):
                    row = await coro
                    progress["done"] += 1
                    print(f"[probe {progress['done']}/{total}] {row['model']} {row['format']} "
                          f"{os.path.basename(row['file'])} -> {row['status']}")
                    rows.append(row)
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    handle.flush()
            finally:
                await provider.close()

        tasks = [asyncio.create_task(run_pair(model, fmt)) for model, fmt in pairs]
        try:
            await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                task.cancel()
    return rows


def summarize(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row["model"], row["format"]), []).append(row)
    out = []
    for (model, format_name), group in sorted(groups.items()):
        ok = [row for row in group if row["status"] == "ok"]
        out.append({
            "model": model,
            "format": format_name,
            "requests": len(group),
            "ok": len(ok),
            "avg_similarity_char": mean([row["similarity_char"] for row in ok if row["similarity_char"] is not None]),
            "avg_similarity_token": mean([row["similarity_token"] for row in ok if row["similarity_token"] is not None]),
            "syntax_valid_rate": mean([1.0 if row["syntax_valid_merged"] else 0.0 for row in ok]),
            "leak_rate": mean([1.0 if row["leaked"] else 0.0 for row in ok]),
            "composite_score": mean([row["composite"] for row in group]),
        })
    return out


def contender_keys(summaries):
    by_model = {}
    for summary in summaries:
        by_model.setdefault(summary["model"], []).append(summary)
    chosen = set()
    for model, group in by_model.items():
        top = max(summary["composite_score"] for summary in group)
        for summary in group:
            if top - summary["composite_score"] <= JUDGE_MARGIN:
                chosen.add((model, summary["format"]))
    return chosen


def judge_targets(rows, summaries):
    chosen = contender_keys(summaries)
    return [row for row in rows if row["status"] == "ok" and (row["model"], row["format"]) in chosen]


def judged_value(row):
    score = row.get("judge_score")
    if score is None:
        return None
    return (0.6 * (score / 10)
            + 0.2 * (1.0 if row["syntax_valid_merged"] else 0.0)
            + 0.2 * (0.0 if row["leaked"] else 1.0))


def update_judged_composites(summaries, targets):
    values = {}
    for row in targets:
        value = judged_value(row)
        if value is not None:
            values.setdefault((row["model"], row["format"]), []).append(value)
    for summary in summaries:
        scores = values.get((summary["model"], summary["format"]))
        if scores:
            summary["composite_score"] = mean(scores)
            summary["judged"] = True


def run_judge(args, rows, summaries):
    try:
        judges = load_config(args.judge_config)
    except ConfigError as e:
        print(f"judge config error: {e}", file=sys.stderr)
        return 2
    if len(judges) != 1:
        print("judge config must contain exactly one model", file=sys.stderr)
        return 2
    targets = judge_targets(rows, summaries)
    if not targets:
        print("nothing to judge")
        return 0

    async def go():
        provider = Provider(judges[0])
        try:
            await judge_rows(targets, provider, args.workers)
        finally:
            await provider.close()

    asyncio.run(go())
    update_judged_composites(summaries, targets)
    return 0


def ranked_summaries(group):
    judged = [summary for summary in group if summary.get("judged")]
    rest = [summary for summary in group if not summary.get("judged")]
    key = lambda summary: summary["composite_score"]
    return sorted(judged, key=key, reverse=True) + sorted(rest, key=key, reverse=True)


def print_report(models, summaries):
    by_model = {}
    for summary in summaries:
        by_model.setdefault(summary["model"], []).append(summary)
    for model in models:
        group = by_model.get(model.name)
        if not group:
            continue
        ranked = ranked_summaries(group)
        print(f"\n{model.name}")
        for summary in ranked:
            mark = "*" if summary is ranked[0] else " "
            print(f" {mark} {summary['format']:20s} composite {show(summary['composite_score'])}  "
                  f"token {show(summary['avg_similarity_token'])}  syntax {show(summary['syntax_valid_rate'])}  "
                  f"leak {show(summary['leak_rate'])}  ({summary['requests']} req)")
        winner = ranked[0]
        fmt = next(fmt for fmt in FORMATS if fmt["name"] == winner["format"])
        snippet = {"fim_protocol": fmt["protocol"]}
        if fmt["template"]:
            snippet["fim_template"] = fmt["template"]
        if fmt["name"] == "llamacpp_infill" and model.fim_endpoint:
            snippet["fim_endpoint"] = model.fim_endpoint
        print(f"   {json.dumps(snippet)}")


def current_format(entry):
    protocol = entry.get("fim_protocol")
    if protocol is None:
        return None
    template = entry.get("fim_template")
    for fmt in FORMATS:
        if fmt["protocol"] == protocol and (fmt["template"] or None) == (template or None):
            return fmt
    return None


def apply_winners(config_path, all_models, selected_names, summaries):
    with open(config_path, encoding="utf-8") as handle:
        entries = json.load(handle)
    by_model = {}
    for summary in summaries:
        by_model.setdefault(summary["model"], []).append(summary)
    picks = []
    for entry, model in zip(entries, all_models):
        if not model.is_active():
            print(f"{model.name}: deactivated, skipped")
            continue
        if model.name not in selected_names:
            continue
        group = by_model.get(model.name)
        if not group:
            print(f"{model.name}: not probed, skipped")
            continue
        winner = ranked_summaries(group)[0]
        if winner["ok"] == 0:
            print(f"{model.name}: {winner['format']} had no successful requests, skipped")
            continue
        fmt = next(fmt for fmt in FORMATS if fmt["name"] == winner["format"])
        current = current_format(entry)
        if current is not None and current["name"] == winner["format"]:
            print(f"{model.name}: already {winner['format']}, unchanged")
            continue
        if not entry.get("fim_endpoint"):
            print(f"{model.name}: no fim_endpoint, cannot apply {winner['format']}, skipped")
            continue
        if current is None:
            print(f"{model.name}: no matching current format, applying {winner['format']}")
        else:
            current_summary = next((s for s in group if s["format"] == current["name"]), None)
            if current_summary is None:
                print(f"{model.name}: current format {current['name']} not probed, applying {winner['format']}")
            elif (winner["composite_score"] < current_summary["composite_score"]
                  and bool(winner.get("judged")) == bool(current_summary.get("judged"))):
                print(f"{model.name}: current {current['name']} scored higher, kept")
                continue
        picks.append((entry, model, fmt))
    if not picks:
        print("nothing to apply")
        return
    shutil.copy(config_path, config_path + ".bak")
    for entry, model, fmt in picks:
        old = entry.get("fim_protocol")
        entry["fim_protocol"] = fmt["protocol"]
        if fmt["template"]:
            entry["fim_template"] = fmt["template"]
        else:
            entry.pop("fim_template", None)
        print(f"{model.name}: {old} -> {fmt['protocol']}")
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(entries, handle, indent=2)
        handle.write("\n")
    print(f"wrote {config_path} ({len(picks)} changes), backup {config_path}.bak")


def main(argv=None):
    args = parse_args(argv)
    try:
        all_models = load_config(args.config)
    except ConfigError as e:
        print(f"config error: {e}", file=sys.stderr)
        return 2
    selected_models = split_csv(args.models)
    unknown = [name for name in selected_models if name not in {model.name for model in all_models}]
    if unknown:
        print(f"unknown models: {', '.join(unknown)}", file=sys.stderr)
        return 2
    selected_formats = split_csv(args.formats)
    unknown = [name for name in selected_formats if name not in format_names()]
    if unknown:
        print(f"unknown formats: {', '.join(unknown)} (valid: {', '.join(format_names())})", file=sys.stderr)
        return 2
    models = [model for model in all_models
              if model.is_active() and (not selected_models or model.name in selected_models)]
    for model in all_models:
        if not model.is_active():
            print(f"deactivated, skipped: {model.name}")
    if not models:
        print("no active models to probe", file=sys.stderr)
        return 2
    languages = split_csv(args.languages)
    for model in models:
        if model.timeout is None:
            model.timeout = args.timeout
    if args.max_tokens:
        for model in models:
            model.max_tokens = args.max_tokens
    holes = build_holes(args.path, languages, args.samples_per_format * len(languages),
                        args.seed, 30, 1, 20, "mixed", "balanced")
    if not holes:
        print("no usable holes found", file=sys.stderr)
        return 1
    formats_by_model = {model.name: applicable_formats(model, selected_formats) for model in models}
    if not any(formats_by_model[model.name] for model in models):
        print("no formats to probe", file=sys.stderr)
        return 1
    pairs = sum(len(formats_by_model[model.name]) for model in models)
    print(f"probing {pairs} model/format pairs x {len(holes)} holes = {pairs * len(holes)} requests")
    rows = asyncio.run(probe_all(models, formats_by_model, holes, args.workers, args.out_jsonl))
    summaries = summarize(rows)
    if args.judge:
        code = run_judge(args, rows, summaries)
        if code:
            return code
    write_jsonl(args.out_jsonl, rows)
    write_csv(args.out_csv, summaries, SUMMARY_FIELDS)
    print(f"wrote {len(rows)} rows -> {args.out_jsonl}, {len(summaries)} summaries -> {args.out_csv}")
    print_report(models, summaries)
    if args.apply:
        apply_winners(args.config, all_models, {model.name for model in models}, summaries)
    return 0


if __name__ == "__main__":
    sys.exit(main())
