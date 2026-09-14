import asyncio
import json
import random
import time
import uuid
from datetime import datetime, timezone

from config import load_config
from files import collect_files, read_source
from metrics import check_syntax, exact_match, normalize, postprocess, similarity
from pricing import cost_usd
from providers import Provider, ProviderError
from spans import hole_from_sample, make_hole, trim_context

HOLE_ATTEMPTS = 50


def build_holes(root, languages, samples, seed, min_file_lines, span_min, span_max, cut):
    rng = random.Random(seed)
    files = collect_files(root, languages)
    by_lang = {}
    for path, language in files.items():
        by_lang.setdefault(language, []).append(path)
    present = [l for l in languages if by_lang.get(l)]
    if not present:
        return []
    base = samples // len(present)
    extra = samples - base * len(present)
    shuffled = present[:]
    rng.shuffle(shuffled)
    quotas = {l: base for l in present}
    for i in range(extra):
        quotas[shuffled[i % len(shuffled)]] += 1
    holes = []
    for language in present:
        paths = by_lang[language][:]
        rng.shuffle(paths)
        for i in range(quotas[language]):
            path = paths[i % len(paths)]
            text = read_source(path)
            if text is None:
                continue
            hole = None
            for _ in range(HOLE_ATTEMPTS):
                hole = make_hole(rng, language, text, min_file_lines, span_min, span_max, cut)
                if hole is not None:
                    break
            if hole is None:
                continue
            hole.file = path
            hole.language = language
            holes.append(hole)
    return holes


def load_samples(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        raise ValueError(f"samples file must be a non-empty JSON array: {path}")
    return data


def write_samples(path, holes):
    with open(path, "w", encoding="utf-8") as f:
        json.dump([h.to_sample() for h in holes], f, ensure_ascii=False, indent=2)


def rebuild_holes(samples):
    holes = []
    for sample in samples:
        text = read_source(sample["file"])
        if text is None:
            continue
        holes.append(hole_from_sample(sample, text))
    return holes


def base_row(run_id, seed, model, mode, hole):
    return {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "model": model.name,
        "mode": mode,
        "file": hole.file,
        "language": hole.language,
        "start_line": hole.start_line,
        "end_line": hole.end_line,
        "cut": hole.cut,
        "removed_text": hole.removed_text,
        "completion_raw": None,
        "completion_norm": None,
        "prompt_tokens": None,
        "completion_tokens": None,
        "context_truncated": False,
        "latency_ms": None,
        "ttft_ms": None,
        "cost_usd": None,
        "exact_match_raw": None,
        "exact_match_norm": None,
        "similarity_char": None,
        "similarity_token": None,
        "syntax_valid_fragment": None,
        "syntax_valid_merged": None,
        "status": "ok",
        "error": None,
        "attempts": 1,
    }


async def run(args, models, holes, jsonl_path):
    run_id = uuid.uuid4().hex[:8]
    providers = {m.name: Provider(m) for m in models}
    semaphore = asyncio.Semaphore(args.workers)
    rows = []
    total = len(models) * len(holes) * 2
    done = 0

    async def job(model, mode, hole):
        nonlocal done
        async with semaphore:
            provider = providers[model.name]
            budget = None
            if model.max_context_size is not None:
                budget = model.max_context_size - model.max_tokens
            prefix, suffix, truncated = trim_context(hole.prefix, hole.suffix, budget)
            row = base_row(run_id, args.seed, model, mode, hole)
            row["context_truncated"] = truncated
            if mode == "fim" and not model.has_fim():
                row["status"] = "unsupported"
                return row
            start = time.perf_counter()
            try:
                if mode == "fim":
                    result = await provider.fim(prefix, suffix, not args.no_stream)
                else:
                    result = await provider.chat(prefix, suffix, not args.no_stream)
            except ProviderError as e:
                row["status"] = "error"
                row["error"] = f"{e.kind}: {e.message}"
                row["attempts"] = 3
                row["latency_ms"] = (time.perf_counter() - start) * 1000
                return row
            row["latency_ms"] = (time.perf_counter() - start) * 1000
            row["ttft_ms"] = result["ttft_ms"]
            row["prompt_tokens"] = result["prompt_tokens"]
            row["completion_tokens"] = result["completion_tokens"]
            raw = result["text"]
            row["completion_raw"] = raw
            norm = postprocess(raw, hole.language, hole.indent)
            row["completion_norm"] = norm
            removed_norm = normalize(hole.removed_text)
            row["exact_match_raw"] = exact_match(raw, hole.removed_text)
            row["exact_match_norm"] = exact_match(norm, removed_norm)
            char_sim, token_sim = similarity(norm, removed_norm)
            row["similarity_char"] = char_sim
            row["similarity_token"] = token_sim
            fragment, merged = check_syntax(hole.language, norm, prefix, suffix)
            row["syntax_valid_fragment"] = fragment
            row["syntax_valid_merged"] = merged
            row["cost_usd"] = cost_usd(
                model.model, row["prompt_tokens"], row["completion_tokens"],
                model.pricing, result["live_cost"],
            )
            return row

    tasks = []
    for model in models:
        for mode in ("fim", "chat"):
            for hole in holes:
                tasks.append(asyncio.create_task(job(model, mode, hole)))
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for coro in asyncio.as_completed(tasks):
            row = await coro
            rows.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            done += 1
            print(f"[{done}/{total}] {row['model']} {row['mode']} {row['file']} -> {row['status']}")
    for provider in providers.values():
        await provider.close()
    return rows
