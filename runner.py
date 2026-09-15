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


def build_holes(root, languages, samples, seed, min_file_lines, span_min, span_max, cut, sampling="balanced"):
    rng = random.Random(seed)
    files = collect_files(root, languages)
    by_lang = {}
    for path, language in files.items():
        by_lang.setdefault(language, []).append(path)
    present = [l for l in languages if by_lang.get(l)]
    if not present:
        return []
    if sampling == "random":
        quotas = None
    else:
        base = samples // len(present)
        extra = samples - base * len(present)
        shuffled = present[:]
        rng.shuffle(shuffled)
        quotas = {l: base for l in present}
        for i in range(extra):
            quotas[shuffled[i % len(shuffled)]] += 1
    holes = []
    if sampling == "random":
        all_paths = []
        for language in present:
            all_paths.extend(by_lang[language])
        rng.shuffle(all_paths)
        for i in range(samples):
            path = all_paths[i % len(all_paths)]
            language = files[path]
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
        "judge_score": None,
        "judge_notes": None,
    }


async def run(args, models, holes, jsonl_path, append=False):
    run_id = uuid.uuid4().hex[:8]
    providers = {m.name: Provider(m) for m in models}
    semaphores = {m.name: asyncio.Semaphore(args.workers) for m in models}
    devices = {}
    for model in models:
        if model.device_type and model.device_type not in devices:
            devices[model.device_type] = asyncio.Semaphore(1)
    rows = []
    total = sum(len(m.modes) for m in models) * len(holes)
    done = 0

    async def job(model, mode, hole):
        async with semaphores[model.name]:
            device = devices.get(model.device_type)
            if device is None:
                return await execute(model, mode, hole)
            async with device:
                return await execute(model, mode, hole)

    async def execute(model, mode, hole):
        provider = providers[model.name]
        budget = None
        if model.max_context_size is not None:
            budget = max(1, model.max_context_size - model.max_tokens)
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
            row["attempts"] = e.attempts
            row["latency_ms"] = (time.perf_counter() - start) * 1000
            return row
        except Exception as e:
            row["status"] = "error"
            row["error"] = f"unexpected {type(e).__name__}: {e}"
            row["attempts"] = 1
            row["latency_ms"] = (time.perf_counter() - start) * 1000
            return row
        row["latency_ms"] = (time.perf_counter() - start) * 1000
        row["ttft_ms"] = result.get("ttft_ms")
        row["prompt_tokens"] = result.get("prompt_tokens")
        row["completion_tokens"] = result.get("completion_tokens")
        row["attempts"] = result.get("attempts", 1)
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
            model.pricing, result.get("live_cost"),
        )
        return row

    tasks = []
    for model in models:
        for mode in model.modes:
            for hole in holes:
                tasks.append(asyncio.create_task(job(model, mode, hole)))
    try:
        with open(jsonl_path, "a" if append else "w", encoding="utf-8") as f:
            for coro in asyncio.as_completed(tasks):
                row = await coro
                rows.append(row)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                done += 1
                print(f"[{done}/{total}] {row['model']} {row['mode']} {row['file']} -> {row['status']}")
    finally:
        for task in tasks:
            task.cancel()
        for provider in providers.values():
            try:
                await provider.close()
            except Exception:
                pass
    return rows
