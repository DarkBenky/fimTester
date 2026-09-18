import argparse
import asyncio
import hashlib
import json
import os
import random
import sys
import time
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import ConfigError, load_config
from files import read_source
from judge import parse_score
from metrics import check_syntax, strip_fences
from providers import Provider, ProviderError
from report import read_jsonl
from runner import build_holes
from spans import is_meaningful

DATA_DIR = os.path.join(ROOT, "synteticData", "data")

DOMAINS = [
    "systems",
    "networking",
    "parsing",
    "data_structures",
    "algorithms",
    "cli_tools",
    "concurrency",
    "filesystem",
    "text_processing",
    "error_handling",
    "serialization",
    "math",
]

GEN_SYSTEM = (
    "You create fill-in-the-middle (FIM) training samples for a code completion model. "
    "You design HARD samples: the removed middle must require careful reasoning over the whole "
    "surrounding code, yet be uniquely determined by it. Every sample is machine-checked: "
    "prefix + completion + suffix must parse as one exact source file or it is rejected. "
    "Reply with ONLY one JSON object."
)

GEN_TEMPLATE = (
    "Create one synthetic FIM sample for {language}. Target difficulty: {difficulty}/5. "
    "Domain hint: {domain}.\n\n"
    "Return ONLY this JSON object:\n"
    "{\"language\": \"{language}\", \"domain\": \"...\", \"difficulty\": {difficulty}, "
    "\"task_type\": \"...\", \"hole_type\": \"block|lines|expression\", \"concepts\": [\"...\"], "
    "\"prefix\": \"...\", \"suffix\": \"...\", \"completion\": \"...\"}\n\n"
    "What the three parts mean:\n"
    "- prefix = every line BEFORE the hole (a file beginning; open braces there are fine)\n"
    "- completion = ONLY the removed lines, including every closing brace that belongs to them\n"
    "- suffix = every line AFTER the hole, at least 3 meaningful lines, and it must use what the\n"
    "  completion defines\n"
    "- prefix + completion + suffix must be exactly one valid {language} file\n\n"
    "Worked example (c): the file defines a helper, then the function upper_str, then main.\n"
    "  prefix ends after the helper's closing brace; completion is the whole upper_str function\n"
    "  INCLUDING its final brace; suffix starts on a new line with main() calling upper_str.\n\n"
    "Rules:\n"
    "- never leave completion or suffix empty, tiny, or a mere closing brace\n"
    "- hard but deterministic: uniquely inferable from the context, no guessing games\n"
    "- hard means it needs distant context: identifiers/constants defined far above, and the\n"
    "  suffix must depend on what the completion defines or changes\n"
    "- no meta commentary anywhere: never mention AI, prompts, JSON, datasets, training, or what\n"
    "  the sample is supposed to demonstrate\n"
    "- realistic idiomatic {language}; no TODO/FIXME/placeholders; no markdown fences; escape\n"
    "  newlines inside the JSON strings\n"
    "- prefix under 60 lines, suffix under 20 lines, completion under 30 lines\n"
    "- never repeat or redefine code that already appears in the prefix or suffix\n"
    "- cut only at line boundaries, never inside a line; no stray or missing tokens\n"
    "- domain and task_type in short snake_case (systems, memory_management); concepts: 2-5 short tags\n"
)

JUDGE_SYSTEM = (
    "You verify synthetic fill-in-the-middle (FIM) training samples for a code completion model. "
    "You get the prefix and suffix around a removed middle plus the candidate completion. "
    "Check: (1) prefix + completion + suffix form one coherent, syntactically valid unit, "
    "(2) the completion is the natural, deterministic choice given the context, "
    "(3) it requires real understanding of the surrounding code. "
    "Penalize syntax errors, undefined identifiers, duplication of prefix or suffix content, and "
    "trivial or arbitrary completions. Reply with ONLY one JSON object: "
    "{\"score\": <integer 0 to 10>, \"notes\": \"<one short sentence>\"}. "
    "10 = coherent, well-determined, appropriately hard; 0 = broken or meaningless."
)

CLASSIFY_SYSTEM = (
    "You label code for a fill-in-the-middle dataset. Reply with ONLY one JSON object: "
    "{\"domain\": \"<short snake_case domain>\", \"task_type\": \"<short snake_case task>\", "
    "\"difficulty\": <integer 1 to 5>, \"concepts\": [\"<2-5 short concept tags>\"]}. "
    "Rate difficulty by how hard the missing code is to infer from the surrounding context: "
    "1 = trivial, 5 = requires careful reasoning over distant context."
)

JUDGE_TEXT = 3500
CLASSIFY_TEXT = 1200
MIN_COMPLETION_CHARS = 20
MAX_COMPLETION_CHARS = 8000
CLASSIFY_MAX_TOKENS = 400

REASON_HINTS = {
    "suffix too short": ("the suffix was empty or too short - it must contain at least 3 complete "
                         "lines of code that come AFTER the hole"),
    "prefix too short": "the prefix must contain the code that comes before the hole",
    "merged syntax invalid": ("prefix + completion + suffix did not parse as one file - usually a newline "
                              "or a closing brace is missing at the boundary between the parts; include "
                              "every removed line and end the completion at a complete line"),
    "completion too short": "the completion must contain the whole removed span",
    "completion too long": "the completion must stay under 30 lines",
    "completion is comments/whitespace only": "the completion must be real code, not comments or blanks",
    "sample too large": "keep the whole sample under the size budget",
    "completion duplicates suffix": "the completion must not repeat code already in the suffix",
    "completion duplicates prefix": "the completion must not repeat code already in the prefix",
}


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Build a mixed real + synthetic FIM fine-tuning dataset")
    parser.add_argument("-path", "--path", required=True, help="directory with real source code")
    parser.add_argument("-n", "--samples", type=int, default=200, help="total samples to write this run")
    parser.add_argument("-ratio", "--ratio", type=float, default=0.5, help="share of real samples (0..1)")
    parser.add_argument("-languages", "--languages", default="c,go,python,javascript")
    parser.add_argument("-seed", "--seed", type=int, default=854)
    parser.add_argument("-config", "--config", default=os.path.join(ROOT, "config.json"))
    parser.add_argument("-generator", "--generator", default="deepseek/deepseek-v4-flash-0731:floor")
    parser.add_argument("-temperature", "--temperature", type=float, default=0.8)
    parser.add_argument("-max-tokens", "--max-tokens", type=int, default=6000)
    parser.add_argument("-difficulty", "--difficulty", default="4,5", help="comma list of target difficulties 1-5")
    parser.add_argument("-gen-workers", "--gen-workers", type=int, default=8)
    parser.add_argument("-judge-workers", "--judge-workers", type=int, default=8)
    parser.add_argument("-judge-config", "--judge-config", default=os.path.join(ROOT, "judge.config.json"))
    parser.add_argument("-judge-min", "--judge-min", type=int, default=6)
    parser.add_argument("-gen-retries", "--gen-retries", type=int, default=2)
    parser.add_argument("-max-tries", "--max-tries", type=int, default=4, help="max generation waves per missing sample")
    parser.add_argument("-span-lines", "--span-lines", type=int, nargs=2, default=[1, 20], metavar=("MIN", "MAX"))
    parser.add_argument("-min-file-lines", "--min-file-lines", type=int, default=30)
    parser.add_argument("-cut", "--cut", choices=["mixed", "lines", "block"], default="mixed")
    parser.add_argument("-max-sample-chars", "--max-sample-chars", type=int, default=48000)
    parser.add_argument("-out", "--out", default=os.path.join(DATA_DIR, "dataset.jsonl"))
    parser.add_argument("-rejects", "--rejects", default=os.path.join(DATA_DIR, "rejected.jsonl"))
    parser.add_argument("-manifest", "--manifest", default=os.path.join(DATA_DIR, "manifest.json"))
    parser.add_argument("-no-judge", "--no-judge", action="store_true")
    parser.add_argument("-fresh", "--fresh", action="store_true",
                        help="overwrite the output file instead of appending to it")
    return parser.parse_args(argv)


def clean_text(text):
    return text.replace("\r\n", "\n").replace("\r", "\n")


def normalize_ws(text):
    return clean_text(text).strip()


def sample_hash(language, prefix, completion, suffix):
    digest = hashlib.sha256()
    for part in (language, normalize_ws(prefix), normalize_ws(completion), normalize_ws(suffix)):
        digest.update(part.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


def parse_json_object(text):
    cleaned = strip_fences(clean_text(text))
    try:
        data = json.loads(cleaned)
    except Exception:
        data = None
    if isinstance(data, dict):
        return data
    decoder = json.JSONDecoder()
    index = cleaned.find("{")
    while index != -1:
        try:
            data, _ = decoder.raw_decode(cleaned[index:])
        except Exception:
            data = None
        if isinstance(data, dict):
            return data
        index = cleaned.find("{", index + 1)
    return None


def clamp_difficulty(value, fallback):
    try:
        number = int(value)
    except (TypeError, ValueError):
        return fallback
    return max(1, min(5, number))


def sanitize_label(value):
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in str(value).lower())
    return "_".join(part for part in cleaned.split("_") if part)[:40]


def concepts_field(value):
    if not isinstance(value, list):
        return []
    out = []
    for item in value:
        item = str(item).strip()
        if item and item not in out:
            out.append(item)
    return out[:8]


def text_field(value):
    return value if isinstance(value, str) else ""


def truncate_sample(sample):
    out = dict(sample)
    for key in ("prefix", "suffix", "completion"):
        if isinstance(out.get(key), str) and len(out[key]) > 3000:
            out[key] = out[key][:3000] + "...[truncated]"
    return out


def bump(counter, key):
    counter[key] = counter.get(key, 0) + 1


def record_reject(rejects, source, reason, language=None, detail=None,
                  judge_score=None, judge_notes=None, sample=None):
    entry = {"source": source, "reason": reason}
    if language is not None:
        entry["language"] = language
    if detail:
        entry["detail"] = detail
    if judge_score is not None:
        entry["judge_score"] = judge_score
    if judge_notes:
        entry["judge_notes"] = judge_notes
    if sample:
        entry["sample"] = truncate_sample(sample)
    rejects.append(entry)


def add_usage(stats, result):
    stats["prompt_tokens"] += result.get("prompt_tokens") or 0
    stats["completion_tokens"] += result.get("completion_tokens") or 0
    cost = result.get("live_cost")
    if cost:
        stats["cost_usd"] += cost


def generate_messages(language, difficulty, domain, hint=None):
    user = (GEN_TEMPLATE
            .replace("{language}", language)
            .replace("{difficulty}", str(difficulty))
            .replace("{domain}", domain))
    if hint:
        user += (f"\n\nYour previous attempt was rejected: {hint}. "
                 "Fix exactly that problem and return the corrected JSON object.")
    return [
        {"role": "system", "content": GEN_SYSTEM},
        {"role": "user", "content": user},
    ]


def judge_messages(sample):
    prefix = sample["prefix"][-JUDGE_TEXT:]
    middle = sample["completion"][:JUDGE_TEXT]
    suffix = sample["suffix"][:JUDGE_TEXT]
    user = (
        f"Language: {sample['language']}\n\n"
        f"PREFIX (end):\n{prefix}\n\n"
        f"COMPLETION (the removed middle):\n{middle}\n\n"
        f"SUFFIX (start):\n{suffix}"
    )
    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": user},
    ]


def classify_messages(sample):
    tail = sample["prefix"][-CLASSIFY_TEXT:]
    middle = sample["completion"][:CLASSIFY_TEXT]
    head = sample["suffix"][:CLASSIFY_TEXT]
    user = (
        f"Language: {sample['language']}\n\n"
        f"CONTEXT BEFORE THE MISSING CODE (end of prefix):\n{tail}\n\n"
        f"MISSING CODE (rate this):\n{middle}\n\n"
        f"CONTEXT AFTER THE MISSING CODE (start of suffix):\n{head}"
    )
    return [
        {"role": "system", "content": CLASSIFY_SYSTEM},
        {"role": "user", "content": user},
    ]


def heuristic_domain(sample):
    folder = os.path.basename(os.path.dirname(sample.get("file") or ""))
    return sanitize_label(folder) or "unknown"


def heuristic_difficulty(sample):
    if sample.get("hole_type") == "block":
        return 4
    span = (sample.get("end_line") or 0) - (sample.get("start_line") or 0) + 1
    if span <= 2:
        return 2
    if span <= 5:
        return 3
    if span <= 12:
        return 4
    return 5


def apply_metadata(sample, meta):
    domain = sanitize_label(meta.get("domain") or "")
    task = sanitize_label(meta.get("task_type") or "")
    difficulty = clamp_difficulty(meta.get("difficulty"), None)
    if difficulty is None:
        difficulty = heuristic_difficulty(sample)
    sample["domain"] = domain or heuristic_domain(sample)
    sample["task_type"] = task or "unknown"
    sample["difficulty"] = difficulty
    sample["concepts"] = concepts_field(meta.get("concepts"))


def align_join(language, prefix, suffix, raw_completion):
    completions = []
    for comp in (raw_completion, strip_fences(raw_completion)):
        if comp not in completions:
            completions.append(comp)
        with_newline = comp if comp.endswith("\n") else comp + "\n"
        if with_newline not in completions:
            completions.append(with_newline)
    prefixes = [prefix]
    if not prefix.endswith("\n"):
        prefixes.append(prefix + "\n")
    for comp in completions:
        for pre in prefixes:
            _, merged = check_syntax(language, comp, pre, suffix)
            if merged:
                return pre, comp
    return None, None


def build_candidate(data, languages, max_sample_chars, expected_language=None):
    language = str(data.get("language", "")).lower().strip()
    if not language:
        language = expected_language or ""
    if language not in languages:
        return None, f"language not allowed: {language or '?'}"
    prefix = clean_text(text_field(data.get("prefix")))
    suffix = clean_text(text_field(data.get("suffix")))
    raw_completion = clean_text(text_field(data.get("completion")))
    completion = strip_fences(raw_completion)
    completion_stripped = completion.strip()
    if len(prefix.strip()) < 40:
        return None, "prefix too short"
    if len(suffix.strip()) < 20:
        return None, "suffix too short"
    if len(completion_stripped) < MIN_COMPLETION_CHARS:
        return None, "completion too short"
    if len(completion_stripped) > MAX_COMPLETION_CHARS:
        return None, "completion too long"
    if len(prefix) + len(suffix) + len(completion) > max_sample_chars:
        return None, "sample too large"
    if not is_meaningful(completion, language):
        return None, "completion is comments/whitespace only"
    if suffix.strip().startswith(completion_stripped) and len(completion_stripped) > 20:
        return None, "completion duplicates suffix"
    if prefix.strip().endswith(completion_stripped):
        return None, "completion duplicates prefix"
    prefix, completion = align_join(language, prefix, suffix, raw_completion)
    if completion is None:
        return None, "merged syntax invalid"
    hole_type = str(data.get("hole_type", "")).lower().strip()
    if hole_type not in ("block", "lines", "expression"):
        hole_type = "lines"
    return {
        "language": language,
        "domain": sanitize_label(data.get("domain") or "") or "unknown",
        "difficulty": clamp_difficulty(data.get("difficulty"), 4),
        "task_type": sanitize_label(data.get("task_type") or "") or "unknown",
        "hole_type": hole_type,
        "prefix": prefix,
        "suffix": suffix,
        "completion": completion,
        "concepts": concepts_field(data.get("concepts")),
    }, None


def real_sample(hole, language, prefix, suffix, completion):
    return {
        "language": language,
        "domain": None,
        "difficulty": None,
        "task_type": None,
        "hole_type": "block" if hole.cut == "block" else "lines",
        "prefix": prefix,
        "suffix": suffix,
        "completion": completion,
        "concepts": [],
        "generator_model": None,
        "temperature": None,
        "judge_score": None,
        "verified": True,
        "source": "real",
        "file": hole.file,
        "start_line": hole.start_line,
        "end_line": hole.end_line,
    }


def synthetic_sample(candidate, args, generator_name, score, notes):
    return {
        "language": candidate["language"],
        "domain": candidate["domain"],
        "difficulty": candidate["difficulty"],
        "task_type": candidate["task_type"],
        "hole_type": candidate["hole_type"],
        "prefix": candidate["prefix"],
        "suffix": candidate["suffix"],
        "completion": candidate["completion"],
        "concepts": candidate["concepts"],
        "generator_model": generator_name,
        "temperature": args.temperature,
        "judge_score": score,
        "verified": True,
        "source": "synthetic",
        "judge_notes": notes,
    }


def dedup_rows(rows):
    seen = set()
    out = []
    for row in rows:
        digest = sample_hash(row["language"], row["prefix"], row["completion"], row["suffix"])
        if digest in seen:
            continue
        seen.add(digest)
        out.append(row)
    return out


def interleave(real, synthetic):
    out = []
    for index in range(max(len(real), len(synthetic))):
        if index < len(real):
            out.append(real[index])
        if index < len(synthetic):
            out.append(synthetic[index])
    return out


def quota_split(languages, total, rng):
    base = total // len(languages)
    extra = total - base * len(languages)
    shuffled = languages[:]
    rng.shuffle(shuffled)
    quotas = {language: base for language in languages}
    for index in range(extra):
        quotas[shuffled[index % len(shuffled)]] += 1
    return quotas


def parse_difficulty(text):
    out = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        try:
            value = max(1, min(5, int(part)))
        except ValueError:
            continue
        if value not in out:
            out.append(value)
    return out or [4]


def build_real(args, languages, count, seen, rejects, stats, out_log):
    holes = build_holes(
        args.path, languages, count, args.seed,
        args.min_file_lines, args.span_lines[0], args.span_lines[1], args.cut, "balanced",
    )
    print(f"real holes found: {len(holes)} from {args.path} (requested {count})", flush=True)
    samples = []
    for hole in holes:
        language = hole.language
        text = read_source(hole.file)
        if text is None:
            record_reject(rejects, "real", "file unreadable", language=language)
            bump(stats["by_reason"], "real file unreadable")
            print(f"[real skip] file unreadable: {hole.file}", flush=True)
            continue
        prefix = hole.prefix
        suffix = hole.suffix
        if hole.cut == "lines":
            completion = hole.removed_text + "\n"
        else:
            completion = hole.removed_text
        raw = {
            "prefix": prefix,
            "completion": completion,
            "suffix": suffix,
            "file": hole.file,
            "start_line": hole.start_line,
            "end_line": hole.end_line,
        }
        if prefix + completion + suffix != text:
            record_reject(rejects, "real", "does not reconstruct file", language=language, sample=raw)
            bump(stats["by_reason"], "real does not reconstruct")
            print(f"[real skip] does not reconstruct: {hole.file}", flush=True)
            continue
        if not is_meaningful(completion, language):
            record_reject(rejects, "real", "completion is comments/whitespace only", language=language, sample=raw)
            bump(stats["by_reason"], "real not meaningful")
            print(f"[real skip] not meaningful: {hole.file}", flush=True)
            continue
        _, merged = check_syntax(language, completion, prefix, suffix)
        if not merged:
            stats["real_ts_flagged"] += 1
        digest = sample_hash(language, prefix, completion, suffix)
        if digest in seen:
            stats["duplicates"] += 1
            continue
        seen.add(digest)
        sample = real_sample(hole, language, prefix, suffix, completion)
        samples.append(sample)
        out_log.append(sample)
        print(f"[real {len(samples)}/{count}] {language} {os.path.basename(hole.file)}", flush=True)
    return samples


async def classify_real(provider, samples, args, stats):
    if not samples:
        return
    semaphore = asyncio.Semaphore(max(1, args.gen_workers // 2))

    async def classify_one(sample):
        async with semaphore:
            try:
                result = await provider.complete(
                    classify_messages(sample), temperature=0, max_tokens=CLASSIFY_MAX_TOKENS,
                )
            except ProviderError:
                result = None
        meta = {}
        if result is not None:
            add_usage(stats, result)
            parsed = parse_json_object(result["text"])
            if parsed is not None:
                meta = parsed
        apply_metadata(sample, meta)

    await asyncio.gather(*[asyncio.create_task(classify_one(sample)) for sample in samples])


async def produce_synthetic(provider, judge_provider, args, rng, language, difficulties,
                            seen, rejects, stats, gen_sem, judge_sem, out_log):
    last_problem = None
    for _ in range(args.gen_retries + 1):
        difficulty = rng.choice(difficulties)
        messages = generate_messages(language, difficulty, rng.choice(DOMAINS), last_problem)
        async with gen_sem:
            try:
                result = await provider.complete(
                    messages, temperature=args.temperature, max_tokens=args.max_tokens,
                )
            except ProviderError as e:
                record_reject(rejects, "synthetic", f"generator {e.kind}",
                              language=language, detail=e.message[:300])
                bump(stats["by_reason"], f"generator {e.kind}")
                continue
        stats["gen_calls"] += 1
        call = stats["gen_calls"]
        add_usage(stats, result)
        data = parse_json_object(result["text"])
        if data is None:
            record_reject(rejects, "synthetic", "unparsed json",
                          language=language, detail=result["text"][:300])
            bump(stats["by_reason"], "unparsed json")
            print(f"[gen {call}] {language} reject: unparsed json", flush=True)
            last_problem = "the reply was not parseable JSON"
            continue
        candidate, problem = build_candidate(data, args.languages, args.max_sample_chars, language)
        if candidate is None:
            record_reject(rejects, "synthetic", problem, language=language, sample=data)
            bump(stats["by_reason"], problem)
            print(f"[gen {call}] {language} reject: {problem}", flush=True)
            last_problem = REASON_HINTS.get(problem, problem)
            continue
        if candidate["language"] != language:
            record_reject(rejects, "synthetic", "language mismatch",
                          language=language, detail=candidate["language"], sample=candidate)
            bump(stats["by_reason"], "language mismatch")
            print(f"[gen {call}] {language} reject: language mismatch", flush=True)
            last_problem = f"the language field must be {language}"
            continue
        digest = sample_hash(candidate["language"], candidate["prefix"], candidate["completion"], candidate["suffix"])
        if digest in seen:
            stats["duplicates"] += 1
            print(f"[gen {call}] {language} skip: duplicate", flush=True)
            continue
        score = None
        notes = None
        if judge_provider is not None:
            print(f"[gen {call}] {language} validated, judging...", flush=True)
            async with judge_sem:
                try:
                    answer = await judge_provider.raw_chat(judge_messages(candidate))
                except ProviderError as e:
                    record_reject(rejects, "synthetic", f"judge {e.kind}", language=language, sample=candidate)
                    bump(stats["by_reason"], f"judge {e.kind}")
                    print(f"[gen {call}] {language} reject: judge {e.kind}", flush=True)
                    continue
            stats["judged"] += 1
            score, notes = parse_score(answer)
            if score is None:
                record_reject(rejects, "synthetic", "judge unparsed",
                              language=language, detail=answer[:300], sample=candidate)
                bump(stats["by_reason"], "judge unparsed")
                print(f"[gen {call}] {language} reject: judge unparsed", flush=True)
                continue
            stats["scores"].append(score)
            if score < args.judge_min:
                stats["judge_rejected"] += 1
                record_reject(rejects, "synthetic", "judge below min", language=language,
                              judge_score=score, judge_notes=notes, sample=candidate)
                bump(stats["by_reason"], "judge below min")
                print(f"[gen {call}] {language} reject: judge {score} < {args.judge_min}", flush=True)
                last_problem = f"the judge scored it {score}/10: {notes}"
                continue
        seen.add(digest)
        stats["kept_synth"] += 1
        sample = synthetic_sample(candidate, args, provider.cfg.name, score, notes)
        out_log.append(sample)
        print(f"[gen {call}] {language} KEPT judge={score} (kept {stats['kept_synth']}/{stats['synth_target']})", flush=True)
        return sample
    return None


async def run_synthetic(args, provider, judge_provider, rng, languages, total, seen, rejects, stats, out_log):
    quotas = quota_split(languages, total, rng)
    gen_sem = asyncio.Semaphore(max(1, args.gen_workers))
    judge_sem = asyncio.Semaphore(max(1, args.judge_workers))
    difficulties = parse_difficulty(args.difficulty)
    kept = []
    counts = {language: 0 for language in languages}
    started = time.perf_counter()
    stats["synth_target"] = total
    for wave in range(max(1, args.max_tries)):
        missing = {language: quotas[language] - counts[language]
                   for language in languages if quotas[language] > counts[language]}
        if not missing:
            break
        tasks = []
        for language, needed in missing.items():
            for _ in range(needed):
                tasks.append(asyncio.create_task(
                    produce_synthetic(provider, judge_provider, args, rng, language, difficulties,
                                      seen, rejects, stats, gen_sem, judge_sem, out_log)
                ))
        stats["waves"] = wave + 1
        for coro in asyncio.as_completed(tasks):
            sample = await coro
            if sample is not None:
                kept.append(sample)
                counts[sample["language"]] += 1
        print(f"synthetic wave {wave + 1}: kept {len(kept)}/{total}, generation calls {stats['gen_calls']} ({time.perf_counter() - started:.0f}s)", flush=True)
    return kept


class JsonlList(list):
    def __init__(self, path, fresh=False):
        super().__init__()
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        self.path = path
        self.handle = open(path, "w" if fresh else "a", encoding="utf-8", buffering=1)

    def append(self, row):
        super().append(row)
        self.handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def close(self):
        self.handle.close()


def write_rows(path, rows, append=False):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "a" if append else "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_manifest(path, manifest):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def build_manifest(args, generator, judge_config, n_real, n_synth, real, synthetic,
                   existing, rejects, stats):
    per_language = {}
    for sample in real + synthetic:
        per_language[sample["language"]] = per_language.get(sample["language"], 0) + 1
    scores = stats["scores"]
    judge_block = None
    if judge_config is not None:
        judge_block = {
            "name": judge_config.name,
            "model": judge_config.model,
            "min_score": args.judge_min,
            "scored": len(scores),
            "rejected": stats["judge_rejected"],
            "avg_score": (sum(scores) / len(scores)) if scores else None,
            "min_score_seen": min(scores) if scores else None,
            "max_score_seen": max(scores) if scores else None,
        }
    cost = stats["cost_usd"]
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "path": args.path,
        "config": args.config,
        "generator": {
            "name": generator.name,
            "model": generator.model,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "difficulty": args.difficulty,
        },
        "judge": judge_block,
        "requested": {"total": args.samples, "real": n_real, "synthetic": n_synth},
        "written": {
            "total": len(real) + len(synthetic),
            "real": len(real),
            "synthetic": len(synthetic),
            "per_language": per_language,
            "existing_rows": len(existing),
            "real_tree_sitter_flagged": stats["real_ts_flagged"],
        },
        "duplicates_skipped": stats["duplicates"],
        "rejects": {"total": len(rejects), "by_reason": stats["by_reason"]},
        "generator_usage": {
            "gen_calls": stats["gen_calls"],
            "waves": stats["waves"],
            "prompt_tokens": stats["prompt_tokens"],
            "completion_tokens": stats["completion_tokens"],
            "cost_usd": round(cost, 6) if cost else None,
        },
    }


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    args = parse_args(argv)
    languages = [l.strip().lower() for l in args.languages.split(",") if l.strip()]
    if not languages:
        print("no languages given", file=sys.stderr)
        return 2
    if args.samples < 1:
        print("samples must be >= 1", file=sys.stderr)
        return 2
    if not 0.0 <= args.ratio <= 1.0:
        print("ratio must be between 0 and 1", file=sys.stderr)
        return 2
    if args.span_lines[0] > args.span_lines[1]:
        print("span-lines MIN must be <= MAX", file=sys.stderr)
        return 2
    if not 0 <= args.judge_min <= 10:
        print("judge-min must be between 0 and 10", file=sys.stderr)
        return 2
    try:
        models = load_config(args.config)
    except ConfigError as e:
        print(f"config error: {e}", file=sys.stderr)
        return 2
    generator = None
    for model in models:
        if model.name == args.generator:
            generator = model
            break
    if generator is None:
        names = ", ".join(model.name for model in models)
        print(f"generator {args.generator} not found in {args.config}: {names}", file=sys.stderr)
        return 2
    if not generator.is_active():
        print(f"generator {generator.name} is deactivated", file=sys.stderr)
        return 2
    if "chat" not in generator.modes:
        print(f"generator {generator.name} has no chat mode", file=sys.stderr)
        return 2
    judge_config = None
    if not args.no_judge:
        try:
            judges = load_config(args.judge_config)
        except ConfigError as e:
            print(f"judge config error: {e}", file=sys.stderr)
            return 2
        if len(judges) != 1:
            print("judge config must contain exactly one model", file=sys.stderr)
            return 2
        if not judges[0].is_active():
            print("judge model is deactivated", file=sys.stderr)
            return 2
        judge_config = judges[0]

    args.languages = languages
    n_real = int(round(args.samples * args.ratio))
    n_synth = args.samples - n_real
    rng = random.Random(args.seed + 1)
    stats = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cost_usd": 0.0,
        "gen_calls": 0,
        "judged": 0,
        "judge_rejected": 0,
        "kept_synth": 0,
        "synth_target": 0,
        "duplicates": 0,
        "waves": 0,
        "real_ts_flagged": 0,
        "scores": [],
        "by_reason": {},
    }
    seen = set()
    rejects = JsonlList(args.rejects, fresh=True)
    existing = []
    if not args.fresh and os.path.exists(args.out):
        existing = read_jsonl(args.out)
        for row in existing:
            if isinstance(row, dict) and all(
                isinstance(row.get(key), str) for key in ("language", "prefix", "completion", "suffix")
            ):
                seen.add(sample_hash(row["language"], row["prefix"], row["completion"], row["suffix"]))
        print(f"appending to {args.out} ({len(existing)} rows, -fresh to start over)")

    out_log = JsonlList(args.out, fresh=args.fresh)
    real = build_real(args, languages, n_real, seen, rejects, stats, out_log)
    print(f"real kept: {len(real)}/{n_real}", flush=True)

    async def pipeline():
        provider = Provider(generator)
        judge_provider = Provider(judge_config) if judge_config is not None else None
        try:
            synthetic, _ = await asyncio.gather(
                run_synthetic(args, provider, judge_provider, rng, languages,
                              n_synth, seen, rejects, stats, out_log),
                classify_real(provider, real, args, stats),
            )
            return synthetic
        finally:
            await provider.close()
            if judge_provider is not None:
                await judge_provider.close()

    print(f"synthetic: generating {n_synth} with {generator.name}"
          + ("" if judge_config is None else f", judging with {judge_config.name} min {args.judge_min}"))
    synthetic = asyncio.run(pipeline())

    final = dedup_rows(interleave(real, synthetic))
    out_log.close()
    rejects.close()
    write_rows(args.out, existing + final)
    manifest = build_manifest(args, generator, judge_config, n_real, n_synth, real, synthetic,
                              existing, rejects, stats)
    write_manifest(args.manifest, manifest)

    print(f"wrote {len(final)} samples ({len(real)} real, {len(synthetic)} synthetic) -> {args.out}")
    print(f"rejects: {len(rejects)} -> {args.rejects}")
    if stats["scores"]:
        avg = sum(stats["scores"]) / len(stats["scores"])
        print(f"judge avg {avg:.1f}/10 over {len(stats['scores'])} scored ({stats['judge_rejected']} below min)")
    if stats["cost_usd"]:
        print(f"generator cost ~${stats['cost_usd']:.4f}")
    print(f"manifest -> {args.manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
