# Implement: FIM Format Auto-Detection

You are working in an existing repo (a FIM/code-completion evaluation harness). This doc is
self-contained — read it fully before writing code. It does not assume you have any prior
conversation context about this repo.

## Problem

`config.json` (validated by `config.py`) lets each model declare `fim_protocol` (one of
`deepseek_fim` | `openai_completions` | `llamacpp_infill`) and, for `openai_completions`, an
optional `fim_template` string with `{prefix}`/`{suffix}` placeholders (e.g. the Qwen tag style
`<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>`). Nobody knows in advance which
literal tag format (if any) a given model/server was actually fine-tuned on, so today someone has
to guess-and-check by hand.

Build a new script, `detect_fim_format.py`, that automatically probes each model in a config file
with a handful of real FIM requests using several candidate prompt formats, scores the results,
and reports (or writes back) the best `fim_protocol`/`fim_template`/`fim_endpoint` per model.

## Relevant existing code (read these files before writing anything)

- `config.py` — `load_config(path)` returns a list of `ModelConfig` objects (see the class for all
  fields: `name`, `model`, `endpoint`, `fim_endpoint`, `fim_protocol`, `fim_template`, `api_key`,
  `max_context_size`, `max_tokens`, `temperature`, `timeout`, `headers`, `extra_body`, `modes`,
  `pricing`, `device_type`). Do not change validation rules; for probing, build modified copies of
  a `ModelConfig` in memory per candidate format instead (`copy.copy(model_config)` then override
  `fim_protocol`/`fim_template`/`fim_endpoint` on the copy).
- `providers.py` — `Provider(model_config)` wraps an `AsyncOpenAI` client built from
  `model_config.endpoint`. Use `await provider.fim(prefix, suffix, stream=False)` — it already
  dispatches on `cfg.fim_protocol`:
  - `"openai_completions"`: if `cfg.fim_template` is set, it renders
    `template.replace("{prefix}", prefix).replace("{suffix}", suffix)` and POSTs it as the raw
    `prompt` to `{endpoint-base}/completions`. If no template, falls back to passing `prompt`/
    `suffix` as separate API params (same call shape as `deepseek_fim`).
  - `"deepseek_fim"`: passes `prompt=prefix, suffix=suffix` as native API params (no visible tags)
    to `{endpoint-base}/completions`, `max_tokens` capped at 4096.
  - `"llamacpp_infill"`: POSTs JSON `{input_prefix, input_suffix, n_predict, temperature,
    cache_prompt: true}` directly to `cfg.fim_endpoint` (a distinct URL, not derived from
    `endpoint`). Only meaningful for models that already declare a `fim_endpoint` in config.
  - Returns `{"text": ..., "prompt_tokens", "completion_tokens", "ttft_ms", "attempts", ...}` or
    raises `ProviderError(kind, message, attempts)` on failure (already retries 3x internally on
    retryable errors — do not add another retry loop around it).
  - `await provider.close()` when done with a `Provider` instance.
- `runner.py` — `build_holes(root, languages, samples, seed, min_file_lines, span_min, span_max,
  cut, sampling)` walks `testcorpus/` (or any path) and returns a list of `Hole` objects (see
  `spans.py` for the `Hole` class: `.file`, `.language`, `.start_line`, `.end_line`, `.cut`,
  `.removed_text`, `.prefix`, `.suffix`, `.indent`). Reuse this instead of writing new fixed
  snippets — call it once with a small `samples` count (e.g. 2-3 per language) and reuse the same
  holes across every model × format combination so the comparison is apples-to-apples.
- `metrics.py` — reuse `normalize(text)`, `postprocess(completion, language, indent)`,
  `similarity(a, b)` (returns `(char_ratio, token_ratio)` via `difflib.SequenceMatcher`), and
  `check_syntax(language, completion, prefix, suffix)` (returns `(fragment_valid, merged_valid)`
  booleans via tree-sitter). Do not reimplement these.
- `judge.py` — reuse the pattern of `judge_prompt(row)` / `parse_score(text)` /
  `Provider.raw_chat(messages)` for an optional LLM-judge tie-break pass. `judge.py`'s functions
  are plain module-level functions, importable via `from judge import judge_prompt, parse_score`.
- `report.py` — reuse `write_csv(path, rows, fields)` / `read_jsonl(path)` helpers for output
  files; keep the same style (plain csv/jsonl, no extra deps).
- `PROMPT.md` — the original spec for the whole tool. Follow its STYLE GUIDE: keep code clean,
  simple, understandable, no comments, one function does one job, split by file, entry file only
  calls into other modules.

## What to build

### 1. New file `fim_formats.py`

A `FORMATS` list of candidate dicts, each `{"name": ..., "protocol": ..., "template": ...}`
(`template` is `None` when not applicable):

1. `starcoder` — protocol `openai_completions`, template
   `"<fim_prefix>{prefix}<fim_suffix>{suffix}<fim_middle>"`
2. `codellama` — protocol `openai_completions`, template
   `"<PRE> {prefix} <SUF>{suffix} <MID>"`
3. `deepseek_tokens` — protocol `openai_completions`, template
   `"<｜fim▁begin｜>{prefix}<｜fim▁hole｜>{suffix}<｜fim▁end｜>"`
4. `qwen_openai_tags` — protocol `openai_completions`, template
   `"<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>"`
5. `native_suffix_param` — protocol `deepseek_fim`, template `None` (exercises the API-level
   `prompt`/`suffix` params directly, no visible tags)
6. `llamacpp_infill` — protocol `llamacpp_infill`, template `None` — only test this one for a
   model that already has a non-null `fim_endpoint` in its config entry (it hits that literal URL
   with the `/infill` JSON shape, unrelated to tag guessing)

Every sentinel token/tag appearing across all templates should also be exposed as a flat list
(e.g. `LEAK_TOKENS`) for the leak-detection check below.

### 2. New file `detect_fim_format.py`

CLI (argparse, same two-dash-friendly style as `eval_completions.py`):

```
python detect_fim_format.py -config config.json
  [-path testcorpus] [-languages c,go,python,javascript]
  [-samples-per-format 3] [-workers 4] [-seed 42] [-timeout 60]
  [-formats starcoder,codellama,...]        # optional filter, default = all applicable
  [-out-jsonl fim_format_results.jsonl] [-out-csv fim_format_summary.csv]
  [-judge] [-judge-config judge.config.json]  # optional LLM tie-break pass
  [-apply]                                    # optional: patch -config in place
```

Steps:

1. `models = load_config(args.config)`.
2. Build one shared list of holes via `runner.build_holes(args.path, languages, n, args.seed,
   min_file_lines=30, span_min=1, span_max=20, cut="mixed", sampling="balanced")` where `n =
   args.samples_per_format * len(languages)` (roughly — just needs to be small, e.g. 2-3 per
   language). Reuse the exact same hole list for every model and every format.
3. For each model, determine its applicable format list: all of `FORMATS` except skip
   `llamacpp_infill` when `model.fim_endpoint` is `None`. Apply `-formats` filter if given.
4. For each `(model, format)` pair: `cfg = copy.copy(model)`; set `cfg.fim_protocol =
   format["protocol"]`, `cfg.fim_template = format["template"]`; if format is not
   `llamacpp_infill`, leave `cfg.fim_endpoint` as-is (it's unused by that protocol path anyway).
   Create one `Provider(cfg)`, reuse it across all holes for that pair, bound concurrency with an
   `asyncio.Semaphore(args.workers)` (mirror the pattern in `runner.run`), call
   `await provider.fim(hole.prefix, hole.suffix, stream=False)` per hole, `await provider.close()`
   when done with that pair. Catch `ProviderError` per hole (record `status="error"`, empty
   completion, don't crash the whole run).
5. Score every row:
   - `completion_norm = postprocess(row_text, hole.language, hole.indent)` then `normalize(...)`
   - `char_sim, token_sim = similarity(completion_norm, normalize(hole.removed_text))`
   - `frag_ok, merged_ok = check_syntax(hole.language, completion_norm, hole.prefix, hole.suffix)`
   - `leaked = any(tok in row_text for tok in LEAK_TOKENS)`
   - `composite = 0.0 if row.status == "error" or not completion_norm.strip() else
     0.5 * token_sim + 0.3 * (1.0 if merged_ok else 0.0) + 0.2 * (0.0 if leaked else 1.0)`
6. Write every row to `-out-jsonl` (fields: `model, format, protocol, file, language, start_line,
   end_line, removed_text, completion_raw, completion_norm, similarity_char, similarity_token,
   syntax_valid_fragment, syntax_valid_merged, leaked, composite, status, error`). Aggregate per
   `(model, format)` into `-out-csv` (fields: `model, format, requests, avg_similarity_char,
   avg_similarity_token, syntax_valid_rate, leak_rate, composite_score`), using `report.write_csv`.
7. If `-judge` is passed: load a single judge model via `load_config(args.judge_config)` (same
   one-model constraint as `judge.py`), for each model take its top 1-2 formats by composite
   score, build judge prompts with `judge_prompt`-equivalent rows (`removed_text`,
   `completion_norm`) and score via `raw_chat` + `parse_score`, then recompute those formats'
   composite as `0.6 * (avg_judge_score / 10) + 0.2 * (1.0 if merged_ok else 0.0) + 0.2 * (0.0 if
   leaked else 1.0)` to break near-ties before picking the final winner.
8. Print a ranked table per model (format → composite score, mark the winner), plus a
   ready-to-paste JSON snippet of the winning `fim_protocol` (+ `fim_template` if applicable, +
   `fim_endpoint` only for the `llamacpp_infill` case) for each model.
9. If `-apply` is passed: read the raw JSON array from `-config` (plain `json.load`, not through
   `ModelConfig`), copy it to `<config>.bak` first, then for each model whose winning format beats
   its currently configured result (or if it has no `fim_protocol` at all yet), overwrite that
   entry's `fim_protocol`/`fim_template` (and `fim_endpoint` only when the winner is
   `llamacpp_infill` and the entry doesn't already have one — never invent a fake URL, skip
   applying that model's `llamacpp_infill` winner if it has no `fim_endpoint`), then
   `json.dump(data, f, indent=2)` back to the same path. Print what changed per model.

## Output artifacts

- `fim_format_results.jsonl` — one row per model × format × hole
- `fim_format_summary.csv` — one row per model × format, aggregated
- stdout: ranked table + copy-pasteable config snippet per model
- (`-apply` only) `<config>.bak` backup + patched config file

## Verification checklist

1. Run against at least one locally-served model (see `serve_cpu.sh`) with
   `-samples-per-format 2 -languages python` — confirm requests succeed and the format the server
   actually expects scores highest.
2. Confirm a model without `fim_endpoint` never gets a `llamacpp_infill` row.
3. Confirm `-apply` output still loads cleanly through `config.load_config` afterward and that
   `<config>.bak` was created and matches the pre-apply file.
4. Re-run `eval_completions.py` with the patched config and sanity-check that fim-mode results
   didn't regress compared to before.

## Constraints / style

- No comments in code beyond what the repo's existing files already show (one-line only, and only
  when the code truly can't say it — match the existing modules' near-comment-free style).
- One function does one job; `detect_fim_format.py` should mainly orchestrate calls into
  `fim_formats.py`, `config.py`, `runner.py`, `providers.py`, `metrics.py`, `report.py`, and
  (optionally) `judge.py` — avoid re-implementing logic that already exists in those modules.
- Async/concurrent like the rest of the harness (`asyncio`, `Semaphore` for `-workers`), not
  sequential blocking requests.
- Never log or write API keys anywhere in the output files.
