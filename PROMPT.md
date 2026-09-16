TODO: Build a Python code-completion/FIM evaluation harness.

Goal: measure how well a model fills a removed middle span of real code, and compare models on the exact same holes.

## FIM Format Auto-Detection

`detect_fim_format.py` sends a small number of real FIM requests per model using several candidate
prompt formats (StarCoder tags, CodeLlama tags, DeepSeek-Coder tokens, Qwen/OpenAI tags, native
suffix-param protocol, llama.cpp infill), scores the completions, and reports (or applies) the best
`fim_protocol`/`fim_template`/`fim_endpoint` per model. The candidate table lives in `fim_formats.py`.

CLI:
  python detect_fim_format.py -config config.json
    -path testcorpus -languages c,go,python,javascript -samples-per-format 3
    -workers 4 -seed 42 -timeout 60 -max-tokens 192
    -formats starcoder,codellama,deepseek_tokens,qwen_openai_tags,native_suffix_param,llamacpp_infill
    -models name1,name2 -remote-only
    -out-jsonl fim_format_results.jsonl -out-csv fim_format_summary.csv
    -judge -judge-config judge.config.json
    -apply

  - one shared hole set is probed with every model x format pair; rows stream to the output file as they finish
  - -max-tokens caps the probe completion length per model (useful on slow local servers)
  - -remote-only skips models with a local endpoint (localhost / 127.0.0.1), so only remote APIs are probed
  - composite = 0.5 * token similarity + 0.3 * merged syntax + 0.2 * (0 when a template tag leaked into the completion)
  - fim_format_results.jsonl rows: model, format, protocol, file, language, lines, removed_text,
    completion_raw/norm, similarities, syntax flags, leaked, composite, status, error
  - fim_format_summary.csv rows: model, format, requests, avg_similarity_char, avg_similarity_token,
    syntax_valid_rate, leak_rate, composite_score
  - -judge re-scores the contender formats per model (all within 0.10 composite of the best) with the judge\n    model (0.6 * judge/10 + 0.2 * syntax + 0.2 * no leak) and picks the winner among the judged formats
  - -apply patches config.json in place (backup config.json.bak) and skips any model whose winner is not safe to apply

Full implementation spec (self-contained, written for a model with no prior context on this
repo): [FIM_FORMAT_DETECTION.md](FIM_FORMAT_DETECTION.md)

CLI:
  python eval_completions.py -path ./my-codebase -samples 100 -config ./config.json
    -seed 42 -jsonl results.jsonl -csv summary.csv -per-file-csv per_file.csv
    -workers 4 -timeout 60 -span-lines 1 20 -cut mixed|lines|block
    -min-file-lines 30 -languages c,go,python,javascript
    -sampling balanced|random
    -write-samples samples.json -samples-file samples.json -no-stream
    -fresh -remote-only

  - every flag also works with two dashes (-path / --path)
  - defaults: seed 42, jsonl results.jsonl, csv summary.csv, per-file-csv per_file.csv, workers 4, timeout 60, cut mixed, span-lines 1 20, min-file-lines 30, languages = all four
  - results accumulate: a run appends its rows to the existing -jsonl and the summaries are recomputed over all rows from every previous run, so repeated runs average together; -fresh overwrites instead
  - -remote-only skips every model whose endpoint is local (localhost / 127.0.0.1), for benchmarking only the remote APIs

Core:
  - recursively collect source files (skip .git, node_modules, vendor, build, dist, target, hidden dirs)
  - split -samples into balanced per-language quotas across the languages present; redistribute leftover quota to languages that still have usable files
  - or -sampling random: pick -samples files uniformly at random from the whole repo, no language balancing
  - pick holes with one seeded RNG (same seed = same holes); prefer a distinct file per hole, repeat a file only when the quota exceeds the number of usable files
  - remove a middle span, keep prefix + suffix, trim context to the model's max_context_size
  - every model completes the same holes in both modes (fim + chat)
  - compare the completion against the original removed code
  - write JSONL/CSV results as requests finish

Cutting a hole:
  - only in the middle region: never in the first or last 3 lines
  - file needs at least --min-file-lines lines, prefix and suffix must both be non-empty
  - reject spans that are whitespace-only or comment-only
  - --cut mixed (default): the RNG picks per hole between a line span and a whole statement block
  - line span: random start, length between --span-lines MIN and MAX
  - block: one complete tree-sitter node from the per-language block types (function/class/if/for/while/declaration/...), fully inside the middle region, at most 40 lines, never the first or last top-level node
  - the cut style is part of the recorded hole, so a replay reproduces it exactly

Modes (every model runs twice per hole):
  - fim: a real FIM request through fim_endpoint + fim_protocol
  - chat: the same prefix/suffix sent to the chat endpoint with a strict instruction (complete only the missing code, output only code)
  - a model without fim_endpoint gets a fim row with status unsupported, not an error

FIM protocols (endpoint URLs are full URLs from the config, the harness never guesses paths):
  - deepseek_fim: POST {model, prompt, suffix, max_tokens, temperature}, read choices[0].text (DeepSeek beta URL, max_tokens <= 4096)
  - openai_completions: POST {model, prompt, suffix, max_tokens, temperature}, read choices[0].text; if fim_template is set, render it first with {prefix} and {suffix}
  - llamacpp_infill: POST {input_prefix, input_suffix, n_predict, temperature, cache_prompt: true}, read the content field

Completion post-processing:
  - strip markdown fences and leading/trailing blank lines
  - Python: re-indent the completion to the hole's indent level
  - completion_raw is kept as returned, completion_norm is used for every metric and syntax check

Context budget:
  - budget = max_context_size - max_tokens, tokens estimated as chars/4
  - keep the prefix tail and the suffix head (the text nearest the hole) and record context_truncated

Metrics (per model x hole x mode):
  - exact match: exact_match_raw and exact_match_norm (rstrip every line, drop leading/trailing blank lines)
  - similarity: similarity_char and similarity_token (difflib.SequenceMatcher ratio on characters and on tokens)
  - syntax validity: syntax_valid_fragment and syntax_valid_merged via tree-sitter (merged = prefix + completion + suffix; valid when the tree has no ERROR/MISSING nodes; the fragment check wraps the completion in a minimal per-language container)
  - response latency: latency_ms and ttft_ms (time to first streamed token)
  - token usage: prompt_tokens and completion_tokens from the API, estimated as chars/4 when usage is missing
  - failed requests: status ok | error | unsupported plus error type, message and attempts
  - per-file results: every row carries file + language, so results aggregate per file

Nice To Have (in scope)
  - multiple models/endpoints answering the same holes in parallel
  - benchmark reproducibility via fixed seed and a replayable samples.json
  - cost tracking (OpenRouter live cost, DeepSeek and others from the price table)
  - TTFT via streaming

Focus mainly on languages that i use so C, Go, Python, JavaScript 

Pipeline:
  1. take the codebase path and collect source files
  2. choose files and holes with one seeded RNG, balanced across languages
  3. cut a middle span, keep prefix + suffix, trim context to max_context_size
  4. send prefix + suffix to every model in both modes (fim + chat)
  5. score the completion against the removed original (exact match, similarity, syntax)
  6. record latency, ttft, tokens and cost into JSONL/CSV rows as they finish

CLI example : ```python eval_completions.py \ -path ./my-codebase \ -samples 100 \ -config ./config.json```

Config:
  JSON array, valid JSON, one entry per model; the whole file is validated before the first request and every problem is reported at once.
  - name: label used in all results
  - model: model id sent to the API
  - endpoint: chat completions URL (used by chat mode)
  - fim_endpoint + fim_protocol: FIM URL + protocol (deepseek_fim | openai_completions | llamacpp_infill); omit for chat-only models
  - fim_template: optional raw FIM template for openai_completions with {prefix} and {suffix} placeholders (e.g. Qwen: <|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>)
  - api_key_env: name of the env var holding the key (preferred); api_key: literal value, $NAME also resolves to an env var; null for local endpoints
  - max_context_size: token budget for prefix + suffix
  - max_tokens (default 256; DeepSeek FIM caps it at 4096), temperature (default 0), timeout seconds (default 60)
  - modes: subset of [fim, chat] to run for this model (default both); disabled modes are skipped entirely, no rows
  - device_type: optional label of the physical device shared by local servers (e.g. cpu, gpu); models with the same device_type keep at most one request in flight in total across all of them, models without it are unrestricted
  - deactivated: optional object (e.g. {"reason": "Too expensive"}); deactivated models are never requested - eval, detection and judge all skip them
  - headers: extra HTTP headers (OpenRouter etc.)
  - pricing: {input: usd_per_mtok, output: usd_per_mtok} overrides the built-in price table
  - api keys are never logged and never written to the result files

Example config.json:
```json
[
  {
    "name": "gpt-4o",
    "model": "gpt-4o",
    "endpoint": "https://api.openai.com/v1/chat/completions",
    "api_key_env": "OPENAI_API_KEY",
    "max_context_size": 128000
  },
  {
    "name": "local-qwen-30b",
    "model": "qwen3-coder-30b",
    "endpoint": "http://localhost:8012/v1/chat/completions",
    "fim_endpoint": "http://localhost:8012/infill",
    "fim_protocol": "llamacpp_infill",
    "api_key_env": null,
    "max_context_size": 16000
  },
  {
    "name": "deepseek-flash",
    "model": "deepseek-flash",
    "endpoint": "https://api.deepseek.com/v1/chat/completions",
    "fim_endpoint": "https://api.deepseek.com/beta/completions",
    "fim_protocol": "deepseek_fim",
    "api_key_env": "DEEPSEEK_API_KEY",
    "max_context_size": 64000
  },
  {
    "name": "qwen3-coder-next",
    "model": "qwen/qwen3-coder-next",
    "endpoint": "https://openrouter.ai/api/v1/chat/completions",
    "api_key_env": "OPENROUTER_API_KEY",
    "max_context_size": 128000
  }
]
```

Example usage of FIM from deep seek:
```
FIM Completion (Beta)
In FIM (Fill In the Middle) completion, users can provide a prefix and a suffix (optional), and the model will complete the content in between. FIM is commonly used for content completion、code completion.

Notice
The max tokens of FIM completion is 4K.
The user needs to set base_url=https://api.deepseek.com/beta to enable the Beta feature.
Sample Code
Below is a complete Python code example for FIM completion. In this example, we provide the beginning and the end of a function to calculate the Fibonacci sequence, allowing the model to complete the content in the middle.

from openai import OpenAI

client = OpenAI(
    api_key="<your api key>",
    base_url="https://api.deepseek.com/beta",
)

response = client.completions.create(
    model="deepseek-flash",
    prompt="def fib(a):",
    suffix="    return fib(a-1) + fib(a-2)",
    max_tokens=128
)
print(response.choices[0].text)
```

Outputs:
  - results.jsonl, one row per model x mode x hole:
    run_id, timestamp, seed, model, mode, file, language, start_line, end_line, cut, removed_text,
    completion_raw, completion_norm, prompt_tokens, completion_tokens, context_truncated,
    latency_ms, ttft_ms, cost_usd, exact_match_raw, exact_match_norm, similarity_char, similarity_token,
    syntax_valid_fragment, syntax_valid_merged, status, error, attempts
  - summary.csv, per model x mode:
    samples, ok, unsupported, errors, error_rate, exact_match %, exact_match_norm %,
    avg similarity_char, avg similarity_token, syntax_valid %, avg / median / p95 latency,
    avg ttft, total prompt tokens, total completion tokens, total cost
  - per_file.csv, per model x mode x file:
    samples, exact %, avg similarity, syntax_valid %, errors, cost

Robustness:
  - retries: 3 attempts with 1s / 2s / 4s backoff on 429, 5xx, timeouts and connection errors; other 4xx fail immediately
  - a failed request still writes a row, so error rate is per model x mode, not per run
  - concurrency: async, every model x hole x mode in one queue, -workers caps in-flight requests, rows are written as they finish
  - streaming is on by default so ttft_ms is recorded; -no-stream disables it

Cost:
  - OpenRouter requests ask for usage.include and use the real cost from the response when present
  - DeepSeek and everything else use the static table in pricing.py (USD per million tokens) or the per-model pricing override
  - cost_usd is null when no price is known

Reproducibility / replay:
  - one seed drives all randomness; seed, hole, cut and mode are recorded in every row
  - -write-samples writes every sampled hole to samples.json
  - -samples-file replays exactly those holes against any config, so a model added later is compared on the same samples
  - samples.json entry: {file, language, start_line, end_line, cut, removed_text}

Dependencies: openai, httpx, tree_sitter, tree_sitter_python, tree_sitter_javascript, tree_sitter_go, tree_sitter_c (Python 3.10+)

Files (each module owns one job; the entry file only calls into them):
  - eval_completions.py: argument parsing, calls the pipeline, nothing else
  - config.py: load + validate config.json, resolve api keys
  - files.py: recursive file collection, language detection, filters
  - spans.py: hole selection (line spans and tree-sitter blocks), cut, indent helpers
  - providers.py: chat + FIM clients (openai SDK + httpx), retries, streaming, response post-processing
  - metrics.py: normalization, similarity, tree-sitter syntax checks
  - pricing.py: price table + cost calculation
  - report.py: JSONL/CSV writers and summary aggregation
  - judge.py: post-run LLM judge (python judge.py -results results.jsonl -config judge.config.json) - scores each ok completion 0-10 with notes via a cheap judge model (e.g. z-ai/glm-5.3-flash:floor), writes results_judged.jsonl and summary_judged.csv (override with -csv); -reuse carries over scores from the previous results_judged.jsonl and only judges rows that have none yet
  - fim_formats.py: candidate FIM prompt formats (tags/templates) + leak tokens
  - detect_fim_format.py: FIM format auto-detection (probe, score, rank, optional -judge tie-break, optional -apply)
  - runner.py: job building, async orchestration, samples.json read/write

STYLE GUIDE:
  - keep code clean simple understandable no comments
  - split functionality into different files so eval_completions.py only calls methods from the other modules
  - one function does one job
