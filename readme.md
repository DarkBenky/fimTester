# TODOs
- [x] for each model test / check what fim format to use maybe test few samples for each model eval whitch is best then update config with fim fomat 

## Detect the FIM format per model

```bash
python detect_fim_format.py -config config.json -models cpu-starcoder2-3b -languages python -samples-per-format 2
python detect_fim_format.py -config config.json -apply      # patch config.json (keeps config.json.bak)
```

Probes every model x candidate format on one shared hole set from `testcorpus`, writes
`fim_format_results.jsonl` + `fim_format_summary.csv`, prints a ranked table per model with the
winning `fim_protocol`/`fim_template` snippet. `-judge` re-scores the top formats with the judge
model before picking winners; `-apply` patches config.json in place and skips anything unsafe.

## Run the benchmark
```bash
python eval_completions.py -path ~/Desktop/gengin -samples 128 -config config.json -sampling random -workers 16 -seed 123
python eval_completions.py -path ~/Desktop/gengin -samples 128 -config config.json -sampling random -workers 16 -seed 123 -remote-only # run benchmark only on API models
```

Add `-remote-only` to skip every local model (localhost / 127.0.0.1) — nothing is sent to local
servers, only the remote APIs are benchmarked.

Rows are appended to `results.jsonl` as they finish, but `summary.csv` / `per_file.csv` are
only written when the run ends. Killing the run (Ctrl-C) keeps every finished row — rebuild
and finish everything with one command:

```bash
python summarize.py            # rebuild summary.csv + per_file.csv
python summarize.py -judge     # also score any unscored rows (already scored rows are
                               # reused, nothing is re-paid), then write summary.csv,
                               # per_file.csv, results_judged.jsonl, summary_judged.csv
```

`avg_judge_score` is part of the same summary table, so after `-judge` the scores are filled
in both `summary.csv` and `summary_judged.csv`.

## Eval the results

```bash
python judge.py -results results.jsonl -config judge.config.json -workers 16
```

## Build the LoRA dataset (real + synthetic FIM samples)

`synteticData/generate.py` mixes real holes (same machinery as `eval_completions.py`) 50/50 with
synthetic samples from `deepseek/deepseek-v4-flash-0731:floor`. In the default `file` mode the model
writes a whole file and the hole is cut locally with tree-sitter (`-synthetic-mode triple` makes the
model author prefix/suffix/completion directly instead). Every synthetic sample is compiled and
executed with the standard toolchain (C: `gcc` + ASan/UBSan, Python, JavaScript via `node`, Go) —
compile errors, crashes, leaks, and non-zero exits are rejected, and the toolchain's message is fed
back into the next attempt; `exec_verified: true` marks rows that passed. Disable with
`-no-verify-exec`. Samples also pass the judge (`>= -judge-min`); real rows must reconstruct their
source file exactly.

```bash
python synteticData/generate.py -path ~/Desktop/gengin -n 200 -fresh   # first run
python synteticData/generate.py -path ~/Desktop/gengin -n 100          # append a batch
python synteticData/generate.py -h                                     # all flags
```

Outputs in `synteticData/data/`: `dataset.jsonl`, `rejected.jsonl` (reasons), `manifest.json`.
Rows are your schema + `source` (real: `file`/lines; synthetic: `judge_notes`). Without `-fresh`
runs append (duplicates skipped). The judge is slow (thinking model) — `-judge-workers N` or
`-no-judge` for speed; rejected slots are refilled up to `-max-tries`. Without `-seed` a fresh
random seed is picked each run (printed at start and stored in the manifest; pass it back via
`-seed N` to reproduce). Re-running with the same `-seed` keeps 0 new real samples once its holes
are stored — just omit `-seed` (or change it) to mine more. A `<out>.lock` (e.g.
`dataset.jsonl.lock`) refuses a second concurrent run against the same `-out`.

## CPU model servers

```bash
bash serve_cpu.sh start             # start all cpu-* models + locllm (ports 8020-8025)
bash serve_cpu.sh start zeta-2.1    # start one model
bash serve_cpu.sh status            # pid / health / rss per model
bash serve_cpu.sh ram               # total ram of all running servers
bash serve_cpu.sh stop              # stop all
```

