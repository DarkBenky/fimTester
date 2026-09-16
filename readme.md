# TODOs
- [ ] for each model test / check what fim format to use maybe test few samples for each model eval whitch is best then update config with fim fomat 

## Run the benchmark
```bash
python eval_completions.py -path ~/Desktop/gengin -samples 128 -config config.json -sampling random -workers 16 -seed 123
```

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

## CPU model servers

```bash
bash serve_cpu.sh start             # start all cpu-* models + locllm (ports 8020-8025)
bash serve_cpu.sh start zeta-2.1    # start one model
bash serve_cpu.sh status            # pid / health / rss per model
bash serve_cpu.sh ram               # total ram of all running servers
bash serve_cpu.sh stop              # stop all
```
