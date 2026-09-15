#!/usr/bin/env bash
set -euo pipefail

BIN="$HOME/Desktop/llama.cpp/build/bin/llama-server"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE="$DIR/cpu_servers"
HUB="/media/user/2TB/huggingface_cache/hub"
CTX=32768
THREADS=16

ORDER=(starcoder2-3b starcoder2-7b zeta-2.1 qwen2.5-coder-7b qwen2.5-coder-3b locllm)

declare -A PORTS=(
  [starcoder2-3b]=8020
  [starcoder2-7b]=8021
  [zeta-2.1]=8022
  [qwen2.5-coder-7b]=8023
  [qwen2.5-coder-3b]=8024
  [locllm]=8025
)

declare -A KIND=(
  [locllm]=locllm
)

declare -A WAIT=(
  [locllm]=600
)

LOCLLM_DIR="/home/user/Desktop/locLLM"
LOCLLM_ENTRY="$LOCLLM_DIR/model/infrance.py"
LOCLLM_PY="$LOCLLM_DIR/.venv/bin/python"
if ! "$LOCLLM_PY" -c "import torch" >/dev/null 2>&1; then
  LOCLLM_PY="/home/user/miniconda3/bin/python3"
fi

declare -A MODELS=(
  [starcoder2-3b]="$HUB/models--second-state--StarCoder2-3B-GGUF/snapshots/7fca3e2da2ce31df411461e2cb9cae2d2b492f35/starcoder2-3b-Q4_K_M.gguf"
  [starcoder2-7b]="$HUB/models--second-state--StarCoder2-7B-GGUF/snapshots/b5a3389fafb6c1686d221d4400ee3be5a702bcde/starcoder2-7b-Q4_K_M.gguf"
  [zeta-2.1]="$HUB/models--mradermacher--zeta-2.1-GGUF/snapshots/8c2528af755e2a640232eb25e7991cd812e04b07/zeta-2.1.Q4_K_M.gguf"
  [qwen2.5-coder-7b]="$HUB/models--Qwen--Qwen2.5-Coder-7B-Instruct-GGUF/snapshots/13fb94bfda8c8cf22497dc57b78f391a9acb426a/qwen2.5-coder-7b-instruct-q4_k_m-00001-of-00002.gguf"
  [qwen2.5-coder-3b]="$HUB/models--unsloth--Qwen2.5-Coder-3B-Instruct-128K-GGUF/snapshots/5326551926d06f7f9cab53c9b9b552e3bedfe8ba/Qwen2.5-Coder-3B-Instruct-Q4_K_M.gguf"
)

pid_file() { echo "$STATE/$1.pid"; }
log_file() { echo "$STATE/$1.log"; }

running() {
  local pid
  pid=$(cat "$(pid_file "$1")" 2>/dev/null) || return 1
  [[ -n "$pid" ]] || return 1
  kill -0 "$pid" 2>/dev/null
}

rss_kb() {
  local pid rss
  pid=$(cat "$(pid_file "$1")" 2>/dev/null) || { echo 0; return 0; }
  rss=$(ps -o rss= -p "$pid" 2>/dev/null | tr -d ' ') || rss=""
  echo "${rss:-0}"
}

health() {
  curl -sf -m 2 "http://127.0.0.1:${PORTS[$1]}/health" >/dev/null 2>&1
}

wait_health() {
  local name=$1 tries=${2:-180} i
  for ((i = 0; i < tries; i++)); do
    health "$name" && return 0
    running "$name" || return 1
    sleep 1
  done
  return 1
}

human() {
  awk -v kb="${1:-0}" 'BEGIN { printf "%.2f GiB", kb / 1048576 }'
}

start_one() {
  local name=$1 port=${PORTS[$1]:-} path=${MODELS[$1]:-} pid
  local kind=${KIND[$1]:-llama}
  if [[ -z "$port" ]]; then
    echo "$name: unknown model" >&2
    return 1
  fi
  if running "$name"; then
    echo "$name: already running (pid $(cat "$(pid_file "$name")"))"
    return 0
  fi
  if [[ "$kind" == locllm ]]; then
    if [[ ! -f "$LOCLLM_ENTRY" ]] || ! "$LOCLLM_PY" -c "import torch" >/dev/null 2>&1; then
      echo "$name: launcher not usable: $LOCLLM_PY $LOCLLM_ENTRY" >&2
      return 1
    fi
  elif [[ ! -f "$path" ]]; then
    echo "$name: model file not found: $path" >&2
    return 1
  fi
  if curl -sf -m 2 "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
    echo "$name: port $port already serves a model, not starting" >&2
    return 1
  fi
  mkdir -p "$STATE"
  if [[ "$kind" == locllm ]]; then
    CUDA_VISIBLE_DEVICES= LOCLLM_PORT="$port" nohup "$LOCLLM_PY" "$LOCLLM_ENTRY" \
      >"$(log_file "$name")" 2>&1 &
  else
    CUDA_VISIBLE_DEVICES= nohup "$BIN" -m "$path" -ngl 0 -c "$CTX" -t "$THREADS" \
      --parallel 1 --host 127.0.0.1 --port "$port" --alias "$name" \
      >"$(log_file "$name")" 2>&1 &
  fi
  pid=$!
  echo "$pid" >"$(pid_file "$name")"
  echo "$name: starting on port $port (pid $pid)"
  if wait_health "$name" "${WAIT[$1]:-180}"; then
    echo "$name: ready, rss $(human "$(rss_kb "$name")")"
  else
    echo "$name: failed to become healthy, see $(log_file "$name")" >&2
    return 1
  fi
}

stop_one() {
  local name=$1 pid i
  pid=$(cat "$(pid_file "$name")" 2>/dev/null) || { echo "$name: stopped"; return 0; }
  if ! kill -0 "$pid" 2>/dev/null; then
    rm -f "$(pid_file "$name")"
    echo "$name: stopped (stale pid $pid)"
    return 0
  fi
  kill "$pid" 2>/dev/null || true
  for ((i = 0; i < 50; i++)); do
    kill -0 "$pid" 2>/dev/null || break
    sleep 0.2
  done
  if kill -0 "$pid" 2>/dev/null; then
    kill -9 "$pid" 2>/dev/null || true
  fi
  rm -f "$(pid_file "$name")"
  echo "$name: stopped"
}

status_one() {
  local name=$1 pid rss state
  if running "$name"; then
    pid=$(cat "$(pid_file "$name")")
    rss=$(rss_kb "$name")
    if health "$name"; then state=ready; else state=loading; fi
    printf '%-18s port %-5s pid %-8s %-8s rss %s\n' "$name" "${PORTS[$name]}" "$pid" "$state" "$(human "$rss")"
  else
    printf '%-18s port %-5s stopped\n' "$name" "${PORTS[$name]}"
  fi
}

ram() {
  local total=0 name rss
  for name in "${ORDER[@]}"; do
    if running "$name"; then
      rss=$(rss_kb "$name")
      total=$((total + rss))
      printf '%-18s rss %s\n' "$name" "$(human "$rss")"
    fi
  done
  printf '%-18s rss %s\n' TOTAL "$(human "$total")"
  awk '/^MemTotal|^MemAvailable/ { printf "%-18s %.2f GiB\n", $1, $2/1048576 }' /proc/meminfo
  if command -v free >/dev/null 2>&1; then free -h; fi
}

usage() {
  cat >&2 <<EOF
usage: $0 {start|stop|status|ram} [model ...]

models: ${ORDER[*]}
EOF
  exit 2
}

cmd=${1:-}
if [[ $# -gt 0 ]]; then shift; fi

names=()
if [[ $# -gt 0 ]]; then
  for name in "$@"; do
    if [[ -z "${PORTS[$name]:-}" ]]; then
      echo "unknown model: $name" >&2
      exit 2
    fi
    names+=("$name")
  done
else
  names=("${ORDER[@]}")
fi

case "$cmd" in
  start)
    mkdir -p "$STATE"
    rc=0
    for name in "${names[@]}"; do
      start_one "$name" || rc=1
    done
    exit "$rc"
    ;;
  stop)
    for name in "${names[@]}"; do
      stop_one "$name"
    done
    ;;
  status)
    for name in "${names[@]}"; do
      status_one "$name"
    done
    ;;
  ram)
    ram
    ;;
  *)
    usage
    ;;
esac
