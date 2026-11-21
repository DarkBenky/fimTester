#!/usr/bin/env python3
"""Switch the small model between the fast and the big-model (Strata) profiles.

Two llama.cpp profiles run the SAME small model (Nemotron-3.5-Bolt-E4B):
  fast     = whole model on the RTX 3090 (llama-server-zeta.service)
  companion = whole model on the RTX 3060 (llama-server-zeta-3060.service)
Both bind port 8030, so /small/, /infill and /v1/completions answer identically
in either mode; only the speed differs.

Policy:
- Fast profile (default): small model on the RTX 3090 only; the Strata server is
  up on 8080 with its model unloaded (started lazily, "idle_unload_s" frees it).
- Local-first: when the big model is loaded and no other chat is running, a chat
  request on 8013 stops the fast profile, starts the 3060 companion, loads (or
  wakes) the Strata model, then forwards the request.
- Cloud overflow (enabled when DEEPSEEK_API_KEY is set): a chat request goes to
  the DeepSeek API instead when the local model cannot serve it right away -
  another request is already running (Strata serves one at a time), a profile
  switch is in progress, or the model is idle-unloaded/cold. The local model is
  warmed in the background either way, so the next request lands locally.
- When Strata has idle-slept and no big request came for five minutes the
  router restores the fast 3090-only profile. The Strata HTTP server stays
  up (model unloaded) so the next big request is a warm reload.
- /small/... always answers from the small model: no profile switch, no cloud.
- /infill and /v1/completions are always served by the small model (Strata is
  chat-only: /v1/chat/completions and /v1/messages). They are relayed through
  the 8031 capture proxy because this model has no FIM tokens: the proxy turns
  them into a chat turn and logs the pair for training.
- Every POST that reaches the router is appended as one JSON line to
  ROUTER_LOG_DIR/requests-YYYY-MM-DD.jsonl (default /media/user/sde1/conversationsLLM).
"""

from __future__ import annotations

import http.client
import json
import os
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

HOST = "127.0.0.1"          # upstream connections (8030 / 8080)
BIND_HOST = "0.0.0.0"      # listen on all interfaces (LAN clients from the notebook)
LISTEN_PORT = 8013
FAST_PORT = 8030
BIG_PORT = 8080
# /infill and /v1/completions are relayed through the capture proxy: the small
# model has no FIM tokens, so the proxy rewrites them into a chat turn (and logs
# the pair for training). See ~/Desktop/zeta-proxy/zeta_proxy.py.
FIM_PORT = 8031
# /small/... is the "general purpose small model" endpoint: it always answers
# from the small local model on FAST_PORT, never switches profiles and never
# overflows to the cloud. Thinking is on unless the client says otherwise.
SMALL_PREFIX = "/small"
FAST_UNIT = "llama-server-zeta.service"
SHARED_UNIT = "llama-server-zeta-3060.service"   # same small model on the RTX 3060 while Strata runs
BIG_UNIT = "strata-iq3s.service"                 # Strata server (model loaded on demand)
SYSTEMCTL = "/usr/bin/systemctl"

HOP = {
    "host", "connection", "keep-alive", "transfer-encoding", "upgrade",
    "proxy-connection", "te", "trailer", "content-length", "accept-encoding",
}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except ValueError:
        return default


# DeepSeek cloud overflow (only active when a key is present).
DEEPSEEK_KEY = os.environ.get("DEEPSEEK_API_KEY", "").strip()
DEEPSEEK_HOST = "api.deepseek.com"
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-flash").strip()
DEEPSEEK_EFFORT = os.environ.get("DEEPSEEK_REASONING_EFFORT", "max").strip()
MAX_LOCAL_INFLIGHT = max(1, _env_int("MAX_LOCAL_INFLIGHT", 1))
LOCAL_WAIT_S = max(0.0, _env_float("LOCAL_WAIT_S", 0.0))
# Chat answers get at least this many completion tokens: with thinking enabled the
# reasoning is billed and capped through max_tokens, so a small client budget can
# be spent entirely on reasoning and return empty content.
MIN_MAX_TOKENS = max(0, _env_int("MIN_MAX_TOKENS", 16384))
# Reasoning reaches the chat UI as one block per paragraph (or per this many
# characters) so the thought reads as a few sections instead of one wall of text.
# A paragraph shorter than REASONING_MIN_BLOCK waits for the next break: thinking
# traces have many one-line paragraphs that would each be a tiny "Thought" chip.
REASONING_BLOCK_CHARS = max(200, _env_int("REASONING_BLOCK_CHARS", 1200))
REASONING_MIN_BLOCK = max(80, _env_int("REASONING_MIN_BLOCK", 300))

# On-disk request log: one JSON object per line, one file per day.
LOG_DIR = os.environ.get("ROUTER_LOG_DIR", "/media/user/sde1/conversationsLLM")
LOG_FALLBACK_DIR = os.path.expanduser("~/Desktop/fimTester/logs")
LOG_MAX_BYTES = max(4096, _env_int("ROUTER_LOG_MAX_BYTES", 8 * 1024 * 1024))
LOG_SKIP_INFILL = os.environ.get("ROUTER_LOG_SKIP_INFILL", "0").strip().lower() in {"1", "true", "yes"}
LOG_RAW_SSE = os.environ.get("ROUTER_LOG_RAW_SSE", "0").strip().lower() in {"1", "true", "yes"}
LOG_LOCK = threading.Lock()


def parse_json(body: bytes) -> Any:
    if not body:
        return None
    try:
        return json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None


def reasoning_only_chunk(line: bytes) -> dict[str, Any] | None:
    """Parsed SSE chunk when it carries reasoning and no answer/tool call."""
    text = line.strip()
    if not text.startswith(b"data:"):
        return None
    try:
        obj = json.loads(text[5:])
        delta = ((obj.get("choices") or [{}])[0].get("delta")) or {}
    except (UnicodeDecodeError, json.JSONDecodeError, AttributeError, IndexError, TypeError):
        return None
    if not delta.get("reasoning_content") or delta.get("content") or delta.get("tool_calls"):
        return None
    return obj


def _capped(value: Any) -> tuple[Any, bool]:
    if value is None:
        return None, False
    try:
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        text = str(value)
    encoded = text.encode("utf-8", "ignore")
    if len(encoded) <= LOG_MAX_BYTES:
        return value, False
    return encoded[:LOG_MAX_BYTES].decode("utf-8", "ignore") + "\n...[truncated]", True


def raise_token_budget(body: bytes, for_cloud: bool) -> tuple[bytes, dict[str, Any] | None]:
    """Lift a too-small response budget so thinking models can still answer.

    Returns the possibly rewritten body and a note describing the change.
    """
    if MIN_MAX_TOKENS <= 0:
        return body, None
    payload = parse_json(body)
    if not isinstance(payload, dict):
        return body, None
    if "max_tokens" in payload:
        key = "max_tokens"
    elif "max_completion_tokens" in payload:
        key = "max_completion_tokens"
    else:
        key = "max_tokens"
    current = payload.get(key)
    if isinstance(current, bool) or not isinstance(current, int):
        current = None
    if current is None and not for_cloud:
        # local servers pick their own generous default
        return body, None
    if current is not None and current >= MIN_MAX_TOKENS:
        return body, None
    payload[key] = MIN_MAX_TOKENS
    note = {"field": key, "from": current, "to": MIN_MAX_TOKENS}
    return json.dumps(payload).encode("utf-8"), note


def enable_thinking(body: bytes) -> tuple[bytes, dict[str, Any] | None]:
    """Turn the chat template's reasoning mode on unless the client set it."""
    payload = parse_json(body)
    if not isinstance(payload, dict):
        return body, None
    kwargs = payload.get("chat_template_kwargs")
    if not isinstance(kwargs, dict):
        kwargs = {}
        payload["chat_template_kwargs"] = kwargs
    if "enable_thinking" in kwargs:
        return body, None
    kwargs["enable_thinking"] = True
    return json.dumps(payload).encode("utf-8"), {"enable_thinking": True}


def log_directory() -> str:
    """Prefer the external drive; never create it when the drive is unmounted."""
    if os.path.isdir(LOG_DIR) and os.access(LOG_DIR, os.W_OK):
        return LOG_DIR
    parent = os.path.dirname(LOG_DIR.rstrip("/"))
    if parent and os.path.isdir(parent):
        try:
            os.makedirs(LOG_DIR, exist_ok=True)
            if os.access(LOG_DIR, os.W_OK):
                return LOG_DIR
        except OSError:
            pass
    os.makedirs(LOG_FALLBACK_DIR, exist_ok=True)
    return LOG_FALLBACK_DIR


def write_log(record: dict[str, Any]) -> None:
    if LOG_SKIP_INFILL and record.get("path") == "/infill":
        return
    try:
        stamp = datetime.now().astimezone()
        record.setdefault("ts", stamp.isoformat(timespec="milliseconds"))
        directory = log_directory()
        name = f"requests-{stamp:%Y-%m-%d}.jsonl"
        line = json.dumps(record, ensure_ascii=False)
        with LOG_LOCK:
            with open(os.path.join(directory, name), "a", encoding="utf-8") as handle:
                handle.write(line + "\n")
    except Exception as error:
        print(f"llama-router: request log write failed: {error}", file=sys.stderr, flush=True)


class StreamCapture:
    """Accumulate a response so the request log records what was answered."""

    def __init__(self) -> None:
        self.content: list[str] = []
        self.reasoning: list[str] = []
        self.tool_calls: dict[int, dict[str, Any]] = {}
        self.usage: Any = None
        self.finish_reason: Any = None
        self.response_obj: Any = None
        self.raw_sse: list[str] = []
        self.client_gone = False

    def feed(self, line: bytes) -> None:
        if LOG_RAW_SSE:
            self.raw_sse.append(line.decode("utf-8", "replace"))
        text = line.decode("utf-8", "replace").strip()
        if not text.startswith("data:"):
            return
        payload = text[5:].strip()
        if not payload or payload == "[DONE]":
            return
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError:
            return
        if obj.get("usage"):
            self.usage = obj["usage"]
        choices = obj.get("choices") or []
        for choice in choices:
            delta = choice.get("delta") or choice.get("message") or {}
            if delta.get("content"):
                self.content.append(delta["content"])
            if delta.get("reasoning_content"):
                self.reasoning.append(delta["reasoning_content"])
            if choice.get("text"):
                self.content.append(choice["text"])
            for call in delta.get("tool_calls") or []:
                index = call.get("index", 0)
                entry = self.tool_calls.setdefault(index, {"id": None, "name": "", "arguments": ""})
                if call.get("id"):
                    entry["id"] = call["id"]
                function = call.get("function") or {}
                if function.get("name"):
                    entry["name"] += function["name"]
                if function.get("arguments"):
                    entry["arguments"] += function["arguments"]
            if choice.get("finish_reason"):
                self.finish_reason = choice["finish_reason"]
        if not choices and "content" in obj:
            self.content.append(str(obj.get("content") or ""))

    def feed_response(self, data: bytes) -> None:
        if not data:
            return
        try:
            self.response_obj = json.loads(data)
        except (UnicodeDecodeError, json.JSONDecodeError):
            self.response_obj = data.decode("utf-8", "replace")

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        if self.response_obj is not None:
            result["body"] = self.response_obj
        else:
            if self.content:
                result["content"] = "".join(self.content)
            if self.reasoning:
                result["reasoning_content"] = "".join(self.reasoning)
            if self.tool_calls:
                result["tool_calls"] = [self.tool_calls[key] for key in sorted(self.tool_calls)]
            if self.usage is not None:
                result["usage"] = self.usage
            if self.finish_reason is not None:
                result["finish_reason"] = self.finish_reason
        if LOG_RAW_SSE and self.raw_sse:
            result["raw_sse"] = "".join(self.raw_sse)
        if self.client_gone:
            result["client_disconnected"] = True
        return result


class RouterError(RuntimeError):
    pass


class Router:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.state_lock = threading.Lock()
        self.mode = "fast"
        self.big_activity = time.monotonic()
        self.stop_event = threading.Event()
        self.inflight_local = 0
        self.warming = False

    @staticmethod
    def run_systemctl(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        command = [SYSTEMCTL, "--user", *args]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if check and result.returncode:
            raise RouterError(result.stderr.strip() or result.stdout.strip() or str(command))
        return result

    def active(self, unit: str) -> bool:
        return self.run_systemctl("is-active", unit, check=False).stdout.strip() == "active"

    @staticmethod
    def log(message: str) -> None:
        print(f"llama-router: {message}", file=sys.stderr, flush=True)

    # --- cloud overflow / load-through policy -------------------------------

    @staticmethod
    def cloud_enabled() -> bool:
        return bool(DEEPSEEK_KEY)

    def begin_local(self) -> None:
        with self.state_lock:
            self.inflight_local += 1

    def end_local(self) -> None:
        with self.state_lock:
            self.inflight_local = max(0, self.inflight_local - 1)

    def warm_local(self) -> None:
        """Bring the big profile up in the background without blocking anyone."""
        with self.state_lock:
            if self.warming:
                return
            self.warming = True
        threading.Thread(target=self._warm_worker, daemon=True).start()

    def _warm_worker(self) -> None:
        try:
            with self.lock:
                self.ensure_shared_and_big()
            self.log("background warm-up finished; big model is ready")
        except Exception as error:
            self.log(f"background warm-up failed: {error}")
        finally:
            with self.state_lock:
                self.warming = False

    def route_chat(self) -> tuple[str, str]:
        """Choose 'cloud' or 'local' for a chat request, plus the reason."""
        if not self.cloud_enabled():
            return "local", "cloud-disabled"
        with self.state_lock:
            busy = self.inflight_local >= MAX_LOCAL_INFLIGHT
            warming = self.warming
        if busy:
            return "cloud", "local-busy"
        if warming:
            return "cloud", "warming"
        mode = self.mode
        if mode == "fast":
            self.warm_local()
            return "cloud", "cold-start"
        if mode == "switching":
            return "cloud", "switching"
        try:
            sleeping = self.big_is_sleeping()
        except Exception as error:
            self.log(f"status probe failed ({error}); treating the big model as unloaded")
            sleeping = True
        if sleeping:
            self.warm_local()
            deadline = time.monotonic() + LOCAL_WAIT_S
            while time.monotonic() < deadline:
                time.sleep(0.5)
                try:
                    if not self.big_is_sleeping():
                        return "local", "local-woke"
                except Exception:
                    pass
            return "cloud", "strata-idle-unloaded"
        return "local", "local-ready"

    def wait_inactive(self, unit: str, timeout: float = 120.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and not self.stop_event.is_set():
            if not self.active(unit):
                return
            time.sleep(0.5)
        raise RouterError(f"{unit} did not stop within {timeout:.0f}s")

    @staticmethod
    def request(port: int, method: str, path: str, body: bytes = b"",
                headers: dict[str, str] | None = None, timeout: float = 1800.0
                ) -> tuple[int, bytes, dict[str, str]]:
        connection = http.client.HTTPConnection(HOST, port, timeout=timeout)
        try:
            connection.request(method, path, body=body or None, headers=headers or {})
            response = connection.getresponse()
            return response.status, response.read(), {
                name.lower(): value for name, value in response.getheaders()
            }
        finally:
            connection.close()

    def health(self, port: int, timeout: float = 180.0) -> None:
        deadline = time.monotonic() + timeout
        last_error: Exception | None = None
        while time.monotonic() < deadline and not self.stop_event.is_set():
            try:
                status, _, _ = self.request(port, "GET", "/health", timeout=2)
                if status == 200:
                    return
                last_error = RouterError(f"health on {port}: HTTP {status}")
            except Exception as error:
                last_error = error
            time.sleep(0.5)
        raise RouterError(f"server on {port} failed health check: {last_error}")

    def big_is_sleeping(self) -> bool:
        """True when the Strata model is not loaded (its server may be up)."""
        if not self.active(BIG_UNIT):
            return True
        try:
            status, data, _ = self.request(BIG_PORT, "GET", "/props", timeout=4)
            if status == 200:
                props = json.loads(data)
                if "is_sleeping" in props:
                    return bool(props["is_sleeping"])
        except Exception:
            pass
        # older servers: /v1/status "loaded" is the inverse
        status, data, _ = self.request(BIG_PORT, "GET", "/v1/status", timeout=4)
        if status != 200:
            raise RouterError(f"big status returned HTTP {status}")
        return not bool(json.loads(data).get("loaded"))

    def ensure_shared_and_big(self) -> None:
        if (self.mode == "shared" and self.active(SHARED_UNIT)
                and self.active(BIG_UNIT) and not self.big_is_sleeping()):
            return
        self.mode = "switching"
        try:
            # 1) Unload the 3090-only small model so Strata gets the full card.
            self.run_systemctl("stop", FAST_UNIT, check=False)
            self.wait_inactive(FAST_UNIT)
            # 2) Load the 3060 companion (keeps /small/, FIM and autocomplete alive).
            if not self.active(SHARED_UNIT):
                self.run_systemctl("reset-failed", SHARED_UNIT, check=False)
                self.run_systemctl("start", SHARED_UNIT)
            self.health(FAST_PORT)
            # 3) Ensure the Strata server is up, then load its model if it is
            #    unloaded (cold ~2 min, warm ~30 s) so the forwarded request
            #    does not wait behind a silent load.
            if not self.active(BIG_UNIT):
                self.run_systemctl("reset-failed", BIG_UNIT, check=False)
                self.run_systemctl("start", BIG_UNIT)
                self.health(BIG_PORT, timeout=180)
            if self.big_is_sleeping():
                try:
                    self.request(BIG_PORT, "POST", "/load", body=b"{}",
                                 headers={"Content-Type": "application/json"}, timeout=10)
                except Exception:
                    pass
                deadline = time.monotonic() + 300
                while self.big_is_sleeping():
                    if time.monotonic() > deadline:
                        raise RouterError("Strata did not load within 300 s")
                    time.sleep(1.0)
            self.mode = "shared"
            self.big_activity = time.monotonic()
        except Exception:
            self.log("switch to shared failed; rolling back to fast profile")
            self.run_systemctl("stop", BIG_UNIT, check=False)
            self.run_systemctl("stop", SHARED_UNIT, check=False)
            self.wait_inactive(SHARED_UNIT, 120)
            self.run_systemctl("reset-failed", FAST_UNIT, check=False)
            self.run_systemctl("start", FAST_UNIT)
            self.health(FAST_PORT)
            self.mode = "fast"
            raise

    def restore_fast(self) -> None:
        if self.mode != "shared":
            return
        self.mode = "switching"
        try:
            # Keep the Strata server up (model unloaded) so the next big
            # request is a warm reload instead of a cold start.
            self.run_systemctl("stop", SHARED_UNIT, check=False)
            self.wait_inactive(SHARED_UNIT)
            if not self.active(FAST_UNIT):
                self.run_systemctl("reset-failed", FAST_UNIT, check=False)
                self.run_systemctl("start", FAST_UNIT)
            self.health(FAST_PORT)
            self.mode = "fast"
        except Exception:
            self.log("fast restore failed; restarting the 3060 companion profile")
            self.run_systemctl("start", SHARED_UNIT, check=False)
            self.mode = "shared"
            raise

    def initial_setup(self) -> None:
        """Settle the mode indicated by the current units and the big model."""
        with self.lock:
            try:
                # The Strata HTTP server should always be up (its model loads
                # on demand); start it if it is not running.
                if not self.active(BIG_UNIT):
                    self.run_systemctl("reset-failed", BIG_UNIT, check=False)
                    self.run_systemctl("start", BIG_UNIT)
                # Wait for the HTTP server even when the unit is merely "active":
                # at login both units start together and 8080 may still be
                # binding - probing it too early fails the whole setup.
                self.health(BIG_PORT, timeout=180)
                if self.big_is_sleeping():
                    if self.active(SHARED_UNIT):
                        # The router restarted while a big-model session was
                        # running (model loading or idle-unloaded): keep the
                        # 3060 profile and finish the warm-up instead of
                        # yanking the small model back onto the 3090.
                        self.mode = "shared"
                        self.warm_local()
                    else:
                        # Big model resting: the fast 3090-only profile serves 8030.
                        if not self.active(FAST_UNIT):
                            self.run_systemctl("reset-failed", FAST_UNIT, check=False)
                            self.run_systemctl("start", FAST_UNIT)
                            self.health(FAST_PORT, timeout=300)
                        self.mode = "fast"
                else:
                    # Strata is awake: port 8030 must serve the 3060 profile.
                    if not self.active(SHARED_UNIT):
                        self.run_systemctl("reset-failed", SHARED_UNIT, check=False)
                        self.run_systemctl("start", SHARED_UNIT)
                    self.health(FAST_PORT, timeout=300)
                    self.mode = "shared"
                self.log(f"initial mode: {self.mode}")
            except Exception as error:
                self.log(f"initial setup failed: {error}")

    def monitor(self) -> None:
        while not self.stop_event.wait(2.0):
            try:
                with self.lock:
                    if (self.mode == "shared"
                            and time.monotonic() - self.big_activity >= 300
                            and self.big_is_sleeping()):
                        self.log("Strata idle-slept for 5 min; restoring the fast profile")
                        self.restore_fast()
            except Exception as error:
                self.log(f"idle monitor: {error}")


router = Router()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"llama-router {self.client_address[0]} {fmt % args}", file=sys.stderr, flush=True)

    def respond(self, status: int, body: bytes, headers: dict[str, str]) -> None:
        self.send_response(status)
        content_type = headers.get("content-type", "application/json")
        self.send_header("Content-Type", content_type)
        if "text/event-stream" not in content_type.lower():
            self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self._headers_sent = True
        try:
            self.wfile.write(body)
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        self.close_connection = True

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path == "/health":
            self.respond(200, b'{"status":"ok"}', {"content-type": "application/json"})
            return
        small = path == SMALL_PREFIX or path.startswith(SMALL_PREFIX + "/")
        target = self.path[len(SMALL_PREFIX):] if small else self.path
        try:
            with router.lock:
                port = FAST_PORT
                if not small and path in {"/props", "/models", "/v1/models", "/metrics"}:
                    # Report Strata while its model is loaded, the small model otherwise.
                    try:
                        if router.active(BIG_UNIT) and not router.big_is_sleeping():
                            port = BIG_PORT
                    except Exception:
                        pass
                status, body, headers = router.request(port, "GET", target, timeout=8)
            self.respond(status, body, headers)
        except Exception as error:
            self.respond(502, json.dumps({"error": str(error)}).encode(),
                         {"content-type": "application/json"})

    def stream_upstream(self, response: http.client.HTTPResponse, capture: StreamCapture,
                        merge_reasoning: bool = False) -> None:
        """Forward the upstream response to the client without buffering tokens.

        merge_reasoning holds the reasoning deltas back and emits them as one SSE
        event right before the answer: VS Code opens a new collapsible "Thought
        for Xs" block for every read that carries reasoning, so a per-token
        reasoning stream renders as hundreds of fragments.
        """
        content_type = response.getheader("Content-Type", "application/json")
        self.send_response(response.status)
        self.send_header("Content-Type", content_type)
        value = response.getheader("Cache-Control")
        if value:
            self.send_header("Cache-Control", value)
        self.send_header("Connection", "close")
        self.end_headers()
        self._headers_sent = True
        self.close_connection = True
        if "text/event-stream" in content_type.lower():
            template: dict[str, Any] = {}
            pending: list[str] = []
            pending_chars = 0

            def flush_reasoning() -> None:
                nonlocal pending_chars
                text = "".join(pending)
                pending.clear()
                pending_chars = 0
                if not text:
                    return
                event = dict(template)
                event.pop("usage", None)
                event["choices"] = [{"index": 0, "delta": {"reasoning_content": text},
                                     "finish_reason": None}]
                self.wfile.write(("data: " + json.dumps(event) + "\n\n").encode("utf-8"))
                self.wfile.flush()

            while line := response.readline():
                capture.feed(line)
                try:
                    if merge_reasoning:
                        chunk = reasoning_only_chunk(line)
                        if chunk is not None:
                            template.clear()
                            template.update(chunk)
                            piece = chunk["choices"][0]["delta"]["reasoning_content"]
                            pending.append(piece)
                            pending_chars += len(piece)
                            # a blank line ends the block; a very long paragraph is
                            # cut so the thought still shows progress
                            paragraph = "\n\n" in piece or (
                                piece.startswith("\n") and len(pending) > 1
                                and pending[-2].endswith("\n"))
                            if (paragraph and pending_chars >= REASONING_MIN_BLOCK) \
                                    or pending_chars >= REASONING_BLOCK_CHARS:
                                flush_reasoning()
                            continue
                        # SSE blank lines and keep-alive comments separate every
                        # delta: only an answer/tool/finish event ends the thought.
                        if line.lstrip().startswith(b"data:"):
                            flush_reasoning()
                    self.wfile.write(line)
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    capture.client_gone = True
                    return
            if merge_reasoning:
                try:
                    flush_reasoning()
                except (BrokenPipeError, ConnectionResetError):
                    capture.client_gone = True
        else:
            data = response.read()
            capture.feed_response(data)
            try:
                self.wfile.write(data)
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                capture.client_gone = True

    def relay_local(self, path: str, body: bytes, headers: dict[str, str], port: int,
                    capture: StreamCapture) -> int:
        connection = http.client.HTTPConnection(HOST, port, timeout=1800)
        try:
            connection.request("POST", path, body=body or None, headers=headers)
            response = connection.getresponse()
            self.stream_upstream(response, capture, merge_reasoning=True)
            return response.status
        finally:
            connection.close()

    def relay_cloud(self, path: str, body: bytes, capture: StreamCapture) -> tuple[bool, int, bytes]:
        """POST to the DeepSeek API. Returns (relayed, status, error_body)."""
        payload = parse_json(body)
        if isinstance(payload, dict):
            payload["model"] = DEEPSEEK_MODEL
            if DEEPSEEK_EFFORT and not payload.get("reasoning_effort"):
                payload["reasoning_effort"] = DEEPSEEK_EFFORT
            # llama.cpp-only fields the cloud API does not understand
            for key in ("id_slot", "n_cmpl", "n_predict", "cache_prompt", "t_max_prompt_ms",
                        "t_max_predict_ms", "input_prefix", "input_suffix", "input_extra",
                        "n_indent", "top_k", "min_p", "repeat_penalty", "repeat_last_n"):
                payload.pop(key, None)
            data = json.dumps(payload).encode("utf-8")
        else:
            data = body
        try:
            connection = http.client.HTTPSConnection(DEEPSEEK_HOST, 443, timeout=1800)
            try:
                connection.request("POST", path, body=data, headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {DEEPSEEK_KEY}",
                })
                response = connection.getresponse()
                if response.status != 200:
                    return False, response.status, response.read()[:4096]
                self.stream_upstream(response, capture, merge_reasoning=True)
                return True, response.status, b""
            finally:
                connection.close()
        except Exception as error:
            return False, 0, str(error).encode("utf-8", "replace")

    def do_POST(self) -> None:
        started = time.monotonic()
        self._headers_sent = False
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        upstream_path = self.path
        path = self.path.split("?", 1)[0]
        # /ds/... is the explicit DeepSeek endpoint: same API, but the request
        # never goes through route_chat() and never falls back to a local model.
        force_cloud = path.startswith("/ds/")
        if force_cloud:
            path = path[3:]
            upstream_path = upstream_path[3:]
        # /small/... always targets the small local model: strip the prefix and
        # keep the rest of the path for the backend.
        small = path == SMALL_PREFIX or path.startswith(SMALL_PREFIX + "/")
        if small:
            path = path[len(SMALL_PREFIX):] or "/"
            upstream_path = upstream_path[len(SMALL_PREFIX):] or "/"
        if path not in {"/infill", "/v1/completions", "/v1/chat/completions"}:
            self.send_error(404, "unsupported model-router endpoint")
            return
        headers = {key: value for key, value in self.headers.items() if key.lower() not in HOP}
        request_obj = parse_json(body)
        record: dict[str, Any] = {
            "id": uuid.uuid4().hex,
            "client": self.client_address[0],
            "path": self.path.split("?", 1)[0],
            "query": upstream_path.partition("?")[2] or None,
            "forced": "deepseek" if force_cloud else None,
            "model_requested": request_obj.get("model") if isinstance(request_obj, dict) else None,
            "stream": bool(request_obj.get("stream")) if isinstance(request_obj, dict) else False,
            "request": request_obj if request_obj is not None else body.decode("utf-8", "replace"),
        }
        capture = StreamCapture()
        route, reason, status = "local-fast", "fim", None
        if small:
            route, reason = "local-fast", "small"
        model_body = body
        budget_note: dict[str, Any] | None = None
        if path == "/v1/chat/completions":
            model_body, budget_note = raise_token_budget(body, for_cloud=False)
            if small:
                model_body, think_note = enable_thinking(model_body)
                if think_note:
                    budget_note = dict(budget_note or {}, **think_note)
        try:
            if small:
                # general-purpose small model: no profile switch, no cloud overflow
                status = self.relay_local(upstream_path, model_body, headers, FAST_PORT, capture)
            elif path in {"/infill", "/v1/completions"}:
                # FIM / native completions stay on the small model; the proxy
                # provides the chat-based FIM shim the model needs.
                status = self.relay_local(upstream_path, body, headers, FIM_PORT, capture)
            else:
                if force_cloud:
                    route, reason = "cloud", "forced-deepseek"
                else:
                    route, reason = router.route_chat()
                if route == "cloud":
                    cloud_body, cloud_note = raise_token_budget(model_body, for_cloud=True)
                    budget_note = cloud_note or budget_note
                    ok, status, detail = self.relay_cloud(upstream_path, cloud_body, capture)
                    if not ok:
                        record["cloud_error"] = {"status": status,
                                                "body": detail.decode("utf-8", "replace")}
                        if force_cloud:
                            # the /ds endpoint promises DeepSeek: report the error
                            # instead of quietly answering from a local model
                            router.log(f"forced cloud failed ({status}): {detail[:200]!r}")
                            if not self._headers_sent:
                                self.respond(status or 502,
                                             detail or b'{"error":"deepseek unavailable"}',
                                             {"content-type": "application/json"})
                        else:
                            router.log(f"cloud rejected ({status}); falling back to local: {detail[:200]!r}")
                            route, reason = "local", reason + "+cloud-failed"
                if route == "local":
                    with router.lock:
                        router.ensure_shared_and_big()
                    router.begin_local()
                    try:
                        router.big_activity = time.monotonic()
                        status = self.relay_local(upstream_path, model_body, headers, BIG_PORT, capture)
                    finally:
                        router.end_local()
                        router.big_activity = time.monotonic()
        except Exception as error:
            record["error"] = str(error)
            status = status or 502
            if not self._headers_sent:
                self.respond(502, json.dumps({"error": str(error)}).encode(),
                             {"content-type": "application/json"})
        finally:
            record.update({
                "route": route,
                "reason": reason,
                "status": status,
                "duration_s": round(time.monotonic() - started, 3),
                "model_served": DEEPSEEK_MODEL if route == "cloud" else (
                    record.get("model_requested") or "local"),
                "cloud_effort": DEEPSEEK_EFFORT if route == "cloud" else None,
                "budget": budget_note,
                "response": capture.as_dict(),
            })
            request_value, request_big = _capped(record.get("request"))
            response_value, response_big = _capped(record.get("response"))
            record["request"], record["response"] = request_value, response_value
            if request_big or response_big:
                record["truncated"] = True
            write_log(record)


def main() -> None:
    # Log records contain prompts and answers: keep them readable by this user only.
    os.umask(0o077)
    # Listen immediately; settle the profile in the background so a slow model
    # start never keeps port 8013 (and /health) down. Requests that arrive
    # early run through ensure_shared_and_big() themselves.
    threading.Thread(target=router.initial_setup, daemon=True).start()
    threading.Thread(target=router.monitor, daemon=True).start()
    server = ThreadingHTTPServer((BIND_HOST, LISTEN_PORT), Handler)
    server.daemon_threads = True
    print(f"small-model fast/companion switcher listening on {BIND_HOST}:{LISTEN_PORT} "
          f"(cloud={'on:' + DEEPSEEK_MODEL if DEEPSEEK_KEY else 'off'}, log={LOG_DIR})", flush=True)
    try:
        server.serve_forever()
    finally:
        router.stop_event.set()
        server.server_close()


if __name__ == "__main__":
    main()
