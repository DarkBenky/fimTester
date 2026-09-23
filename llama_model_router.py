#!/usr/bin/env python3
"""Switch the 7B FIM server between fast and 27B-coexistence profiles.

Policy:
- Fast profile (default): 7B on the RTX 3090 only; the 27B is stopped.
- A 27B request on 8013 first stops the fast 7B, starts the shared-GPU 7B
  profile, starts (or keeps) the 27B, then forwards the request.
- The 27B process stays alive and idle-sleeps; when it has slept for five
  minutes the router restores the fast 3090-only 7B profile.
"""

from __future__ import annotations

import http.client
import json
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

HOST = "127.0.0.1"
LISTEN_PORT = 8013
FAST_PORT = 8030
BIG_PORT = 8014
FAST_UNIT = "llama-server-zeta.service"
SHARED_UNIT = "llama-server-zeta-shared.service"
BIG_UNIT = "llama-server-qwen38.service"
SYSTEMCTL = "/usr/bin/systemctl"

HOP = {
    "host", "connection", "keep-alive", "transfer-encoding", "upgrade",
    "proxy-connection", "te", "trailer", "content-length", "accept-encoding",
}


class RouterError(RuntimeError):
    pass


class Router:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.mode = "fast"
        self.big_activity = time.monotonic()
        self.stop_event = threading.Event()

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
        if not self.active(BIG_UNIT):
            return True
        status, data, _ = self.request(BIG_PORT, "GET", "/props", timeout=4)
        if status != 200:
            raise RouterError(f"big /props returned HTTP {status}")
        return bool(json.loads(data).get("is_sleeping"))

    def ensure_shared_and_big(self) -> None:
        if (self.mode == "shared" and self.active(SHARED_UNIT)
                and self.active(BIG_UNIT)):
            return
        self.mode = "switching"
        try:
            # 1) Unload the 3090-only 7B so the large model has full VRAM.
            self.run_systemctl("stop", FAST_UNIT, check=False)
            self.wait_inactive(FAST_UNIT)
            # 2) Load the shared-GPU 7B profile (needed while the 27B runs).
            if not self.active(SHARED_UNIT):
                self.run_systemctl("reset-failed", SHARED_UNIT, check=False)
                self.run_systemctl("start", SHARED_UNIT)
            self.health(FAST_PORT)
            # 3) Start the 27B if it is not running; a sleeping instance just
            #    wakes on the forwarded request (~10 s), so only a cold start
            #    needs the health wait.
            if not self.active(BIG_UNIT):
                self.run_systemctl("reset-failed", BIG_UNIT, check=False)
                self.run_systemctl("start", BIG_UNIT)
                self.health(BIG_PORT, timeout=300)
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
            # Keep the 27B process alive (it is idle-sleeping, ~1 GB VRAM) so
            # the next agent request wakes in ~10 s instead of a cold load.
            self.run_systemctl("stop", SHARED_UNIT, check=False)
            self.wait_inactive(SHARED_UNIT)
            if not self.active(FAST_UNIT):
                self.run_systemctl("reset-failed", FAST_UNIT, check=False)
                self.run_systemctl("start", FAST_UNIT)
            self.health(FAST_PORT)
            self.mode = "fast"
        except Exception:
            self.log("fast restore failed; restarting shared 7B profile")
            self.run_systemctl("start", SHARED_UNIT, check=False)
            self.mode = "shared"
            raise

    def monitor(self) -> None:
        while not self.stop_event.wait(2.0):
            try:
                with self.lock:
                    if (self.mode == "shared"
                            and self.active(BIG_UNIT)
                            and time.monotonic() - self.big_activity >= 300
                            and self.big_is_sleeping()):
                        self.log("27B idle-slept for 5 min; restoring fast 7B")
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
        try:
            self.wfile.write(body)
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        self.close_connection = True

    def do_GET(self) -> None:
        if self.path.split("?", 1)[0] == "/health":
            self.respond(200, b'{"status":"ok"}', {"content-type": "application/json"})
            return
        try:
            with router.lock:
                port = FAST_PORT
                if self.path.split("?", 1)[0] in {"/props", "/models", "/v1/models", "/metrics"}:
                    # Metadata reads are never woken by GET requests; report the
                    # 27B while it exists, otherwise the 7B.
                    if router.active(BIG_UNIT):
                        port = BIG_PORT
                status, body, headers = router.request(port, "GET", self.path, timeout=8)
            self.respond(status, body, headers)
        except Exception as error:
            self.respond(502, json.dumps({"error": str(error)}).encode(),
                         {"content-type": "application/json"})

    def relay_post(self, path: str, body: bytes, headers: dict[str, str]) -> None:
        """Stream the 27B response without buffering tokens."""
        connection = http.client.HTTPConnection(HOST, BIG_PORT, timeout=1800)
        try:
            connection.request("POST", path, body=body or None, headers=headers)
            response = connection.getresponse()
            content_type = response.getheader("Content-Type", "application/json")
            self.send_response(response.status)
            self.send_header("Content-Type", content_type)
            for name in ("Cache-Control",):
                value = response.getheader(name)
                if value:
                    self.send_header(name, value)
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            if "text/event-stream" in content_type.lower():
                while line := response.readline():
                    self.wfile.write(line)
                    self.wfile.flush()
            else:
                data = response.read()
                self.wfile.write(data)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            connection.close()

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        path = self.path.split("?", 1)[0]
        if path not in {"/infill", "/v1/completions", "/v1/chat/completions"}:
            self.send_error(404, "unsupported model-router endpoint")
            return
        headers = {key: value for key, value in self.headers.items() if key.lower() not in HOP}
        try:
            with router.lock:
                router.ensure_shared_and_big()
                router.big_activity = time.monotonic()
            self.relay_post(self.path, body, headers)
            with router.lock:
                router.big_activity = time.monotonic()
        except Exception as error:
            self.respond(502, json.dumps({"error": str(error)}).encode(),
                         {"content-type": "application/json"})


def main() -> None:
    # Detect the current 7B profile from systemd; default mode is fast.
    if router.active(SHARED_UNIT):
        router.mode = "shared"
    elif router.active(FAST_UNIT):
        router.mode = "fast"
        router.health(FAST_PORT, timeout=300)
    else:
        router.run_systemctl("reset-failed", FAST_UNIT, check=False)
        router.run_systemctl("start", FAST_UNIT)
        router.health(FAST_PORT, timeout=300)
        router.mode = "fast"
    threading.Thread(target=router.monitor, daemon=True).start()
    server = ThreadingHTTPServer((HOST, LISTEN_PORT), Handler)
    server.daemon_threads = True
    print(f"Qwen2.5-Coder fast/shared switcher listening on {HOST}:{LISTEN_PORT}", flush=True)
    try:
        server.serve_forever()
    finally:
        router.stop_event.set()
        server.server_close()


if __name__ == "__main__":
    main()
