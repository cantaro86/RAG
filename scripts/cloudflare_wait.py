#!/usr/bin/env python3
"""Wait for Gradio and expose it through a Cloudflare quick tunnel.

This helper is used by ``rag_gradio.sbatch`` in two modes. Before Gradio starts,
``--check-port-free`` verifies that the selected loopback port is available.
After launch, the helper receives the application PID and performs three tasks:

1. Confirm that the tracked process remains alive while polling Gradio's
   ``/config`` endpoint.
2. Start ``cloudflared``, forward its output, and highlight the generated
   ``trycloudflare.com`` public URL.
3. On normal exit or interruption, terminate ``cloudflared`` and kill it if it
   does not stop within five seconds.

The batch script remains responsible for the lifetime of the Gradio process;
this helper owns only readiness verification and the Cloudflare child process.
"""

import argparse
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
from urllib.request import urlopen

URL_PATTERN = re.compile(r"https://[-a-zA-Z0-9]+\.trycloudflare\.com")


def process_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        try:
            listener.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def gradio_is_ready(port: int) -> bool:
    try:
        with urlopen(f"http://127.0.0.1:{port}/config", timeout=1) as response:
            payload = json.load(response)
    except OSError, ValueError:
        return False
    return (
        isinstance(payload, dict)
        and isinstance(payload.get("version"), str)
        and isinstance(payload.get("components"), list)
    )


def wait_for_gradio(port: int, app_pid: int, timeout: float = 240) -> bool:
    deadline = time.monotonic() + timeout
    while process_is_running(app_pid) and time.monotonic() < deadline:
        if gradio_is_ready(port) and process_is_running(app_pid):
            return True
        time.sleep(0.25)
    message = "Gradio stopped before startup" if not process_is_running(app_pid) else "Gradio startup timed out"
    print(message, file=sys.stderr)
    return False


def run_tunnel(port: int) -> int:
    try:
        proc = subprocess.Popen(
            ["cloudflared", "tunnel", "--url", f"http://127.0.0.1:{port}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except OSError as exc:
        print(f"Failed to start cloudflared: {exc}", file=sys.stderr)
        return 127 if isinstance(exc, FileNotFoundError) else 1

    found_url = False
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            if match := URL_PATTERN.search(line):
                found_url = True
                print(f"\nCloudflare public URL: {match.group(0)}\n", flush=True)
        returncode = proc.wait()
        status = returncode if returncode >= 0 else 128 - returncode
        if not found_url:
            print("cloudflared exited before publishing a public URL", file=sys.stderr)
            return status or 1
        return status
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Wait for Gradio, then run a Cloudflare quick tunnel")
    parser.add_argument("port", type=int)
    parser.add_argument("--app-pid", type=int)
    parser.add_argument("--check-port-free", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    if args.check_port_free:
        if port_is_free(args.port):
            return 0
        print(f"Port 127.0.0.1:{args.port} is already occupied", file=sys.stderr)
        return 1
    if args.app_pid is None or args.app_pid <= 0:
        parser.error("--app-pid must be a positive integer")
    signal.signal(signal.SIGTERM, lambda signum, _frame: sys.exit(128 + signum))
    if not wait_for_gradio(args.port, args.app_pid):
        return 1
    return run_tunnel(args.port)


if __name__ == "__main__":
    raise SystemExit(main())
