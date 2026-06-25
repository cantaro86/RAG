#!/usr/bin/env python3
import re
import socket
import subprocess
import sys
import time
from threading import Thread

port = int(sys.argv[1])

for _ in range(120):
    s = socket.socket()
    s.settimeout(1)
    try:
        s.connect(("127.0.0.1", port))
        s.close()
        break
    except Exception:
        s.close()
        time.sleep(2)
else:
    print(f"Gradio did not start on 127.0.0.1:{port}", file=sys.stderr)
    sys.exit(1)

proc = subprocess.Popen(
    ["cloudflared", "tunnel", "--url", f"http://127.0.0.1:{port}"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
)

url = None
pat = re.compile(r"https://[-a-zA-Z0-9]+\.trycloudflare\.com")


def forward():
    global url
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        if url is None:
            m = pat.search(line)
            if m:
                url = m.group(0)
                print(f"\nCloudflare public URL: {url}\n", flush=True)


Thread(target=forward, daemon=True).start()

try:
    while proc.poll() is None:
        time.sleep(1)
finally:
    if proc.poll() is None:
        proc.terminate()
