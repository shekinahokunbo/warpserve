"""Load generator that runs INSIDE the cluster, hitting the Service directly.

Why this exists: driving load through `kubectl port-forward` measures the
port-forward. It is a single TCP tunnel proxied by the API server, and it caps
throughput long before the pods saturate — so adding replicas appears to do
nothing. Running the client in-cluster removes that bottleneck and lets the
Service load-balance across every endpoint, which is what we actually want to
measure.

Runs as a Pod with the same image as the service (numpy + Pillow already there).
"""

import io
import os
import statistics
import threading
import time
import urllib.request
import uuid

import numpy as np
from PIL import Image

URL = os.getenv("TARGET_URL", "http://warpserve/infer")
WORKERS = int(os.getenv("WORKERS", "16"))
SECONDS = int(os.getenv("SECONDS", "45"))

_lat: list[float] = []
_errors = 0
_lock = threading.Lock()
_stop = threading.Event()


def make_frame() -> bytes:
    rng = np.random.default_rng(0)
    arr = (rng.random((1080, 1920, 3)) * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def build_multipart(image_bytes: bytes):
    boundary = uuid.uuid4().hex
    buf = io.BytesIO()
    buf.write(f"--{boundary}\r\n".encode())
    buf.write(b'Content-Disposition: form-data; name="file"; filename="f.jpg"\r\n')
    buf.write(b"Content-Type: image/jpeg\r\n\r\n")
    buf.write(image_bytes)
    buf.write(f"\r\n--{boundary}--\r\n".encode())
    return buf.getvalue(), f"multipart/form-data; boundary={boundary}"


def worker(body: bytes, ctype: str):
    global _errors
    while not _stop.is_set():
        req = urllib.request.Request(URL, data=body, headers={"Content-Type": ctype})
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                r.read()
            with _lock:
                _lat.append((time.perf_counter() - t0) * 1000.0)
        except Exception:
            with _lock:
                _errors += 1


def pct(s, p):
    idx = max(0, min(len(s) - 1, int(len(s) * p / 100) - 1))
    return s[idx]


def main():
    body, ctype = build_multipart(make_frame())
    threads = [threading.Thread(target=worker, args=(body, ctype), daemon=True)
               for _ in range(WORKERS)]
    start = time.perf_counter()
    for t in threads:
        t.start()
    time.sleep(SECONDS)
    _stop.set()
    for t in threads:
        t.join(timeout=35)
    wall = time.perf_counter() - start

    lat = sorted(_lat)
    if not lat:
        print(f"RESULT errors={_errors} no successful requests")
        return
    print(f"RESULT requests={len(lat)} errors={_errors} "
          f"rps={len(lat)/wall:.1f} "
          f"mean={statistics.mean(lat):.0f} "
          f"p50={pct(lat,50):.0f} p95={pct(lat,95):.0f} p99={pct(lat,99):.0f}")


if __name__ == "__main__":
    main()
