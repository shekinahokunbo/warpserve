"""Drive load at WarpServe and report latency percentiles.

Usage:
    python bench/loadtest.py --url http://localhost:8080 --workers 8 --seconds 60

Reports p50/p95/p99 and throughput. Pair the output with `kubectl get hpa -w`
to build the latency-vs-replica-count table — that table, not the YAML, is what
makes the Kubernetes milestone worth putting on a resume.

Stdlib only, so it runs anywhere without installing anything.
"""

import argparse
import io
import statistics
import threading
import time
import urllib.request
import uuid

_latencies: list[float] = []
_errors = 0
_lock = threading.Lock()
_stop = threading.Event()


def build_multipart(image_bytes: bytes, filename: str = "frame.jpg"):
    """Hand-rolled multipart/form-data so this needs no third-party HTTP lib."""
    boundary = uuid.uuid4().hex
    buf = io.BytesIO()
    buf.write(f"--{boundary}\r\n".encode())
    buf.write(
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'.encode()
    )
    buf.write(b"Content-Type: image/jpeg\r\n\r\n")
    buf.write(image_bytes)
    buf.write(f"\r\n--{boundary}--\r\n".encode())
    return buf.getvalue(), f"multipart/form-data; boundary={boundary}"


def worker(url: str, body: bytes, content_type: str):
    global _errors
    while not _stop.is_set():
        req = urllib.request.Request(
            f"{url}/infer", data=body, headers={"Content-Type": content_type}
        )
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                resp.read()
            elapsed = (time.perf_counter() - t0) * 1000.0
            with _lock:
                _latencies.append(elapsed)
        except Exception:
            with _lock:
                _errors += 1


def pct(values, p):
    if not values:
        return 0.0
    s = sorted(values)
    idx = max(0, min(len(s) - 1, int(len(s) * p / 100) - 1))
    return s[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8080")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seconds", type=int, default=60)
    ap.add_argument("--image", default="bench/frame.jpg")
    args = ap.parse_args()

    with open(args.image, "rb") as fh:
        image_bytes = fh.read()
    body, content_type = build_multipart(image_bytes)

    print(f"  {args.workers} workers -> {args.url}/infer for {args.seconds}s "
          f"({len(image_bytes)/1e6:.1f} MB per request)")

    threads = [
        threading.Thread(target=worker, args=(args.url, body, content_type), daemon=True)
        for _ in range(args.workers)
    ]
    start = time.perf_counter()
    for t in threads:
        t.start()

    time.sleep(args.seconds)
    _stop.set()
    for t in threads:
        t.join(timeout=35)
    wall = time.perf_counter() - start

    with _lock:
        lat, errs = list(_latencies), _errors

    if not lat:
        print(f"  no successful requests ({errs} errors)")
        return

    print(f"\n  requests    {len(lat)}   errors {errs}")
    print(f"  throughput  {len(lat)/wall:.1f} req/s")
    print(f"  mean        {statistics.mean(lat):.1f} ms")
    print(f"  p50         {pct(lat, 50):.1f} ms")
    print(f"  p95         {pct(lat, 95):.1f} ms")
    print(f"  p99         {pct(lat, 99):.1f} ms\n")


if __name__ == "__main__":
    main()
