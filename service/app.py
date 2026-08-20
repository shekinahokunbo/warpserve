"""WarpServe — GPU preprocessing microservice.

Accepts an image, runs the resize + normalize + HWC->CHW step that sits in front
of a vision model, and returns the resulting tensor's shape and statistics along
with a timing breakdown.

Two execution paths, one API:
  * CUDA  — loads libpreprocess.so (built from kernels/preprocess.cu) via ctypes
  * NumPy — identical math in pure Python, used when no GPU is present

The fallback is what lets the same image and the same Kubernetes manifests run
on a laptop with no NVIDIA GPU.
"""

import ctypes
import io
import os
import threading
import time
from collections import deque

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

DST_H = DST_W = 640          # YOLO's input size
MAX_UPLOAD_BYTES = 32 << 20  # 32 MB — guardrail, not a limit anyone should hit

app = FastAPI(title="WarpServe", version="0.1.0")

_lib = None                  # ctypes handle, or None on the NumPy path
_backend = "numpy"
_lock = threading.Lock()     # the .so is not reentrant; serialize calls into it
_latencies: deque[float] = deque(maxlen=1000)
_requests = 0


# --------------------------------------------------------------- kernel loading

def _load_kernel():
    """Return a ctypes handle to the CUDA kernel, or None to use NumPy."""
    if os.getenv("WARPSERVE_FORCE_CPU") == "1":
        return None

    path = os.getenv("WARPSERVE_KERNEL_PATH")
    if not path or not os.path.exists(path):
        return None

    lib = ctypes.CDLL(path)
    # Declaring argtypes matters: without it ctypes guesses, and a pointer
    # truncated to 32 bits is a segfault rather than an error message.
    lib.warpserve_preprocess.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.uint8, flags="C_CONTIGUOUS"),
        ctypes.c_int, ctypes.c_int,
        np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
        ctypes.c_int, ctypes.c_int,
    ]
    lib.warpserve_preprocess.restype = ctypes.c_int
    return lib


@app.on_event("startup")
def _startup():
    global _lib, _backend
    try:
        _lib = _load_kernel()
    except OSError as exc:                       # .so present but unloadable
        print(f"[warpserve] kernel load failed, falling back to NumPy: {exc}")
        _lib = None
    _backend = "cuda" if _lib else "numpy"
    print(f"[warpserve] backend={_backend}")


# ------------------------------------------------------------------ preprocess

def _preprocess_numpy(src: np.ndarray) -> np.ndarray:
    """Bilinear resize + normalize + HWC->CHW. Mirrors preprocess.cu exactly,
    including the half-pixel centre correction, so both paths agree."""
    src_h, src_w, _ = src.shape
    scale_y = src_h / DST_H
    scale_x = src_w / DST_W

    fy = (np.arange(DST_H, dtype=np.float32) + 0.5) * scale_y - 0.5
    fx = (np.arange(DST_W, dtype=np.float32) + 0.5) * scale_x - 0.5

    y0 = np.floor(fy).astype(np.int32)
    x0 = np.floor(fx).astype(np.int32)
    dy = (fy - y0).astype(np.float32)[:, None, None]
    dx = (fx - x0).astype(np.float32)[None, :, None]

    y1 = np.clip(y0 + 1, 0, src_h - 1)
    x1 = np.clip(x0 + 1, 0, src_w - 1)
    y0 = np.clip(y0, 0, src_h - 1)
    x0 = np.clip(x0, 0, src_w - 1)

    s = src.astype(np.float32)
    v00 = s[np.ix_(y0, x0)]
    v01 = s[np.ix_(y0, x1)]
    v10 = s[np.ix_(y1, x0)]
    v11 = s[np.ix_(y1, x1)]

    top = v00 + (v01 - v00) * dx
    bot = v10 + (v11 - v10) * dx
    out = (top + (bot - top) * dy) / 255.0

    return np.ascontiguousarray(out.transpose(2, 0, 1), dtype=np.float32)


def _preprocess_cuda(src: np.ndarray) -> np.ndarray:
    src_h, src_w, _ = src.shape
    dst = np.empty((3, DST_H, DST_W), dtype=np.float32)
    src = np.ascontiguousarray(src, dtype=np.uint8)

    with _lock:
        rc = _lib.warpserve_preprocess(src, src_h, src_w, dst, DST_H, DST_W)

    if rc != 0:
        raise RuntimeError(f"CUDA preprocessing failed (code {rc})")
    return dst


# ------------------------------------------------------------------- endpoints

@app.post("/infer")
async def infer(file: UploadFile = File(...)):
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="empty upload")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="image too large")

    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="not a decodable image")

    src = np.asarray(img, dtype=np.uint8)

    t0 = time.perf_counter()
    tensor = _preprocess_cuda(src) if _lib else _preprocess_numpy(src)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    global _requests
    _requests += 1
    _latencies.append(elapsed_ms)

    return {
        "backend": _backend,
        "input_shape": list(src.shape),
        "output_shape": list(tensor.shape),
        "preprocess_ms": round(elapsed_ms, 3),
        "tensor_min": float(tensor.min()),
        "tensor_max": float(tensor.max()),
        "tensor_mean": round(float(tensor.mean()), 6),
    }


@app.get("/healthz")
def healthz():
    """Liveness: the process is up. If this fails, Kubernetes restarts the pod."""
    return {"status": "ok"}


@app.get("/readyz")
def readyz():
    """Readiness: we can actually serve. Kubernetes keeps traffic away until
    this passes, which is what makes a rolling update zero-downtime."""
    return {"status": "ready", "backend": _backend}


@app.get("/metrics")
def metrics():
    lat = sorted(_latencies)
    if not lat:
        return {"requests": 0, "backend": _backend}

    def pct(p: float) -> float:
        # Nearest-rank percentile: index = ceil(p/100 * n) - 1.
        idx = max(0, min(len(lat) - 1, int(np.ceil(p / 100 * len(lat))) - 1))
        return round(lat[idx], 3)

    return {
        "requests": _requests,
        "backend": _backend,
        "window": len(lat),
        "p50_ms": pct(50),
        "p95_ms": pct(95),
        "p99_ms": pct(99),
        "throughput_rps": round(1000.0 / (sum(lat) / len(lat)), 2),
    }
