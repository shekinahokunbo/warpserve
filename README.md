# WarpServe — a GPU inference microservice, from CUDA kernel to Kubernetes

A vision inference service where **the preprocessing step is a hand-written CUDA
kernel**, packaged as a multi-stage container, and orchestrated on Kubernetes
with autoscaling under load.

The premise: in a real serving stack, the model isn't always the bottleneck —
the CPU-side preprocessing in front of it often is, leaving the GPU idle. This
project measures that claim instead of asserting it, then fixes it on the GPU,
then answers the next question a serving team actually has: what happens when
traffic spikes?

**Every layer is benchmarked, and every speedup is checked for correctness
against a scalar reference implementation.** A kernel that's fast and wrong is
not fast.

---

## Why this project

It closes three specific gaps in one coherent artifact rather than three
tutorials: **CUDA**, **Docker**, and **Kubernetes** — each doing real work, each
producing a number you can defend in an interview.

---

## Architecture

```
   HTTP request (image)
          |
          v
   FastAPI service  ---------------------------+
          |                                    |
          v                                    v
   preprocess:                          GPU absent?
   CUDA kernel via ctypes               NumPy fallback path
   (resize + normalize + HWC->CHW)      (same API, same manifests)
          |
          v
   model inference (ONNX Runtime / TensorRT)
          |
          v
   detections JSON  +  /metrics (latency, throughput)

   packaged by:  docker/Dockerfile.gpu  (multi-stage, nvcc never ships)
                 docker/Dockerfile.cpu  (runs on Apple Silicon + kind)
   orchestrated: k8s/deployment.yaml + service.yaml + hpa.yaml
```

The CPU fallback isn't a compromise, it's the design. It's what lets you develop
and prove the orchestration layer locally on a Mac with no NVIDIA GPU, and deploy
the GPU image through the identical manifests.

---

## Milestones

Each one is independently finishable and independently resume-able. Stop at any
point and you still have something true to say.

### Milestone 1 — CUDA kernel (~2 hours) · do this one first

You have no NVIDIA GPU locally, so this runs on **Google Colab's free T4**
(Runtime → Change runtime type → T4 GPU), or on the RTX 6000 from SURF if you
still have access.

```
!nvidia-smi
!nvcc -O3 -arch=sm_75 preprocess.cu -o preprocess && ./preprocess
!nvcc -O3 -arch=sm_75 matmul.cu -o matmul && ./matmul
```

`kernels/preprocess.cu` is the one that powers the service: bilinear resize +
normalize + HWC→CHW in a single pass, one thread per output pixel, benchmarked
against a scalar CPU reference with a max-absolute-error assertion.

It reports **two** GPU timings on purpose — kernel-only, and end-to-end
including host↔device transfers. Read both. The gap between them is the single
most important lesson in GPU programming, and it's the thing that decides
whether moving a step to the GPU actually helps.

`kernels/matmul.cu` is the interview kernel: naive vs. shared-memory tiled,
identical arithmetic, different memory strategy. Be able to explain why the
tiled version wins and what each `__syncthreads()` is protecting.

### Milestone 2 — service + Docker (~2 hours)

Write `service/app.py` yourself — this is the part you've already done in
EdgePulse. The contract:

| Endpoint | Does |
|---|---|
| `POST /infer` | image in → detections + timing breakdown out |
| `GET /healthz` | liveness: process is alive |
| `GET /readyz` | readiness: model loaded, kernel resolved |
| `GET /metrics` | count, throughput, p50 / p95 / p99 latency |

Load the kernel exactly the way EdgePulse loads its C++ core:

```python
lib = ctypes.CDLL(os.environ["WARPSERVE_KERNEL_PATH"])
```

...and fall back to NumPy when that path is missing or `WARPSERVE_FORCE_CPU=1`.
Split `/healthz` from `/readyz` deliberately — that distinction is what makes
rolling updates zero-downtime, and interviewers ask about it.

Then:

```bash
docker build -f docker/Dockerfile.cpu -t warpserve:cpu .
docker run -p 8000:8000 warpserve:cpu
docker images warpserve    # record the size — you'll compare against GPU
```

Record the multi-stage size win (devel image vs. runtime image). It's a real,
quotable number.

### Milestone 3 — Kubernetes (~2 hours)

```bash
brew install kind kubectl
kind create cluster --name warpserve
kind load docker-image warpserve:cpu --name warpserve   # no registry needed
kubectl apply -f k8s/deployment.yaml -f k8s/service.yaml
kubectl get pods -w
kubectl port-forward svc/warpserve 8080:80
```

Then install metrics-server (see the comment block in `k8s/hpa.yaml`), apply the
HPA, and drive load at it:

```bash
kubectl apply -f k8s/hpa.yaml
kubectl get hpa -w        # watch replicas climb
```

**The deliverable is the graph, not the YAML.** Load the service until the HPA
scales 2 → N pods, and record p50/p95/p99 latency and throughput at each replica
count. That table is what makes this a systems project instead of a tutorial.

### Milestone 4 — stretch, if you want it

Swap ONNX Runtime for **TensorRT** and compare FP32 vs. INT8 — you already did
exactly this analysis in your SURF work, so it costs you little and ties the
whole portfolio together.

---

## What to be able to explain

If you can't answer these, the project won't survive an interview — which is the
whole point of building it:

1. What is a warp, and why does 32 matter?
2. Why is shared memory faster than global memory, and what is a bank conflict?
3. In `preprocess.cu`, why does the end-to-end time differ so much from the
   kernel time — and when does that make GPU preprocessing *not* worth it?
4. What are the two `__syncthreads()` in the tiled matmul each protecting?
   What breaks if you delete the second one?
5. Why multi-stage Docker? What's in the devel image you don't want to ship?
6. Liveness vs. readiness probe — what does each one do when it fails?
7. What is the HPA measuring, and why does `resources.requests` have to be set
   for it to work at all?

---

## Honest reporting

Two things this repo will not pretend:

- **The GPU path is validated on Colab/T4, not in the local cluster.** A Mac has
  no NVIDIA GPU, so the k8s milestone runs the CPU image. The GPU manifest is
  documented in `k8s/deployment.yaml` but untested. Say exactly that in the
  README when you publish — a reviewer who catches an unstated gap discounts
  everything else you claimed.
- **Every speedup here is paired with a correctness check** against a scalar
  reference. That's the same standard as MarketLab (engines asserted numerically
  equal) and Stocks.io (look-ahead regression tests). Keep it.

---

## Results (measured)

Hardware: NVIDIA Tesla T4 (Colab), CUDA 12.8, `-O3 -arch=sm_75`.

### Preprocessing kernel — 3840x2160 -> 640x640, 100 iterations

| Path | Latency | Speedup |
|---|---|---|
| CPU (scalar reference) | 10.683 ms | 1x |
| GPU, kernel only | **0.084 ms** | **127.5x** |
| GPU, incl. H2D + D2H transfer | 6.176 ms | 1.7x |

Max absolute error vs. CPU reference: **0.000e+00** — bit-identical, not merely close.

**The finding is the gap between rows 2 and 3.** The compute is 127x faster; the
end-to-end win is 1.7x. Transfer accounts for 6.09 ms of the 6.18 ms — moving
~28 MB across PCIe at roughly 4.9 GB/s, which is what pageable host memory gets
on this link. Compute is not the bottleneck here; the bus is.

That result argues for exactly what production inference stacks do: keep the
frame resident on the device and fuse the whole pipeline, rather than round-
tripping to host memory between stages. Pinned memory (`cudaHostAlloc`) would
roughly double transfer bandwidth and is the obvious next experiment.

### Matrix multiply — 1024x1024, 50 iterations, TILE=16

| Implementation | Latency | Throughput | vs. CPU |
|---|---|---|---|
| CPU (scalar) | 3176.543 ms | 0.68 GFLOP/s | 1x |
| GPU naive (global memory) | 5.221 ms | 411.35 GFLOP/s | 608.5x |
| GPU tiled (shared memory) | **2.646 ms** | **811.67 GFLOP/s** | **1200.6x** |

Shared-memory tiling alone is worth **1.97x** over the naive kernel — identical
arithmetic, purely a memory-locality result.

Max absolute error vs. CPU: 9.155e-05 for both GPU kernels (float accumulation
order differs from the CPU's vectorized reduction; the two GPU versions agree
with each other exactly, since both accumulate in the same k order).

**Honest ceiling:** 812 GFLOP/s is about 10% of the T4's ~8.1 TFLOP/s FP32 peak.
Shared-memory tiling is the first optimization, not the last — register
blocking, vectorized loads, and wider tiles are what close the remaining gap,
and cuBLAS reaches several TFLOP/s on this same problem.

### Milestone 2 — containerized service (measured)

`docker build -f docker/Dockerfile.cpu -t warpserve:cpu .` -> **408 MB**, runs as
a non-root user (uid 10001), with a `HEALTHCHECK` and all four endpoints live:

| Endpoint | Response |
|---|---|
| `GET /healthz` | `{"status":"ok"}` |
| `GET /readyz` | `{"status":"ready","backend":"numpy"}` |
| `POST /infer` | 3840x2160 JPEG in -> `[3,640,640]` float32 tensor, 29.3 ms |
| `GET /metrics` | p50 18.3 ms, p95 29.3 ms, ~49 req/s over a 6-request window |

The NumPy fallback reproduces the kernel's math exactly, including the
half-pixel centre correction, so both backends return the same tensor.

**Untested:** the GPU image (`docker/Dockerfile.gpu`). Building and running it
needs an NVIDIA GPU and the container toolkit; this was developed on Apple
Silicon, so that path is written and reviewed but has not been executed.


### Milestone 3 — Kubernetes (measured)

Local `kind` cluster, CPU image, 1080p JPEG per request, 24 concurrent clients.

**First finding: the measurement harness was the bottleneck.** Driving load
through `kubectl port-forward` gave ~12.5 req/s whether the Deployment had 2
pods or 8 — adding replicas changed nothing. `port-forward` is a single TCP
tunnel proxied by the API server; it saturates long before the pods do. Moving
the load generator *into* the cluster (`bench/incluster_load.py`, hitting the
Service directly so it load-balances across endpoints) is what made the pods
the bottleneck instead of the client.

**Latency vs. replica count**, load generated in-cluster, 45 s per run:

| Replicas | Throughput | p50 | p95 | p99 |
|---|---|---|---|---|
| 2 | 26.5 req/s | 907 ms | 1694 ms | 1783 ms |
| 4 | 47.4 req/s | 334 ms | 1387 ms | 1654 ms |
| 8 | **88.0 req/s** | **201 ms** | **735 ms** | 1094 ms |

4x the pods yields 3.3x the throughput — **83% scaling efficiency** — while p50
drops 4.5x. Sub-linear, as expected: all pods share one kind node, so they
contend for the same host CPU.

**Autoscaling, unattended.** With the HPA applied (CPU target 60% of a 200m
request, min 2 / max 8) and load applied at t=0:

| t | desired | ready |
|---|---|---|
| 15 s | 2 | 2 |
| 45 s | 6 | 2 |
| 60 s | 8 | 6 |
| 75 s | 8 | 8 |

**Second finding: autoscaling is not instant.** It took ~45 s before the HPA
even raised its target and ~75 s to have 8 pods serving — metrics-server scrape
interval, then the HPA sync period, then pod startup, stacked in series. A 45 s
traffic burst is therefore served almost entirely at the *old* capacity: that
run reported 26.2 req/s, matching the 2-replica row above rather than the
8-replica one. Autoscaling handles sustained load shifts, not spikes; spikes
need headroom in `minReplicas` or a faster signal than CPU.

### GPU image (build verified, runtime not)

`docker build --platform linux/amd64 -f docker/Dockerfile.gpu` succeeds under
emulation: `nvcc -O3 -arch=sm_75 --shared -Xcompiler -fPIC` produces
`libpreprocess.so` in the build stage and the runtime stage copies it in.

| Image | Size |
|---|---|
| `nvidia/cuda:12.4.1-devel-ubuntu22.04` (build stage base) | 11.4 GB |
| `warpserve:gpu` (shipped runtime image) | **1.52 GB** |
| `warpserve:cpu` | 408 MB |

Multi-stage keeps the CUDA toolkit out of the shipped artifact: **7.5x smaller,
~9.9 GB saved**. Compiling requires no GPU, so this is fully verified — but the
image has never been *executed*, which needs an NVIDIA GPU and the container
toolkit. That remains the one unvalidated path in this repo.


---

## Status

- [x] `kernels/preprocess.cu` — CUDA preprocessing + CPU reference + benchmark harness
- [x] `kernels/matmul.cu` — naive vs. tiled shared-memory matmul
- [x] `docker/Dockerfile.gpu`, `docker/Dockerfile.cpu` — multi-stage builds
- [x] `k8s/` — deployment, service, HPA with probes and resource requests
- [x] `service/app.py` — FastAPI service with CUDA + NumPy paths
- [x] Benchmark numbers filled in from a real T4 run
- [x] Load-test table: latency percentiles vs. replica count
- [ ] GPU image executed on real NVIDIA hardware (rent an hour, or SURF machine)
