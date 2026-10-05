# WarpServe: GPU image preprocessing, from CUDA kernel to Kubernetes

A vision preprocessing service whose core step is a **hand-written CUDA kernel**,
packaged as a multi-stage container and orchestrated on Kubernetes with
autoscaling under load.

In a vision serving stack the model is not always the bottleneck. The CPU-side
preprocessing in front of it (resize, normalize, layout conversion) often is,
which leaves the GPU idle. This project measures that step, moves it to the GPU,
and then measures what happens to the service when traffic rises.

Every speedup is checked for correctness against a scalar reference
implementation.

## Results at a glance

| Measurement | Result | Where it ran |
|---|---|---|
| Fused preprocessing kernel vs. scalar CPU reference | **127.5x** (10.683 ms to 0.084 ms), bit-identical output | NVIDIA Tesla T4 |
| Same kernel including host/device transfers | **1.7x** (6.176 ms); PCIe transfer is the bottleneck | NVIDIA Tesla T4 |
| Tiled shared-memory matmul vs. naive GPU kernel | **1.97x** (812 vs. 411 GFLOP/s) | NVIDIA Tesla T4 |
| Service scaled from 2 to 8 pods | **88 req/s at 83% scaling efficiency**, p50 907 ms to 201 ms | local `kind` cluster, CPU image |
| Multi-stage GPU image vs. its build-stage base | **1.52 GB vs. 11.4 GB** (7.5x smaller) | build verified, not executed |

## Architecture

```
   HTTP request (image)
          |
          v
   FastAPI service  ---------------------------+
          |                                    |
          v                                    v
   CUDA kernel via ctypes               no GPU present?
   (resize + normalize + HWC->CHW)      NumPy fallback, identical math
          |                                    |
          +----------------+-------------------+
                           v
   CHW float32 tensor (shape, statistics, timing)  +  /metrics

   packaged by:  docker/Dockerfile.gpu  (multi-stage, nvcc never ships)
                 docker/Dockerfile.cpu  (runs on Apple Silicon and kind)
   orchestrated: k8s/deployment.yaml + service.yaml + hpa.yaml
```

The service stops at the preprocessed tensor; it does not run a model yet (see
Limitations). The NumPy fallback reproduces the kernel's math exactly, including
the half-pixel centre correction, so both backends return the same tensor and
the same image and manifests run on a machine with no NVIDIA GPU.

| Endpoint | Does |
|---|---|
| `POST /infer` | image in, preprocessed tensor shape, statistics and timing out |
| `GET /healthz` | liveness: the process is alive |
| `GET /readyz` | readiness: the kernel (or fallback) is resolved |
| `GET /metrics` | request count, throughput, p50 / p95 / p99 latency |

## Results

### Preprocessing kernel

Hardware: NVIDIA Tesla T4 (Google Colab), CUDA 12.8, `-O3 -arch=sm_75`.
Workload: 3840x2160 to 640x640, 100 iterations, one thread per output pixel.

| Path | Latency | Speedup |
|---|---|---|
| CPU (scalar reference) | 10.683 ms | 1x |
| GPU, kernel only | **0.084 ms** | **127.5x** |
| GPU, incl. H2D + D2H transfer | 6.176 ms | 1.7x |

Max absolute error vs. the CPU reference: **0.000e+00** (bit-identical).

**The finding is the gap between rows 2 and 3.** The compute is 127x faster, but
the end-to-end gain is 1.7x. Transfer accounts for 6.09 ms of the 6.18 ms:
about 28 MB moved across PCIe at roughly 4.9 GB/s from pageable host memory.
Compute is not the bottleneck here; the bus is.

That argues for keeping frames resident on the device and fusing the whole
pipeline, instead of round-tripping to host memory between stages. Pinned
memory (`cudaHostAlloc`) and reusing device buffers across requests are the
next experiments.

The baseline is a single-threaded scalar loop, chosen so correctness can be
checked line for line. A vectorized, multithreaded CPU baseline would narrow
the 127x figure.

### Matrix multiply: naive vs. tiled shared memory

1024x1024, 50 iterations, TILE=16, same T4.

| Implementation | Latency | Throughput | vs. CPU |
|---|---|---|---|
| CPU (scalar) | 3176.543 ms | 0.68 GFLOP/s | 1x |
| GPU naive (global memory) | 5.221 ms | 411.35 GFLOP/s | 608.5x |
| GPU tiled (shared memory) | **2.646 ms** | **811.67 GFLOP/s** | **1200.6x** |

Shared-memory tiling is worth **1.97x** over the naive kernel with identical
arithmetic; it is purely a memory-locality result.

Max absolute error vs. CPU: 9.155e-05 for both GPU kernels (float accumulation
order differs from the CPU's; the two GPU versions agree with each other
exactly).

812 GFLOP/s is about 10% of the T4's FP32 peak. Tiling is the first
optimization, not the last, and cuBLAS reaches several TFLOP/s on this problem.

### Containerized service

`docker build -f docker/Dockerfile.cpu -t warpserve:cpu .` gives a **408 MB**
image that runs as a non-root user (uid 10001) with a `HEALTHCHECK`.

| Image | Size |
|---|---|
| `nvidia/cuda:12.4.1-devel-ubuntu22.04` (build-stage base) | 11.4 GB |
| `warpserve:gpu` (shipped runtime image) | **1.52 GB** |
| `warpserve:cpu` | 408 MB |

The multi-stage build keeps the CUDA toolkit out of the shipped image:
**7.5x smaller, about 9.9 GB saved**.

### Kubernetes scaling

Local `kind` cluster, **CPU image**, one 1080p JPEG per request, 24 concurrent
clients.

**First finding: the measurement harness was the bottleneck.** Driving load
through `kubectl port-forward` gave about 12.5 req/s whether the Deployment had
2 pods or 8. `port-forward` is a single TCP tunnel proxied by the API server and
saturates long before the pods do. Moving the load generator into the cluster
(`bench/incluster_load.py`, hitting the Service directly) made the pods the
bottleneck.

Load generated in-cluster, 45 s per run:

| Replicas | Throughput | p50 | p95 | p99 |
|---|---|---|---|---|
| 2 | 26.5 req/s | 907 ms | 1694 ms | 1783 ms |
| 4 | 47.4 req/s | 334 ms | 1387 ms | 1654 ms |
| 8 | **88.0 req/s** | **201 ms** | **735 ms** | 1094 ms |

4x the pods yields 3.3x the throughput (**83% scaling efficiency**) while p50
drops 4.5x. It is sub-linear because all pods share one kind node.

**Second finding: autoscaling is not instant.** With the HPA applied (CPU
target 60% of a 200m request, min 2, max 8) and load applied at t=0:

| t | desired | ready |
|---|---|---|
| 15 s | 2 | 2 |
| 45 s | 6 | 2 |
| 60 s | 8 | 6 |
| 75 s | 8 | 8 |

It took about 45 s before the HPA raised its target and about 75 s to have 8
pods serving: metrics-server scrape interval, HPA sync period and pod startup
stack in series. Autoscaling handles sustained load shifts, not spikes; spikes
need headroom in `minReplicas` or a faster signal than CPU.

## Reproduce

Kernels (needs an NVIDIA GPU; a free Colab T4 works):

```bash
nvcc -O3 -arch=sm_75 kernels/preprocess.cu -o preprocess && ./preprocess
nvcc -O3 -arch=sm_75 kernels/matmul.cu -o matmul && ./matmul
```

Service and cluster (no GPU needed):

```bash
docker build -f docker/Dockerfile.cpu -t warpserve:cpu .
kind create cluster --name warpserve
kind load docker-image warpserve:cpu --name warpserve
kubectl apply -f k8s/deployment.yaml -f k8s/service.yaml -f k8s/hpa.yaml
kubectl apply -f bench/loadgen-pod.yaml      # in-cluster load generator
```

`k8s/hpa.yaml` documents the metrics-server setup that `kind` needs.

## Limitations

- **The GPU path is validated on a Colab T4, not in the cluster.** The
  Kubernetes results use the CPU image. The GPU image builds
  (`libpreprocess.so` is produced in the build stage) but has never been
  executed, which needs an NVIDIA GPU and the container toolkit.
- **No model inference yet.** The service returns the preprocessed tensor.
  Running a model behind it (ONNX Runtime, then TensorRT) is the next step.
- **Device memory is allocated per call** in `warpserve_preprocess`, from
  pageable host memory. Cached device buffers and pinned memory are not
  implemented.
- **The kernel does point-sampled bilinear interpolation with no
  anti-aliasing**, so a 4K to 640 downscale will not match area-averaging
  resizers.

## Repo structure

```
kernels/preprocess.cu     fused resize + normalize + HWC->CHW kernel, CPU reference, benchmark
kernels/matmul.cu         naive vs. shared-memory tiled matrix multiply
service/app.py            FastAPI service with CUDA and NumPy paths
docker/                   multi-stage GPU image, CPU image
k8s/                      deployment, service, horizontal pod autoscaler
bench/                    load generators (local and in-cluster)
```
