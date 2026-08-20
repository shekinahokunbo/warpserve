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


