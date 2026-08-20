// WarpServe — GPU image preprocessing kernel
//
// Preprocessing is the step every vision inference server runs before the model:
// resize the image, normalize to [0,1], and convert HWC (interleaved, how images
// are stored) -> CHW (planar, what PyTorch/TensorRT expect). On CPU this is often
// the actual bottleneck in a serving pipeline, because the GPU sits idle waiting
// for it. This kernel does all three in one pass, one thread per output pixel.
//
// Build:  nvcc -O3 -arch=sm_75 preprocess.cu -o preprocess   (sm_75 = T4 on Colab)
// Run:    ./preprocess

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                         \
                    cudaGetErrorString(err), __FILE__, __LINE__);               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// ---------------------------------------------------------------- GPU kernel

__global__ void preprocess_kernel(const unsigned char* __restrict__ src,
                                  int src_h, int src_w,
                                  float* __restrict__ dst,
                                  int dst_h, int dst_w,
                                  float scale_y, float scale_x)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dst_w || y >= dst_h) return;

    // Map this output pixel's CENTER back into input space. The +/- 0.5 is the
    // half-pixel correction (align_corners=False); without it the resize is
    // subtly shifted and won't match OpenCV/PyTorch.
    const float fx = (x + 0.5f) * scale_x - 0.5f;
    const float fy = (y + 0.5f) * scale_y - 0.5f;

    int x0 = (int)floorf(fx);
    int y0 = (int)floorf(fy);
    const float dx = fx - x0;
    const float dy = fy - y0;

    // Clamp to the edge so border pixels don't read out of bounds.
    int x1 = min(x0 + 1, src_w - 1);
    int y1 = min(y0 + 1, src_h - 1);
    x0 = max(x0, 0);  y0 = max(y0, 0);
    x1 = max(x1, 0);  y1 = max(y1, 0);

    const int plane = dst_h * dst_w;

    // Bilinear interpolation, then normalize, then write planar (CHW).
    #pragma unroll
    for (int c = 0; c < 3; ++c) {
        const float v00 = src[(y0 * src_w + x0) * 3 + c];
        const float v01 = src[(y0 * src_w + x1) * 3 + c];
        const float v10 = src[(y1 * src_w + x0) * 3 + c];
        const float v11 = src[(y1 * src_w + x1) * 3 + c];

        const float top = v00 + (v01 - v00) * dx;
        const float bot = v10 + (v11 - v10) * dx;
        const float val = top + (bot - top) * dy;

        dst[c * plane + y * dst_w + x] = val * (1.0f / 255.0f);
    }
}

// ------------------------------------------------------------ CPU reference
// Deliberately the same math, written the obvious scalar way. This is what we
// check the kernel against — a speedup you can't prove correct is worthless.

void preprocess_cpu(const unsigned char* src, int src_h, int src_w,
                    float* dst, int dst_h, int dst_w,
                    float scale_y, float scale_x)
{
    const int plane = dst_h * dst_w;
    for (int y = 0; y < dst_h; ++y) {
        for (int x = 0; x < dst_w; ++x) {
            const float fx = (x + 0.5f) * scale_x - 0.5f;
            const float fy = (y + 0.5f) * scale_y - 0.5f;

            int x0 = (int)std::floor(fx);
            int y0 = (int)std::floor(fy);
            const float dx = fx - x0;
            const float dy = fy - y0;

            int x1 = std::min(x0 + 1, src_w - 1);
            int y1 = std::min(y0 + 1, src_h - 1);
            x0 = std::max(x0, 0);  y0 = std::max(y0, 0);
            x1 = std::max(x1, 0);  y1 = std::max(y1, 0);

            for (int c = 0; c < 3; ++c) {
                const float v00 = src[(y0 * src_w + x0) * 3 + c];
                const float v01 = src[(y0 * src_w + x1) * 3 + c];
                const float v10 = src[(y1 * src_w + x0) * 3 + c];
                const float v11 = src[(y1 * src_w + x1) * 3 + c];

                const float top = v00 + (v01 - v00) * dx;
                const float bot = v10 + (v11 - v10) * dx;
                const float val = top + (bot - top) * dy;

                dst[c * plane + y * dst_w + x] = val * (1.0f / 255.0f);
            }
        }
    }
}

// ------------------------------------------------- C ABI entry point (ctypes)
// The Python service loads this .so and calls this function — the same pattern
// EdgePulse uses for its C++ core. Everything above is CUDA; this is the only
// symbol Python needs to know about.
//
// src is a host HWC uint8 buffer; dst is a host CHW float32 buffer the caller
// has already sized. Returns 0 on success, nonzero on CUDA failure.
//
// Note: this allocates device memory per call. A high-throughput server would
// cache the device buffers across requests — or better, keep frames resident on
// the device entirely, which is what the transfer-cost finding in the README
// actually argues for.

extern "C" int warpserve_preprocess(const unsigned char* h_src, int src_h, int src_w,
                                    float* h_dst, int dst_h, int dst_w)
{
    const size_t src_bytes = (size_t)src_h * src_w * 3 * sizeof(unsigned char);
    const size_t dst_bytes = (size_t)dst_h * dst_w * 3 * sizeof(float);

    unsigned char* d_src = nullptr;
    float* d_dst = nullptr;
    int rc = 0;

    if (cudaMalloc(&d_src, src_bytes) != cudaSuccess) { rc = 1; goto cleanup; }
    if (cudaMalloc(&d_dst, dst_bytes) != cudaSuccess) { rc = 2; goto cleanup; }
    if (cudaMemcpy(d_src, h_src, src_bytes, cudaMemcpyHostToDevice) != cudaSuccess) { rc = 3; goto cleanup; }

    {
        const float scale_y = (float)src_h / dst_h;
        const float scale_x = (float)src_w / dst_w;
        const dim3 block(16, 16);
        const dim3 grid((dst_w + block.x - 1) / block.x,
                        (dst_h + block.y - 1) / block.y);
        preprocess_kernel<<<grid, block>>>(d_src, src_h, src_w, d_dst,
                                           dst_h, dst_w, scale_y, scale_x);
    }

    if (cudaGetLastError() != cudaSuccess)      { rc = 4; goto cleanup; }
    if (cudaDeviceSynchronize() != cudaSuccess) { rc = 5; goto cleanup; }
    if (cudaMemcpy(h_dst, d_dst, dst_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) { rc = 6; goto cleanup; }

cleanup:
    if (d_src) cudaFree(d_src);
    if (d_dst) cudaFree(d_dst);
    return rc;
}

// -------------------------------------------------------------------- main

int main()
{
    // A 4K frame down to YOLO's 640x640 input — the real shape of the problem.
    const int SRC_H = 2160, SRC_W = 3840;
    const int DST_H = 640,  DST_W = 640;
    const int ITERS = 100;

    const float scale_y = (float)SRC_H / DST_H;
    const float scale_x = (float)SRC_W / DST_W;

    const size_t src_bytes = (size_t)SRC_H * SRC_W * 3 * sizeof(unsigned char);
    const size_t dst_bytes = (size_t)DST_H * DST_W * 3 * sizeof(float);

    std::vector<unsigned char> h_src((size_t)SRC_H * SRC_W * 3);
    std::vector<float> h_ref((size_t)DST_H * DST_W * 3);
    std::vector<float> h_gpu((size_t)DST_H * DST_W * 3);

    srand(42);
    for (auto& p : h_src) p = (unsigned char)(rand() % 256);

    unsigned char* d_src = nullptr;
    float* d_dst = nullptr;
    CUDA_CHECK(cudaMalloc(&d_src, src_bytes));
    CUDA_CHECK(cudaMalloc(&d_dst, dst_bytes));

    const dim3 block(16, 16);
    const dim3 grid((DST_W + block.x - 1) / block.x,
                    (DST_H + block.y - 1) / block.y);

    // ---- CPU baseline
    clock_t c0 = clock();
    for (int i = 0; i < ITERS; ++i)
        preprocess_cpu(h_src.data(), SRC_H, SRC_W, h_ref.data(), DST_H, DST_W,
                       scale_y, scale_x);
    double cpu_ms = 1000.0 * (double)(clock() - c0) / CLOCKS_PER_SEC / ITERS;

    // ---- GPU, kernel only (data already resident)
    CUDA_CHECK(cudaMemcpy(d_src, h_src.data(), src_bytes, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    preprocess_kernel<<<grid, block>>>(d_src, SRC_H, SRC_W, d_dst, DST_H, DST_W,
                                       scale_y, scale_x);   // warm-up
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < ITERS; ++i)
        preprocess_kernel<<<grid, block>>>(d_src, SRC_H, SRC_W, d_dst,
                                           DST_H, DST_W, scale_y, scale_x);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float kernel_ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, start, stop));
    kernel_ms /= ITERS;

    // ---- GPU, end to end (this is the number that decides your architecture)
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < ITERS; ++i) {
        CUDA_CHECK(cudaMemcpy(d_src, h_src.data(), src_bytes, cudaMemcpyHostToDevice));
        preprocess_kernel<<<grid, block>>>(d_src, SRC_H, SRC_W, d_dst,
                                           DST_H, DST_W, scale_y, scale_x);
        CUDA_CHECK(cudaMemcpy(h_gpu.data(), d_dst, dst_bytes, cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float e2e_ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&e2e_ms, start, stop));
    e2e_ms /= ITERS;

    // ---- Correctness
    double max_abs_err = 0.0;
    for (size_t i = 0; i < h_ref.size(); ++i)
        max_abs_err = std::max(max_abs_err, (double)std::fabs(h_ref[i] - h_gpu[i]));

    printf("\n  WarpServe preprocessing benchmark\n");
    printf("  %dx%d -> %dx%d, %d iterations\n\n", SRC_W, SRC_H, DST_W, DST_H, ITERS);
    printf("  CPU (scalar reference)     %8.3f ms\n", cpu_ms);
    printf("  GPU (kernel only)          %8.3f ms   %6.1fx\n", kernel_ms, cpu_ms / kernel_ms);
    printf("  GPU (incl. H2D + D2H)      %8.3f ms   %6.1fx\n", e2e_ms, cpu_ms / e2e_ms);
    printf("\n  max abs error vs CPU       %.3e   %s\n\n",
           max_abs_err, max_abs_err < 1e-5 ? "PASS" : "FAIL");

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_dst));
    return max_abs_err < 1e-5 ? 0 : 1;
}
