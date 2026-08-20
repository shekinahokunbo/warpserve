// WarpServe — naive vs. shared-memory tiled matrix multiply
//
// This is the kernel every GPU interview asks about, because it is the smallest
// program that forces you to understand the memory hierarchy. Both versions do
// the identical arithmetic; the tiled one is faster purely because it stages
// data in shared memory (~100x lower latency than global) and each loaded value
// gets reused TILE times instead of being re-fetched from DRAM.
//
// Build:  nvcc -O3 -arch=sm_75 matmul.cu -o matmul
// Run:    ./matmul

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

#define TILE 16

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                         \
                    cudaGetErrorString(err), __FILE__, __LINE__);               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// Each thread walks the full k dimension, reading straight from global memory.
// Every element of A and B gets re-read N times across the grid.
__global__ void matmul_naive(const float* __restrict__ A,
                             const float* __restrict__ B,
                             float* __restrict__ C, int N)
{
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N || col >= N) return;

    float acc = 0.0f;
    for (int k = 0; k < N; ++k)
        acc += A[row * N + k] * B[k * N + col];
    C[row * N + col] = acc;
}

// The block cooperatively loads one TILE x TILE square of A and of B into shared
// memory, then every thread in the block reads from that fast copy. Each global
// load is now amortized over TILE multiply-adds.
__global__ void matmul_tiled(const float* __restrict__ A,
                             const float* __restrict__ B,
                             float* __restrict__ C, int N)
{
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int row = blockIdx.y * TILE + ty;
    const int col = blockIdx.x * TILE + tx;

    float acc = 0.0f;

    for (int t = 0; t < (N + TILE - 1) / TILE; ++t) {
        const int aCol = t * TILE + tx;
        const int bRow = t * TILE + ty;

        // Zero-pad the edges so N doesn't have to be a multiple of TILE.
        As[ty][tx] = (row < N && aCol < N) ? A[row * N + aCol] : 0.0f;
        Bs[ty][tx] = (bRow < N && col < N) ? B[bRow * N + col] : 0.0f;

        // Everyone must finish loading before anyone starts reading.
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k)
            acc += As[ty][k] * Bs[k][tx];

        // ...and everyone must finish reading before the next tile overwrites it.
        __syncthreads();
    }

    if (row < N && col < N)
        C[row * N + col] = acc;
}

void matmul_cpu(const std::vector<float>& A, const std::vector<float>& B,
                std::vector<float>& C, int N)
{
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < N; ++k) acc += A[i * N + k] * B[k * N + j];
            C[i * N + j] = acc;
        }
}

// A macro rather than a function, because the <<<>>> launch syntax wants the
// kernel's actual name at compile time, not a function pointer.
#define TIME_KERNEL(KERNEL, N, ITERS, OUT_MS)                                   \
    do {                                                                        \
        cudaEvent_t _start, _stop;                                              \
        CUDA_CHECK(cudaEventCreate(&_start));                                   \
        CUDA_CHECK(cudaEventCreate(&_stop));                                    \
        const dim3 _block(TILE, TILE);                                          \
        const dim3 _grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);         \
        KERNEL<<<_grid, _block>>>(dA, dB, dC, N);       /* warm-up */           \
        CUDA_CHECK(cudaDeviceSynchronize());                                    \
        CUDA_CHECK(cudaEventRecord(_start));                                    \
        for (int _i = 0; _i < (ITERS); ++_i)                                    \
            KERNEL<<<_grid, _block>>>(dA, dB, dC, N);                           \
        CUDA_CHECK(cudaEventRecord(_stop));                                     \
        CUDA_CHECK(cudaEventSynchronize(_stop));                                \
        CUDA_CHECK(cudaEventElapsedTime(&(OUT_MS), _start, _stop));             \
        (OUT_MS) /= (ITERS);                                                    \
        CUDA_CHECK(cudaEventDestroy(_start));                                   \
        CUDA_CHECK(cudaEventDestroy(_stop));                                    \
    } while (0)

int main()
{
    const int N = 1024;
    const int ITERS = 50;
    const size_t bytes = (size_t)N * N * sizeof(float);

    std::vector<float> hA((size_t)N * N), hB((size_t)N * N);
    std::vector<float> hRef((size_t)N * N), hNaive((size_t)N * N), hTiled((size_t)N * N);

    srand(42);
    for (auto& v : hA) v = (float)rand() / RAND_MAX;
    for (auto& v : hB) v = (float)rand() / RAND_MAX;

    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMalloc(&dB, bytes));
    CUDA_CHECK(cudaMalloc(&dC, bytes));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), bytes, cudaMemcpyHostToDevice));

    float naive_ms = 0.f, tiled_ms = 0.f;

    TIME_KERNEL(matmul_naive, N, ITERS, naive_ms);
    CUDA_CHECK(cudaMemcpy(hNaive.data(), dC, bytes, cudaMemcpyDeviceToHost));

    TIME_KERNEL(matmul_tiled, N, ITERS, tiled_ms);
    CUDA_CHECK(cudaMemcpy(hTiled.data(), dC, bytes, cudaMemcpyDeviceToHost));

    printf("\n  Timing CPU reference (this takes a few seconds)...\n");
    clock_t c0 = clock();
    matmul_cpu(hA, hB, hRef, N);
    const double cpu_ms = 1000.0 * (double)(clock() - c0) / CLOCKS_PER_SEC;

    double err_naive = 0.0, err_tiled = 0.0;
    for (size_t i = 0; i < hRef.size(); ++i) {
        err_naive = std::max(err_naive, (double)std::fabs(hRef[i] - hNaive[i]));
        err_tiled = std::max(err_tiled, (double)std::fabs(hRef[i] - hTiled[i]));
    }

    const double flops = 2.0 * N * N * N;
    printf("\n  %dx%d matmul, %d iterations, TILE=%d\n\n", N, N, ITERS, TILE);
    printf("  CPU (scalar)         %9.3f ms   %7.2f GFLOP/s\n",
           cpu_ms, flops / (cpu_ms * 1e6));
    printf("  GPU naive            %9.3f ms   %7.2f GFLOP/s   %6.1fx vs CPU\n",
           naive_ms, flops / (naive_ms * 1e6), cpu_ms / naive_ms);
    printf("  GPU tiled (shared)   %9.3f ms   %7.2f GFLOP/s   %6.1fx vs CPU   %.2fx vs naive\n",
           tiled_ms, flops / (tiled_ms * 1e6), cpu_ms / tiled_ms, naive_ms / tiled_ms);
    printf("\n  max abs error  naive %.3e   tiled %.3e   %s\n\n",
           err_naive, err_tiled,
           (err_naive < 1e-2 && err_tiled < 1e-2) ? "PASS" : "FAIL");

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    return 0;
}
