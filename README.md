# SGEMM From Scratch

CUDA FP32 matrix multiplication, optimized from a naive kernel to a warptiled kernel across 7 steps, reaching up to **87.2% of cuBLAS** at N=2048.

## Overview

This project is a hands-on exercise in GPU performance engineering. It starts from the simplest possible CUDA matrix-multiply kernel and rewrites it kernel by kernel, measuring the real effect of each classic optimization against NVIDIA's cuBLAS as the reference target:

- Global memory coalescing
- Shared-memory tiling
- Register blocking (1D and 2D)
- Vectorized memory ops
- Warp-level tiling

Every kernel computes the same thing:

```
C = alpha * A @ B + beta * C
```

for row-major A (M×K), B (K×N), and C (M×N).

## Benchmark setup

- **GPU:** NVIDIA T4, CUDA 13.0
- **Timing:** CUDA events, after a warm-up launch, averaged over 3-10 runs depending on N
- **Throughput:** for square N×N matrices and a kernel time t (in seconds):

  ```
  GFLOPS = 2 * N^3 / (t * 1e9)
  ```

- **Correctness:** every kernel is checked element by element against `cublasSgemm` (relative tolerance 1e-3) at every size, and a kernel's throughput only counts if it passes.

```cpp
// src/common.cuh (lines 44-54)
inline bool matches(const float *got, const float *ref, size_t count)
{
    for (size_t i = 0; i < count; i++)
    {
        if (fabsf(got[i] - ref[i]) > 1e-3f * fmaxf(1.0f, fabsf(ref[i])))
        {
            return false;
        }
    }
    return true;
}
```

## Results

All values are in GFLOPS. *Coal.* = coalesced, *Vec.* = vectorized, *Warp* = warptiled (the final kernel).

| N | Naive | Coal. | SMEM | 1D | 2D | Vec. | Warp | cuBLAS | % cuBLAS |
|-----:|-----:|------:|------:|-------:|-------:|-------:|-------:|-------:|---------:|
| 512 | 37.8 | 515.2 | 651.6 | 1226.7 | 1262.9 | 1407.9 | 1458.1 | 3378.1 | 43.2% |
| 1024 | 57.5 | 776.1 | 849.9 | 1672.9 | 2880.3 | 3352.2 | 4064.8 | 5584.8 | 72.8% |
| 2048 | 61.3 | 598.1 | 861.8 | 1933.0 | 3617.4 | 3790.0 | 4724.2 | 5548.0 | 87.2% |
| 4096 | 62.0 | 581.7 | 900.6 | 1866.6 | 3309.3 | 3655.2 | 4278.3 | 3952.7 | 108.2% |
| 8192 | 60.2 | 461.2 | 749.1 | 1519.1 | 2436.5 | 2966.9 | 3305.9 | 3012.1 | 109.8% |

### Takeaways

- **The naive kernel never moves.** It is latency-bound on uncoalesced loads regardless of problem size.
- **Coalescing alone is a 10-15x jump** with zero algorithm change.
- **2D blocktiling is the biggest single step**, roughly 2x over 1D at N=2048 and above, which tracks with it roughly doubling arithmetic intensity per shared-memory load.
- **At N=4096 and N=8192, the warptiled kernel beats cuBLAS in this sweep.** cuBLAS's kernel-selection heuristics aren't necessarily tuned optimally for every size on a T4, and this one hand-written configuration happens to fit those sizes well.

### A note on beating cuBLAS

Beating cuBLAS here does **not** mean this kernel is faster in general. cuBLAS has to stay correct and fast across every shape, dtype, and GPU, while this kernel is hand-tuned narrowly for exactly these five square FP32 sizes on this one T4.

## The kernels

### 1. Naive
One thread computes one entry of C with a full K-length dot product straight from global memory. `threadIdx.x` maps to the row, so the 32 threads in a warp each read a different row of A, and every warp issues 32 separate memory transactions instead of one. This is the baseline: ~1% of cuBLAS.

```cpp
// src/kernels/01_naive.cuh (lines 15-27)
const int row = blockIdx.x * blockDim.x + threadIdx.x;
const int col = blockIdx.y * blockDim.y + threadIdx.y;

if (row < M && col < N) {
    float tmp = 0.0f;
    for (int k = 0; k < K; ++k) {
        tmp += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = alpha * tmp + beta * C[row * N + col];
}
```

### 2. Global memory coalescing
Same math, same thread count, but remapped so `threadIdx.x` maps to the column instead of the row. Now the 32 threads in a warp read 32 consecutive elements of B, which the GPU can service as a single 128-byte transaction instead of 32. No algorithm change, and roughly 15x the throughput.

```cpp
// src/kernels/02_gmem_coalescing.cuh (lines 14-15)
const int col = blockIdx.x * blockDim.x + threadIdx.x;
const int row = blockIdx.y * blockDim.y + threadIdx.y;
```

### 3. Shared-memory blocking
Each thread block cooperatively loads a 32x32 tile of A and B into shared memory once, calls `__syncthreads()`, then every thread in the block reuses that tile for its partial dot product before the block slides to the next tile along K. This cuts global memory traffic by roughly a factor of the tile width, but every thread still produces exactly one output element, so the kernel ends up bound by shared-memory load throughput rather than global memory.

```cpp
// src/kernels/03_smem_blocking.cuh (lines 26-41)
for (int t = 0; t < K / TILE; t++) {
    As[ty][tx] = A[row * K + (t * TILE + tx)];
    Bs[ty][tx] = B[(t * TILE + ty) * N + col];
    __syncthreads();

    for (int k = 0; k < TILE; k++) {
        tmp += As[ty][k] * Bs[k][tx];
    }
    __syncthreads();
}
```

### 4. 1D blocktiling (register blocking)
Instead of one output per thread, each thread computes `TM = 8` outputs (a strip down a column of C) held in registers. Each B value is loaded from shared memory once and reused `TM` times across those 8 accumulators, which directly raises arithmetic intensity (FLOPs per shared-memory load).

```cpp
// src/kernels/04_1d_blocktiling.cuh (lines 44-50)
for (int k = 0; k < BK; k++) {
    float tmpB = Bs[k * BN + threadCol]; // loaded once
    for (int t = 0; t < TM; t++) {
        threadResults[t] += As[(threadRow * TM + t) * BK + k] * tmpB;
    }
}
```

### 5. 2D blocktiling
Extends the same idea to two dimensions: each thread owns a `TM x TN = 8x8` tile of C. Per K-step, it loads `TM` values of A and `TN` values of B into registers and accumulates their outer product into a 64-register accumulator. This is the point where arithmetic intensity gets high enough that the kernel starts approaching compute-bound behavior, roughly doubling throughput over the 1D version at larger sizes.

```cpp
// src/kernels/05_2d_blocktiling.cuh (lines 56-75)
for (int dotId = 0; dotId < BK; dotId++) {
    for (int i = 0; i < TM; i++) regM[i] = As[(threadRow * TM + i) * BK + dotId];
    for (int j = 0; j < TN; j++) regN[j] = Bs[dotId * BN + threadCol * TN + j];

    for (int i = 0; i < TM; i++)
        for (int j = 0; j < TN; j++)
            threadResults[i * TN + j] += regM[i] * regN[j];
}
```

### 6. Vectorized memory access
Two changes on top of the kernel 5 tiling shape:

1. Global loads and stores are reinterpreted as `float4` (one 128-bit transaction instead of four 32-bit ones).
2. A's shared-memory tile is transposed while it is copied in, so the later reads out of `As` are contiguous 128-bit loads (`LDS.128`) instead of strided scalar ones.

```cpp
// src/kernels/06_vectorize.cuh (lines 44-52)
float4 tmp = reinterpret_cast<const float4 *>(&A[innerRowA * K + innerColA * 4])[0];
As[(innerColA * 4 + 0) * BM + innerRowA] = tmp.x;
As[(innerColA * 4 + 1) * BM + innerRowA] = tmp.y;
As[(innerColA * 4 + 2) * BM + innerRowA] = tmp.z;
As[(innerColA * 4 + 3) * BM + innerRowA] = tmp.w;

reinterpret_cast<float4 *>(&Bs[innerRowB * BN + innerColB * 4])[0] =
    reinterpret_cast<const float4 *>(&B[innerRowB * N + innerColB * 4])[0];
```

### 7. Warptiling
The final and fastest kernel. It adds a third level of tiling between the block tile and the per-thread register tile: a **warp tile**. Each block tile (128x128) is split across its resident warps, each warp owns a 64x64 warp tile, and that warp tile is further split into `WNITER` sub-tiles so the 32 lanes of the warp can cover it with `TM x TN` register tiles each.

The payoff is better register-cache locality: threads in the same warp read overlapping, nearby shared-memory addresses instead of scattered ones. It is also structurally the shape a tensor-core kernel would use.

```cpp
// src/kernels/08_warptiling.cuh (lines 38-45)
const int warpIdx = threadIdx.x / WARPSIZE;
const int warpCol = warpIdx % (BN / WN);
const int warpRow = warpIdx / (BN / WN);

const int laneIdx = threadIdx.x % WARPSIZE;
const int threadColInWarp = laneIdx % (WSUBN / TN);
const int threadRowInWarp = laneIdx / (WSUBN / TN);
```

The kernel is templated on 9 tile parameters (`BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS`) and leans on `static_assert` to reject invalid combinations at compile time rather than at runtime:

```cpp
// src/kernels/08_warptiling.cuh (lines 24-32)
static_assert(NUM_THREADS == (BM / WM) * (BN / WN) * WARPSIZE, "NUM_THREADS must be 32 per warp tile");
static_assert(BM % WM == 0 && BN % WN == 0, "warp tile must divide block tile");
static_assert((WM * WN) % (WARPSIZE * TM * TN * WNITER) == 0, "WMITER must be an integer");
// ...5 more, covering sub-tile divisibility and float4 alignment
```

## Autotuning

`07_autotuned.cuh` is the same algorithm as the vectorized kernel (6), but with `BM, BN, BK, TM, TN` left as free template parameters instead of fixed constants, so a harness can sweep configurations.

`validConfig()` checks tile and thread-count legality (divisibility, shared-memory budget, `float4` alignment) as a `constexpr` function. The sweep uses `if constexpr` to skip illegal configurations before they are compiled into a kernel launch, instead of guarding at runtime:

```cpp
// src/autotune.cu (lines 11-15)
if constexpr (!validConfig<BM, BN, BK, TM, TN>())
{
    printf("%4d %4d %3d %3d %3d %8s %10s %5s\n", BM, BN, BK, TM, TN, "-", "skipped", "-");
}
```

### Sweep results at N=4096

cuBLAS reached 4191.0 GFLOPS in this run.

| BM | BN | BK | TM | TN | threads | GFLOPS | ok |
|----:|----:|---:|---:|---:|--------:|-------:|:---:|
| 128 | 128 | 8 | 8 | 8 | 256 | 3696.6 | YES |
| 128 | 128 | 16 | 8 | 8 | 256 | 3660.5 | YES |
| 128 | 128 | 32 | 8 | 8 | 256 | 3553.1 | YES |
| 64 | 64 | 16 | 4 | 4 | 256 | 2585.9 | YES |
| 64 | 64 | 16 | 8 | 8 | 64 | 2823.4 | YES |
| 128 | 64 | 8 | 8 | 8 | 128 | 3185.0 | YES |
| 64 | 128 | 8 | 8 | 8 | 128 | 3170.7 | YES |
| 128 | 128 | 16 | 8 | 4 | 512 | 3828.2 | YES |
| 128 | 128 | 16 | 4 | 8 | 512 | 2650.7 | YES |
| 128 | 128 | 8 | 4 | 8 | - | skipped | - |

The one skipped row (`TN=8` with `BK=8`) is deliberately illegal. It fails the "a `float4` load must cover a whole row" constraint, and it is included specifically to confirm the guard rejects it instead of silently miscompiling.

## Repository layout

```
src/
├── common.cuh              # shared helpers, including the correctness check
├── autotune.cu             # configuration sweep harness
└── kernels/
    ├── 01_naive.cuh
    ├── 02_gmem_coalescing.cuh
    ├── 03_smem_blocking.cuh
    ├── 04_1d_blocktiling.cuh
    ├── 05_2d_blocktiling.cuh
    ├── 06_vectorize.cuh
    ├── 07_autotuned.cuh
    └── 08_warptiling.cuh
```

<!-- TODO: add "Building and running" instructions (nvcc flags / Makefile / CMake commands, and the -lcublas link) once confirmed. -->

---

