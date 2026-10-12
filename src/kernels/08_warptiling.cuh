// Kernel 8: Warptiling (final, ~94% of cuBLAS in the article)
//
// Three levels of tiling:
//   block tile (BM x BN)  -> one per SM-resident block, staged in SMEM
//   warp tile  (WM x WN)  -> one per warp, split into WMITER x WNITER sub-tiles
//   thread tile (TM x TN) -> registers
// warpIdx = threadIdx.x / 32. Better register-cache locality, and it maps
// onto the warp-level matrix ops a tensor-core version would use.
// Start config: BM=BN=128, BK=16, WM=WN=64, WNITER=4, TM=8, TN=4, 128 threads.

#pragma once
#include <cuda_runtime.h>

template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WNITER, const int TM, const int TN, const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
    matmul_warp(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C)
{
    constexpr int WARPSIZE = 32;
    constexpr int WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);
    constexpr int WSUBM = WM / WMITER;
    constexpr int WSUBN = WN / WNITER;

    static_assert(NUM_THREADS == (BM / WM) * (BN / WN) * WARPSIZE, "NUM_THREADS must be 32 per warp tile");
    static_assert(BM % WM == 0 && BN % WN == 0, "warp tile must divide block tile");
    static_assert((WM * WN) % (WARPSIZE * TM * TN * WNITER) == 0, "WMITER must be an integer");
    static_assert(WM % WMITER == 0 && WN % WNITER == 0, "sub-tiles must divide the warp tile");
    static_assert(WSUBM % TM == 0 && WSUBN % TN == 0, "thread tile must divide the sub-tile");
    static_assert((WSUBM / TM) * (WSUBN / TN) == WARPSIZE, "the 32 lanes must exactly cover a sub-tile");
    static_assert(TN % 4 == 0 && BK % 4 == 0 && BN % 4 == 0, "float4 widths");
    static_assert((BM * BK) % (4 * NUM_THREADS) == 0 && (BK * BN) % (4 * NUM_THREADS) == 0, "tile loads must divide evenly");
    static_assert((NUM_THREADS * 4) % BK == 0 && NUM_THREADS % (BN / 4) == 0, "load passes must cover whole rows");

    const int cRow = blockIdx.y;
    const int cCol = blockIdx.x;

    // which warp this thread is in, and where that warp's tile sits in the block tile
    const int warpIdx = threadIdx.x / WARPSIZE;
    const int warpCol = warpIdx % (BN / WN);
    const int warpRow = warpIdx / (BN / WN);

    // where this thread sits inside a warp sub-tile
    const int laneIdx = threadIdx.x % WARPSIZE;
    const int threadColInWarp = laneIdx % (WSUBN / TN);
    const int threadRowInWarp = laneIdx / (WSUBN / TN);

    __shared__ float As[BK * BM];
    __shared__ float Bs[BK * BN];

    A += cRow * BM * K;
    B += cCol * BN;
    C += (cRow * BM + warpRow * WM) * N + cCol * BN + warpCol * WN;

    const int innerRowA = threadIdx.x / (BK / 4);
    const int innerColA = threadIdx.x % (BK / 4);
    constexpr int strideA = (NUM_THREADS * 4) / BK;

    const int innerRowB = threadIdx.x / (BN / 4);
    const int innerColB = threadIdx.x % (BN / 4);
    constexpr int strideB = NUM_THREADS / (BN / 4);

    float threadResults[WMITER * TM * WNITER * TN] = {0.0f};
    float regM[WMITER * TM] = {0.0f};
    float regN[WNITER * TN] = {0.0f};

    for (int bkIdx = 0; bkIdx < K; bkIdx += BK)
    {
        for (int loadOffset = 0; loadOffset < BM; loadOffset += strideA)
        {
            float4 tmp = reinterpret_cast<const float4 *>(&A[(innerRowA + loadOffset) * K + innerColA * 4])[0];
            As[(innerColA * 4 + 0) * BM + innerRowA + loadOffset] = tmp.x;
            As[(innerColA * 4 + 1) * BM + innerRowA + loadOffset] = tmp.y;
            As[(innerColA * 4 + 2) * BM + innerRowA + loadOffset] = tmp.z;
            As[(innerColA * 4 + 3) * BM + innerRowA + loadOffset] = tmp.w;
        }
        for (int loadOffset = 0; loadOffset < BK; loadOffset += strideB)
        {
            reinterpret_cast<float4 *>(&Bs[(innerRowB + loadOffset) * BN + innerColB * 4])[0] =
                reinterpret_cast<const float4 *>(&B[(innerRowB + loadOffset) * N + innerColB * 4])[0];
        }
        __syncthreads();

        A += BK;
        B += BK * N;

        for (int dotId = 0; dotId < BK; dotId++)
        {
            // registers for this thread's slice of the whole warp tile
            for (int wSubRow = 0; wSubRow < WMITER; wSubRow++)
            {
                for (int i = 0; i < TM; i++)
                {
                    regM[wSubRow * TM + i] =
                        As[dotId * BM + warpRow * WM + wSubRow * WSUBM + threadRowInWarp * TM + i];
                }
            }
            for (int wSubCol = 0; wSubCol < WNITER; wSubCol++)
            {
                for (int j = 0; j < TN; j++)
                {
                    regN[wSubCol * TN + j] =
                        Bs[dotId * BN + warpCol * WN + wSubCol * WSUBN + threadColInWarp * TN + j];
                }
            }

            // outer product, one sub-tile pair at a time
            for (int wSubRow = 0; wSubRow < WMITER; wSubRow++)
            {
                for (int wSubCol = 0; wSubCol < WNITER; wSubCol++)
                {
                    for (int i = 0; i < TM; i++)
                    {
                        for (int j = 0; j < TN; j++)
                        {
                            threadResults[(wSubRow * TM + i) * (WNITER * TN) + wSubCol * TN + j] +=
                                regM[wSubRow * TM + i] * regN[wSubCol * TN + j];
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

    // write back, one sub-tile at a time, 4 columns per float4
    for (int wSubRow = 0; wSubRow < WMITER; wSubRow++)
    {
        for (int wSubCol = 0; wSubCol < WNITER; wSubCol++)
        {
            float *C_sub = C + (wSubRow * WSUBM) * N + wSubCol * WSUBN;
            for (int i = 0; i < TM; i++)
            {
                for (int j = 0; j < TN; j += 4)
                {
                    float4 c4 = reinterpret_cast<float4 *>(
                        &C_sub[(threadRowInWarp * TM + i) * N + threadColInWarp * TN + j])[0];
                    const int r = (wSubRow * TM + i) * (WNITER * TN) + wSubCol * TN + j;
                    c4.x = alpha * threadResults[r + 0] + beta * c4.x;
                    c4.y = alpha * threadResults[r + 1] + beta * c4.y;
                    c4.z = alpha * threadResults[r + 2] + beta * c4.z;
                    c4.w = alpha * threadResults[r + 3] + beta * c4.w;
                    reinterpret_cast<float4 *>(
                        &C_sub[(threadRowInWarp * TM + i) * N + threadColInWarp * TN + j])[0] = c4;
                }
            }
        }
    }
}