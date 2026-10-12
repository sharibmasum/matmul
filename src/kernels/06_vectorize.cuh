// Kernel 6: Vectorized SMEM + GMEM access
//
// - Transpose A while copying it into SMEM so As reads become LDS.128.
// - float4 reinterpret_cast on global loads/stores => LDG.E.128 / STG.E.128
//   (the cast tells the compiler the pointer is 16B-aligned).
// Params: BM=BN=128, BK=TM=TN=8. Needs K, N divisible by 4.

#pragma once
#include <cuda_runtime.h>

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void matmul_vec(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C)
{
    const int cRow = blockIdx.y;
    const int cCol = blockIdx.x;

    const int threadCol = threadIdx.x % (BN / TN);
    const int threadRow = threadIdx.x / (BN / TN);

    __shared__ float As[BK * BM]; // transposed: 8x128
    __shared__ float Bs[BK * BN]; // 8x128

    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

    constexpr int numOfThreads = (BM / TM) * (BN / TN); // 256
    // one float4 per thread per pass must cover each tile exactly
    static_assert(BM * BK == 4 * numOfThreads, "As load must be exactly one float4 per thread");
    static_assert(BK * BN == 4 * numOfThreads, "Bs load must be exactly one float4 per thread");

    const int innerRowA = threadIdx.x / (BK / 4); // 0..127
    const int innerColA = threadIdx.x % (BK / 4); // 0..1
    const int innerRowB = threadIdx.x / (BN / 4); // 0..7
    const int innerColB = threadIdx.x % (BN / 4); // 0..31

    float threadResults[TM * TN] = {0.0f};
    float regM[TM] = {0.0f};
    float regN[TN] = {0.0f};

    for (int bkIdx = 0; bkIdx < K; bkIdx += BK)
    {
        // A: one 128-bit load, then 4 scalar stores into the transposed tile
        float4 tmp = reinterpret_cast<const float4 *>(&A[innerRowA * K + innerColA * 4])[0];
        As[(innerColA * 4 + 0) * BM + innerRowA] = tmp.x;
        As[(innerColA * 4 + 1) * BM + innerRowA] = tmp.y;
        As[(innerColA * 4 + 2) * BM + innerRowA] = tmp.z;
        As[(innerColA * 4 + 3) * BM + innerRowA] = tmp.w;

        // B: 128-bit load straight into a 128-bit SMEM store
        reinterpret_cast<float4 *>(&Bs[innerRowB * BN + innerColB * 4])[0] =
            reinterpret_cast<const float4 *>(&B[innerRowB * N + innerColB * 4])[0];
        __syncthreads();

        A += BK;
        B += BK * N;

        for (int dotId = 0; dotId < BK; dotId++)
        {
            for (int i = 0; i < TM; i++)
            {
                regM[i] = As[dotId * BM + threadRow * TM + i]; // contiguous now
            }
            for (int j = 0; j < TN; j++)
            {
                regN[j] = Bs[dotId * BN + threadCol * TN + j];
            }
            for (int i = 0; i < TM; i++)
            {
                for (int j = 0; j < TN; j++)
                {
                    threadResults[i * TN + j] += regM[i] * regN[j];
                }
            }
        }
        __syncthreads();
    }

    // write-back, 4 columns at a time
    for (int i = 0; i < TM; i++)
    {
        for (int j = 0; j < TN; j += 4)
        {
            float4 c4 = reinterpret_cast<float4 *>(&C[(threadRow * TM + i) * N + threadCol * TN + j])[0];
            c4.x = alpha * threadResults[i * TN + j + 0] + beta * c4.x;
            c4.y = alpha * threadResults[i * TN + j + 1] + beta * c4.y;
            c4.z = alpha * threadResults[i * TN + j + 2] + beta * c4.z;
            c4.w = alpha * threadResults[i * TN + j + 3] + beta * c4.w;
            reinterpret_cast<float4 *>(&C[(threadRow * TM + i) * N + threadCol * TN + j])[0] = c4;
        }
    }
}