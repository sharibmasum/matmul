// Kernel 7: Autotunable kernel 6
//
// Same algorithm as kernel 6 (transposed As, float4 GMEM loads/stores, TM x TN
// register tile), but BM, BN, BK, TM, TN can be any valid combination, so the
// GMEM -> SMEM copies are loops again (several float4 per thread).

#pragma once
#include <cuda_runtime.h>

template <int BM, int BN, int TM, int TN>
__host__ __device__ constexpr int numThreads() { return (BM / TM) * (BN / TN); }

template <int BM, int BN, int BK, int TM, int TN>
__host__ __device__ constexpr bool validConfig()
{
    return (BM % TM == 0) && (BN % TN == 0) && (BK % 4 == 0) && (BN % 4 == 0) && (TN % 4 == 0) && (numThreads<BM, BN, TM, TN>() <= 1024) && (numThreads<BM, BN, TM, TN>() % (BK / 4) == 0) && (numThreads<BM, BN, TM, TN>() % (BN / 4) == 0) && ((BM * BK) % (4 * numThreads<BM, BN, TM, TN>()) == 0) && ((BK * BN) % (4 * numThreads<BM, BN, TM, TN>()) == 0) && ((BM * BK + BK * BN) * 4 <= 48 * 1024);
}

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void matmul_tuned(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C)
{
    static_assert(validConfig<BM, BN, BK, TM, TN>(), "invalid BM/BN/BK/TM/TN combination");

    const int cRow = blockIdx.y;
    const int cCol = blockIdx.x;

    const int threadCol = threadIdx.x % (BN / TN);
    const int threadRow = threadIdx.x / (BN / TN);

    __shared__ float As[BK * BM]; // transposed
    __shared__ float Bs[BK * BN];

    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

    constexpr int threads = numThreads<BM, BN, TM, TN>();

    const int innerRowA = threadIdx.x / (BK / 4);
    const int innerColA = threadIdx.x % (BK / 4);
    constexpr int strideA = threads / (BK / 4); // rows of As covered per pass

    const int innerRowB = threadIdx.x / (BN / 4);
    const int innerColB = threadIdx.x % (BN / 4);
    constexpr int strideB = threads / (BN / 4); // rows of Bs covered per pass

    float threadResults[TM * TN] = {0.0f};
    float regM[TM] = {0.0f};
    float regN[TN] = {0.0f};

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
            for (int i = 0; i < TM; i++)
            {
                regM[i] = As[dotId * BM + threadRow * TM + i];
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