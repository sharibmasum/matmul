// Kernel 5: 2D blocktiling
//
// Each thread computes a TM x TN square of C. Per dotIdx: load TM values of As
// and TN values of Bs into registers, then do an outer product.
// Raises arithmetic intensity. Params: BM=BN=128, BK=TM=TN=8.

#pragma once
#include <cuda_runtime.h>

template <const int BM, const int BN, const int BK, const int TM, const int TN> // BM=BN=128, BK=TM=TN=8

__global__ void matmul_2D(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C)
{

    const int cRow = blockIdx.y;
    const int cCol = blockIdx.x;

    const int threadCol = threadIdx.x % (BN / TN);
    const int threadRow = threadIdx.x / (BN / TN);

    __shared__ float As[BM * BK]; // 128x8
    __shared__ float Bs[BK * BN]; // 8x128

    A += cRow * BM * K;             // start of the BM row tile of A
    B += cCol * BN;                 // start of the BN col tile of B
    C += cRow * BM * N + cCol * BN; // start of the BMxBN tile of C

    const int numOfThreads = (BM / TM) * (BN / TN); // 16x16=256 threads per block

    const int innerColA = threadIdx.x % BK;
    const int innerRowA = threadIdx.x / BK;
    const int strideA = numOfThreads / BK;

    const int innerColB = threadIdx.x % BN;
    const int innerRowB = threadIdx.x / BN;
    const int strideB = numOfThreads / BN; // 256/128=2

    float threadResults[TM * TN] = {0.0f};
    float regM[TM] = {0.0f}, regN[TN] = {0.0f};

    for (int blockID = 0; blockID < K; blockID += BK)
    {
        for (int loadOffset = 0; loadOffset < BM; loadOffset += strideA)
        {
            As[(innerRowA + loadOffset) * BK + innerColA] = A[(innerRowA + loadOffset) * K + innerColA];
        }
        for (int loadOffset = 0; loadOffset < BK; loadOffset += strideB)
        {
            Bs[(innerRowB + loadOffset) * BN + innerColB] = B[(innerRowB + loadOffset) * N + innerColB];
        }
        __syncthreads();

        A += BK;
        B += BK * N;

        for (int dotId = 0; dotId < BK; dotId++)
        {
            for (int i = 0; i < TM; i++)
            {
                regM[i] = As[(threadRow * TM + i) * BK + dotId];
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
        for (int j = 0; j < TN; j++)
        {
            int idx = (threadRow * TM + i) * N + (threadCol * TN + j);
            C[idx] = alpha * threadResults[i * TN + j] + beta * C[idx];
        }
    }
}