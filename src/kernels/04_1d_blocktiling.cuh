// Kernel 4: 1D blocktiling
//
// Each thread computes TM results (a column strip of C) held in registers.
// The B value is loaded once into a register and reused TM times.
// Params: BM=BN=64, BK=8, TM=8.

#pragma once
#include <cuda_runtime.h>

template <const int BM, const int BN, const int BK, const int TM> // BM=BN=64, BK=8, TM=8

__global__ void matmul_1D(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) { 

    // which BM and BN tile 
    const int cRow = blockIdx.y; 
    const int cCol = blockIdx.x;

    const int threadCol = threadIdx.x % BN; 
    const int threadRow = threadIdx.x / BN;

    __shared__ float As[BM * BK]; // 64x8
    __shared__ float Bs[BK * BN]; // 8x64

    A += cRow * BM * K; // start of the BM row tile of A
    B += cCol * BN; // start of the BN col tile of B
    C += cRow * BM * N + cCol * BN; // start of the BMxBN tile of C

    const int innerColA = threadIdx.x % BK; 
    const int innerRowA = threadIdx.x / BK;
    const int innerColB = threadIdx.x % BN;
    const int innerRowB = threadIdx.x / BN;

    float threadResults[TM] = {0.0f}; // each thread computes TM results


    for (int blockIdx = 0; blockIdx < K; blockIdx += BK) { 
        As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
        Bs[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];

        __syncthreads();
        A += BK; // move to the next BK tile of A
        B += BK * N; // move to the next BK tile of B

        for (int k=0; k<BK; k++) { 
            float tmpB = Bs[k * BN + threadCol]; // load B value once into a register

            for (int t=0; t<TM; t++) {
                threadResults[t] += As[(threadRow * TM + t) * BK + k] * tmpB; 
            } 
        }
        __syncthreads(); // wait for all threads to finish using the tile before loading the next one
    }

    for (int t=0; t<TM; t++) { 
        C[(threadRow * TM + t) * N + threadCol] = alpha * threadResults[t] + beta * C[(threadRow * TM + t) * N + threadCol];
    }

}
