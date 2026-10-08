// Kernel 1: Naive
//
// One thread per entry of C; each thread does a full K-length dot product
// straight from global memory.
// threadIdx.x -> row, so the 32 threads of a warp read 32 different rows of A
// => uncoalesced, ~1% of cuBLAS.

#pragma once 
#include <cuda_runtime.h>

// A is MxK, B is KxN, C is MxN, ALL STORED ROW BY ROW (MAJOR ROW ORDER)
__global__ void matmul_naive(int M, int N, int K, float alpha, 
    const float *A, const float *B, const float beta, float *C) { 
    
        const int row = blockIdx.x * blockDim.x + threadIdx.x;
        const int col = blockIdx.y * blockDim.y + threadIdx.y;

        // grids is rounded to whole 32x32 blocks, so some threads will be out of bounds
        if (row < M && col < N) { //gaurd
            float tmp = 0.0f;

            for (int k = 0; k < K; ++k) {
                tmp += A[row * K + k] * B[k * N + col];
            }

            C[row * N + col] = alpha * tmp + beta * C[row * N + col]; 
        }

}