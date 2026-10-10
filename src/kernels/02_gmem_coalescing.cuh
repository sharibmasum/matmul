// Kernel 2: Global memory coalescing
//
// Same math as kernel 1, but threads are remapped (1D block,
// x = threadIdx.x / BLOCKSIZE, y = threadIdx.x % BLOCKSIZE) so consecutive
// threads in a warp touch consecutive columns => one 128B transaction per warp.

#pragma once 
#include <cuda_runtime.h>

// A is MxK, B is KxN, C is MxN, ALL STORED ROW BY ROW (MAJOR ROW ORDER)
__global__ void matmul_coalesced(int M, int N, int K, float alpha, 
    const float *A, const float *B, const float beta, float *C) { 
    
        const int col = blockIdx.x * blockDim.x + threadIdx.x;
        const int row = blockIdx.y * blockDim.y + threadIdx.y;

        // grids is rounded to whole 32x32 blocks, so some threads will be out of bounds
        if (row < M && col < N) { //gaurd
            float tmp = 0.0f;

            for (int k = 0; k < K; ++k) {
                tmp += A[row * K + k] * B[k * N + col];
            }

            C[row * N + col] = alpha * tmp + beta * C[row * N + col]; 
        }

}