// Kernel 3: Shared memory cache-blocking
//
// Block loads a 32x32 tile of A and B into SMEM, __syncthreads, computes the
// partial dot products from SMEM, slides along K. Still 1 result per thread,
// so it ends up bound by SMEM loads (MIO throttle stalls).

#pragma once
#include <cuda_runtime.h>

#define TILE 32 // a warp is 32 threads, so we can use a 32x32 tile of SMEM

__global__ void matmul_smem(int M, int N, int K, float alpha, const float *A, const float *B, 
float beta, float *C ) { 
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int col = blockIdx.x * TILE + tx;
    const int row = blockIdx.y * TILE + ty;

    float tmp = 0.0f;

    // sliding the pair of tiles along K, one tile wide step at a time
    for (int t=0; t<K/TILE; t++) { 
        As[ty][tx] = A[row*K + (t*TILE + tx)];
        Bs[ty][tx] = B[(t*TILE + ty)*N + col]; 

        // waiitng for until the whole tile is in SMEM 
        __syncthreads();

        // partial dot 
        for (int k=0; k< TILE; k++) { 
            tmp += As[ty][k] * Bs[k][tx]; 

        }

        __syncthreads(); // wait for all threads to finish using the tile before loading the next one

    }

    C[row * N + col] = alpha * tmp + beta * C[row * N + col];
} 