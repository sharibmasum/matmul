// Kernel 3: Shared memory cache-blocking
//
// Block loads a 32x32 tile of A and B into SMEM, __syncthreads, computes the
// partial dot products from SMEM, slides along K. Still 1 result per thread,
// so it ends up bound by SMEM loads (MIO throttle stalls).
