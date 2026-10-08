// Kernel 2: Global memory coalescing
//
// Same math as kernel 1, but threads are remapped (1D block,
// x = threadIdx.x / BLOCKSIZE, y = threadIdx.x % BLOCKSIZE) so consecutive
// threads in a warp touch consecutive columns => one 128B transaction per warp.
