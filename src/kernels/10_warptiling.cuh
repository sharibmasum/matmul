// Kernel 10: Warptiling (final, ~94% of cuBLAS in the article)
//
// Three levels of tiling:
//   block tile (BM x BN)  -> one per SM-resident block, staged in SMEM
//   warp tile  (WM x WN)  -> one per warp, split into WMITER x WNITER sub-tiles
//   thread tile (TM x TN) -> registers
// warpIdx = threadIdx.x / 32. Better register-cache locality, and it maps
// onto the warp-level matrix ops a tensor-core version would use.
