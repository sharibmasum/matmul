// Kernel 9: Autotuned
//
// Kernel 6 with all tile sizes as template params (BM, BN, BK, TM, TN),
// plus static_asserts that reject invalid combos (e.g. BM*BK must be
// divisible by 4*NUM_THREADS for vectorized loads).
// Best values are picked by scripts/autotune.sh for this GPU.
