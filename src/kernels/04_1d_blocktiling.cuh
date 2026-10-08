// Kernel 4: 1D blocktiling
//
// Each thread computes TM results (a column strip of C) held in registers.
// The B value is loaded once into a register and reused TM times.
// Params: BM=BN=64, BK=8, TM=8.
