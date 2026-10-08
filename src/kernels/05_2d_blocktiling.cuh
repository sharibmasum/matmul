// Kernel 5: 2D blocktiling
//
// Each thread computes a TM x TN square of C. Per dotIdx: load TM values of As
// and TN values of Bs into registers, then do an outer product.
// Raises arithmetic intensity. Params: BM=BN=128, BK=TM=TN=8.
