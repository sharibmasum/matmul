// Kernel 6: Vectorized SMEM + GMEM access
//
// - Transpose A while copying it into SMEM so As reads become LDS.128.
// - float4 reinterpret_cast on global loads/stores => LDG.E.128 / STG.E.128
//   (the cast tells the compiler the pointer is 16B-aligned).
