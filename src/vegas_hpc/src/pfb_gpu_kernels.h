#ifndef _PFB_GPU_KERNELS
#define _PFB_GPU_KERNELS

#include "vegas_defines.h"
#include "vegas_databuf.h"

__global__ void Accumulate(float4 *pf4FFTOut,
                           float4 *pf4SumStokes);

#endif
