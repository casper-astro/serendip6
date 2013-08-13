#ifndef _GPU_KERNELS
#define _GPU_KERNELS

#include "vegas_defines.h"
#include "vegas_databuf.h"

__global__ void CopyDataForFFT(char4* pc4Data,
                               float4* pf4FFTIn);

__global__ void Accumulate(float4 *pf4FFTOut,
                           float4 *pf4SumStokes);

#endif
