/*
 * gpu_kernels.cu 
 */

#include "vegas_defines.h"
#include "vegas_databuf.h"
#include "spead_heap.h"
#include "gpu_proc.h"


__global__ void CopyDataForFFT(char4 *pc4Data,
                               float4 *pf4FFTIn)
{
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;

    pf4FFTIn[i].x = (float) pc4Data[i].x;
    pf4FFTIn[i].y = (float) pc4Data[i].y;
    pf4FFTIn[i].z = (float) pc4Data[i].z;
    pf4FFTIn[i].w = (float) pc4Data[i].w;

    return;
}

__global__ void Accumulate(float4 *pf4FFTOut,
                           float4 *pf4SumStokes)
{
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    float4 f4FFTOut = pf4FFTOut[i];
    float4 f4SumStokes = pf4SumStokes[i];

    /* Re(X)^2 + Im(X)^2 */
    f4SumStokes.x += (f4FFTOut.x * f4FFTOut.x)
                         + (f4FFTOut.y * f4FFTOut.y);
    /* Re(Y)^2 + Im(Y)^2 */
    f4SumStokes.y += (f4FFTOut.z * f4FFTOut.z)
                         + (f4FFTOut.w * f4FFTOut.w);
    /* Re(XY*) */
    f4SumStokes.z += (f4FFTOut.x * f4FFTOut.z)
                         + (f4FFTOut.y * f4FFTOut.w);
    /* Im(XY*) */
    f4SumStokes.w += (f4FFTOut.y * f4FFTOut.z)
                         - (f4FFTOut.x * f4FFTOut.w);

    pf4SumStokes[i] = f4SumStokes;

    return;
}

