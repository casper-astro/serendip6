#include <stdio.h>
#include <assert.h>
#include <cuda.h>

int main(int argc, char* argv[])
{
    int* p = NULL;
    int* q = NULL;
    int i = 0;
    cudaError_t iRet;

    p = (int*) malloc(8*64*sizeof(int));
    assert(p != NULL);
    q = (int*) malloc(2*64*sizeof(int));
    assert(q != NULL);

    for (i = 0; i < 8*64; ++i)
    {
        p[i] = i;
    }

    iRet = cudaMemcpy2D(q, 2 * sizeof(int), p, 8 * sizeof(int), 2 * sizeof(int), 64, cudaMemcpyHostToHost);
    printf("**********\niRet = %d\n**********\n", iRet);

    for (i = 0; i < 8*64; ++i)
    {
        printf("%d ", p[i]);
    }
    printf("\n**********\n");
    for (i = 0; i < 2*64; ++i)
    {
        printf("%d ", q[i]);
    }

    printf("\n");

    free(q);
    free(p);

    return 0;
}

