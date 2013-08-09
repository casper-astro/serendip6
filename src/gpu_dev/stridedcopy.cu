#include <stdio.h>
#include <assert.h>
#include <cuda.h>

int main(int argc, char* argv[])
{
    char* p = NULL;
    char* q = NULL;
    char* r = NULL;
    int i = 0;
    cudaError_t iRet;

    p = (char*) malloc(100);
    assert(p != NULL);
    q = (char*) malloc(20);
    assert(q != NULL);
    r = (char*) malloc(40);
    assert(r != NULL);

    for (i = 0; i < 100; ++i)
    {
        p[i] = i;
    }

    iRet = cudaMemcpy2D(q, 1, p, 5, 1, 20, cudaMemcpyHostToHost);
    printf("**********\niRet = %d\n**********\n", iRet);
    iRet = cudaMemcpy2D(r, 2, p, 5, 2, 20, cudaMemcpyHostToHost);
    printf("**********\niRet = %d\n**********\n", iRet);

    for (i = 0; i < 100; ++i)
    {
        printf("%d ", p[i]);
    }
    printf("\n**********\n");
    for (i = 0; i < 20; ++i)
    {
        printf("%d ", q[i]);
    }
    printf("\n**********\n");
    for (i = 0; i < 40; ++i)
    {
        printf("%d ", r[i]);
    }

    printf("\n");

    free(r);
    free(q);
    free(p);

    return 0;
}

