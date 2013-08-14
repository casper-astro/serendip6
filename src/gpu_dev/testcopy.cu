#include <stdio.h>
#include <assert.h>
#include <cuda.h>

int main(int argc, char* argv[])
{
    char* p = NULL;
    char4* q = NULL;
    int i = 0;
    cudaError_t iRet;

    p = (char*) malloc(100 * sizeof(char));
    assert(p != NULL);
    q = (char4*) malloc(25 * sizeof(char4));
    assert(q != NULL);

    for (i = 0; i < 100; ++i)
    {
        p[i] = i;
    }

    iRet = cudaMemcpy(q, p, 100 * sizeof(char), cudaMemcpyHostToHost);

    for (i = 0; i < 100; ++i)
    {
        printf("%d ", p[i]);
    }
    printf("\n**********\n");
    for (i = 0; i < 25; ++i)
    {
        printf("%d %d %d %d ", q[i].x, q[i].y, q[i].z, q[i].w);
    }

    printf("\n");

    free(q);
    free(p);

    return 0;
}

