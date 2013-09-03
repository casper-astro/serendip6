#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <errno.h>
#include <cufft.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include "fitshead.h"
#include "vegas_error.h"
#ifdef __cplusplus
extern "C" {
#endif
#include "vegas_status.h"
#include "vegas_databuf.h"
#ifdef __cplusplus
}
#endif
#include "vegas_defines.h"
#include "gpu_proc.h"
#include "gpu_kernels.h"
#include "spead_heap.h"

#define STATUS_KEY "GPUSTAT"

/* ASSUMPTIONS: 1. All blocks contain the same number of heaps. */

extern int run;

/**
 * Global variables: maybe move this to a struct that is passed to each function?
 */
size_t g_buf_in_block_size;
size_t g_buf_out_block_size;
int g_nchan;

cufftHandle g_stPlan = {0};
float4* g_pf4FFTIn_d = NULL;
float4* g_pf4FFTOut_d = NULL;
char4* g_pc4InBuf = NULL;
char4* g_pc4InBufRead = NULL;
char4* g_pc4Data_d = NULL;              /* raw data starting address */
char4* g_pc4DataRead_d = NULL;          /* raw data read address */
char4* g_pc4DataStage_d = NULL;              /* data staging area address */
dim3 g_dimBCopy(1, 1, 1);
dim3 g_dimGCopy(1, 1);
dim3 g_dimBAccum(1, 1, 1);
dim3 g_dimGAccum(1, 1);
float4** g_ppf4SumStokes = NULL;
float4** g_ppf4SumStokes_d = NULL;
int g_iNumSubBands = 0;
int g_iTotHeapOut = 0;
int g_iMaxNumHeapOut = 0;
int g_iPFBCurBlockOut = 0;
int g_iHeapOut = 0;
int g_iBlockInDataSize = 0;
/* these arrays need to be only a little longer than MAX_HEAPS_PER_BLK, but
   since we don't know the exact length, just allocate twice that value */
unsigned int g_auiStatusBits[2*MAX_HEAPS_PER_BLK] = {0};
unsigned int g_auiHeapValid[2*MAX_HEAPS_PER_BLK] = {0};
int g_iFirstHeapIn = 0;
double g_dFirstHeapRcvdMJD = 0.0;
int g_iSpecPerAcc = 0;
int g_iNumConcFFT = 4;
int g_iNumChanBlocks = 0;
int g_iAccID = 0;
int g_iReadID = 0;

void __CUDASafeCall(cudaError_t iCUDARet,
                    const char* pcFile,
                    const int iLine,
                    void (*pCleanUp)(void));

#define CUDASafeCall(iRet)   __CUDASafeCall(iRet,                             \
                                            __FILE__,                         \
                                            __LINE__,                         \
                                            &cleanup_gpu)

/* Initialize all necessary memory, etc for doing PFB 
 * at the given params.
 */
extern "C"
int init_gpu(size_t input_block_sz, size_t output_block_sz, int num_subbands, int num_chans)
{
    int iDevCount = 0;
    cudaDeviceProp stDevProp = {0};
    cufftResult iCUFFTRet = CUFFT_SUCCESS;
    int iMaxThreadsPerBlock = 0;

    g_buf_in_block_size = input_block_sz;
    g_buf_out_block_size = output_block_sz;
    g_nchan = num_chans;
    g_iNumSubBands = num_subbands;

    /* since CUDASafeCall() calls cudaGetErrorString(),
       it should not be used here - will cause crash if no CUDA device is
       found */
    (void) cudaGetDeviceCount(&iDevCount);
    if (0 == iDevCount)
    {
        (void) fprintf(stderr, "ERROR: No CUDA-capable device found!\n");
        run = 0;
        return EXIT_FAILURE;
    }

    /* just use the first device */
    CUDASafeCall(cudaSetDevice(0));

    CUDASafeCall(cudaGetDeviceProperties(&stDevProp, 0));
    iMaxThreadsPerBlock = stDevProp.maxThreadsPerBlock;

    g_iNumChanBlocks = g_iNumSubBands / g_iNumConcFFT;

    /* allocate memory for data array - 32MB is the block size for the VEGAS
       input buffer, allocate 32MB
       NOTE: the actual data in a 32MB block will be only
       (num_heaps * heap_size), but since we don't know that value until data
       starts flowing, allocate the maximum possible size */
    CUDASafeCall(cudaMalloc((void **) &g_pc4Data_d,
                            g_buf_in_block_size));
    g_pc4DataRead_d = g_pc4Data_d;

    /* allocate memory for data staging area. this is where
       (g_iNumConcFFT * g_nchan) samples are stored for FFT */
    CUDASafeCall(cudaMalloc((void **) &g_pc4DataStage_d,
                            (g_iNumConcFFT * g_nchan * sizeof(char4))));

    /* calculate kernel parameters */
    /* ASSUMPTION: g_nchan >= iMaxThreadsPerBlock */
    g_dimBCopy.x = iMaxThreadsPerBlock;
    g_dimBAccum.x = iMaxThreadsPerBlock;
    g_dimGCopy.x = (g_iNumConcFFT * g_nchan) / iMaxThreadsPerBlock;
    g_dimGAccum.x = (g_iNumConcFFT * g_nchan) / iMaxThreadsPerBlock;

    CUDASafeCall(cudaMalloc((void **) &g_pf4FFTIn_d,
                                 g_iNumConcFFT * g_nchan * sizeof(float4)));
    CUDASafeCall(cudaMalloc((void **) &g_pf4FFTOut_d,
                                 g_iNumConcFFT * g_nchan * sizeof(float4)));

    g_ppf4SumStokes = (float4 **) malloc(g_iNumChanBlocks * sizeof(float4 **));
    if (NULL == g_ppf4SumStokes)
    {
        (void) fprintf(stderr,
                       "ERROR: Memory allocation failed! %s.\n",
                       strerror(errno));
        run = 0;
        return EXIT_FAILURE;
    }
    for (int i = 0; i < g_iNumChanBlocks; ++i)
    {
        g_ppf4SumStokes[i] = (float4 *) malloc(g_iNumConcFFT
                                               * g_nchan
                                               * sizeof(float4));
        if (NULL == g_ppf4SumStokes[i])
        {
            (void) fprintf(stderr,
                           "ERROR: Memory allocation failed! %s.\n",
                           strerror(errno));
            run = 0;
            return EXIT_FAILURE;
        }
    }

    g_ppf4SumStokes_d = (float4 **) malloc(g_iNumChanBlocks * sizeof(float4 **));
    if (NULL == g_ppf4SumStokes_d)
    {
        (void) fprintf(stderr,
                       "ERROR: Memory allocation failed! %s.\n",
                       strerror(errno));
        run = 0;
        return EXIT_FAILURE;
    }
    for (int i = 0; i < g_iNumChanBlocks; ++i)
    {
        CUDASafeCall(cudaMalloc((void **) &g_ppf4SumStokes_d[i],
                                g_iNumConcFFT
                                * g_nchan
                                * sizeof(float4)));
        CUDASafeCall(cudaMemset(g_ppf4SumStokes_d[i],
                                '\0',
                                g_iNumConcFFT
                                * g_nchan
                                * sizeof(float4)));
    }

    /* create plan */
    iCUFFTRet = cufftPlanMany(&g_stPlan,
                              FFTPLAN_RANK,
                              &g_nchan,
                              &g_nchan,
                              FFTPLAN_ISTRIDE,
                              FFTPLAN_IDIST,
                              &g_nchan,
                              FFTPLAN_OSTRIDE,
                              FFTPLAN_ODIST,
                              CUFFT_C2C,
                              FFTPLAN_BATCH);
    if (iCUFFTRet != CUFFT_SUCCESS)
    {
        (void) fprintf(stderr, "ERROR: Plan creation failed!\n");
        run = 0;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

/* Actually do the PFB by calling CUDA kernels */
extern "C"
void do_proc(struct vegas_databuf *db_in,
             int curblock_in,
             struct vegas_databuf *db_out,
             int first,
             struct vegas_status st,
             int acc_len)
{
    /* Declare local variables */
    char *hdr_out = NULL;
    struct databuf_index *index_in = NULL;
    struct databuf_index *index_out = NULL;
    int heap_in = 0;
    char *heap_addr_in = NULL;
    char *heap_addr_out = NULL;
    struct time_spead_heap* first_time_heap_in_accum = NULL;
    struct freq_spead_heap* freq_heap_out = NULL;
    cudaError_t iCUDARet = cudaSuccess;
    int iRet = VEGAS_OK;
    char* payload_addr_in = NULL;
    char* payload_addr_out = NULL;
    int num_in_heaps_per_proc = 0;
    int num_in_heaps_gpu_buffer = 0;
    int i = 0;
    int is_proc_done = 0;

    /* Setup input and first output data block stuff */
    index_in = (struct databuf_index*)vegas_databuf_index(db_in, curblock_in);
    /* Get the number of heaps per block of data that will be processed by the GPU */
    g_iBlockInDataSize = (index_in->num_heaps * index_in->heap_size) - (index_in->num_heaps * sizeof(struct time_spead_heap));
    num_in_heaps_per_proc = (g_iNumConcFFT * g_nchan * sizeof(char4)) / (index_in->heap_size - sizeof(struct time_spead_heap));
    num_in_heaps_gpu_buffer = index_in->num_heaps;

    /* Calculate the maximum number of output heaps per block */
    g_iMaxNumHeapOut = (g_buf_out_block_size - (sizeof(struct time_spead_heap) * MAX_HEAPS_PER_BLK))
                       / (g_iNumSubBands * g_nchan * sizeof(float4)); 

    hdr_out = vegas_databuf_header(db_out, g_iPFBCurBlockOut);
    index_out = (struct databuf_index*)vegas_databuf_index(db_out, g_iPFBCurBlockOut);
    memcpy(hdr_out, vegas_databuf_header(db_in, curblock_in),
            VEGAS_STATUS_SIZE);

    /* Set basic params in output index */
    index_out->heap_size = sizeof(struct freq_spead_heap) + (g_iNumSubBands * g_nchan * sizeof(float4));
    /* Read in heap from buffer */
    heap_addr_in = (char*)(vegas_databuf_data(db_in, curblock_in) +
                        sizeof(struct time_spead_heap) * heap_in);
    first_time_heap_in_accum = (struct time_spead_heap*)(heap_addr_in);
    if (first)
    {
        g_iFirstHeapIn = heap_in;
        g_dFirstHeapRcvdMJD = index_in->cpu_gpu_buf[g_iFirstHeapIn].heap_rcvd_mjd;
    }
    /* Here, the payload_addr_in is the start of the contiguous block of data that will be
       copied to the GPU (heap_in = 0) */
    payload_addr_in = (char*)(vegas_databuf_data(db_in, curblock_in) +
                        sizeof(struct time_spead_heap) * MAX_HEAPS_PER_BLK +
                        (index_in->heap_size - sizeof(struct time_spead_heap)) * heap_in);

    if (first)
    {
        /* Sanity check for the first iteration */
        if ((g_iBlockInDataSize % (g_iNumSubBands * g_nchan * sizeof(char4))) != 0)
        {
            (void) fprintf(stderr, "ERROR: Data size mismatch!\n");
            run = 0;
            return;
        }
    }

    /* copy the entire contents (data) of the input block to the GPU */
    CUDASafeCall(cudaMemcpy(g_pc4Data_d,
                            payload_addr_in,
                            g_iBlockInDataSize,
                            cudaMemcpyHostToDevice));
    g_pc4DataRead_d = g_pc4Data_d;

    struct time_spead_heap* time_heap = (struct time_spead_heap*) vegas_databuf_data(db_in, curblock_in);
    for (i = 0; i < index_in->num_heaps; ++i)
    {
        g_auiStatusBits[i] = time_heap->status_bits;
        g_auiHeapValid[i] = index_in->cpu_gpu_buf[i].heap_valid;
        ++time_heap;
    }

    while (!is_proc_done)
    {
        g_iAccID = 0;
        while (g_iAccID < g_iNumChanBlocks)  /* loop till (g_iNumSubBands * g_nchan * sizeof(char4)) of data is processed */
        {
            /* copy data for one chunk of FFT from input block to GPU */
            read_gpu_input_data();

            #if 0
            if (0 == pfb_count)
            {
                /* Check if all heaps necessary for this PFB are valid */
                if (!(is_valid(heap_in, (VEGAS_NUM_TAPS * num_in_heaps_per_proc))))
                {
                    /* Skip all heaps that go into this PFB if there is an invalid heap */
                    iProcData += (VEGAS_NUM_TAPS * g_iNumSubBands * g_nchan * sizeof(char4));
                    /* update the data read pointer */
                    g_pc4DataRead_d += (VEGAS_NUM_TAPS * g_iNumSubBands * g_nchan);
                    if (iProcData == g_iBlockInDataSize)
                    {
                        break;
                    }

                    /* Calculate input heap addresses for the next round of processing */
                    heap_in += (VEGAS_NUM_TAPS * num_in_heaps_per_proc);
                    if (heap_in > num_in_heaps_gpu_buffer)
                    {
                        /* This is not supposed to happen (but may happen if odd number of pkts are dropped
                           right at the end of the buffer, so we therefore do not exit) */
                        (void) fprintf(stderr,
                                       "WARNING: Heap count %d exceeds available number of heaps %d!\n",
                                       heap_in,
                                       num_in_heaps_gpu_buffer);
                    }
                    heap_addr_in = (char*)(vegas_databuf_data(db_in, curblock_in) +
                                        sizeof(struct time_spead_heap) * heap_in);
                    continue;
                }
            }
            #endif

            CopyDataForFFT<<<g_dimGCopy, g_dimBCopy>>>(g_pc4DataStage_d,
                                                       g_pf4FFTIn_d);
            CUDASafeCall(cudaThreadSynchronize());
            iCUDARet = cudaGetLastError();
            if (iCUDARet != cudaSuccess)
            {
                (void) fprintf(stderr,
                               "ERROR: File <%s>, Line %d: %s\n",
                               __FILE__,
                               __LINE__,
                               cudaGetErrorString(iCUDARet));
                run = 0;
                break;
            }
             
            iRet = do_fft();
            if (iRet != VEGAS_OK)
            {
                (void) fprintf(stderr, "ERROR: FFT failed!\n");
                run = 0;
                break;
            }

            /* Accumulate power x, power y, stokes real and imag */
            iRet = accumulate();
            if (iRet != VEGAS_OK)
            {
                (void) fprintf(stderr, "ERROR: Accumulation failed!\n");
                run = 0;
                break;
            }

            if (0 == g_iAccID)
            {
                ++g_iSpecPerAcc;
            }

            if (g_iSpecPerAcc == acc_len)
            {
                /* dump to buffer */
                heap_addr_out = (char*)(vegas_databuf_data(db_out, g_iPFBCurBlockOut) +
                                    sizeof(struct freq_spead_heap) * g_iHeapOut);
                freq_heap_out = (struct freq_spead_heap*)(heap_addr_out);
                payload_addr_out = (char*)(vegas_databuf_data(db_out, g_iPFBCurBlockOut) +
                                    sizeof(struct freq_spead_heap) * MAX_HEAPS_PER_BLK +
                                    (index_out->heap_size - sizeof(struct freq_spead_heap)) * g_iHeapOut);
         
                /* Write new heap header fields */
                freq_heap_out->time_cntr_id = 0x20;
                freq_heap_out->time_cntr = first_time_heap_in_accum->time_cntr;
                freq_heap_out->spectrum_cntr_id = 0x21;
                freq_heap_out->spectrum_cntr = g_iTotHeapOut;
                freq_heap_out->integ_size_id = 0x22;
                freq_heap_out->integ_size = g_iSpecPerAcc;
                freq_heap_out->mode_id = 0x23;
                freq_heap_out->mode = first_time_heap_in_accum->mode;
                freq_heap_out->status_bits_id = 0x24;
                freq_heap_out->status_bits = first_time_heap_in_accum->status_bits;
                freq_heap_out->payload_data_off_addr_mode = 0;
                freq_heap_out->payload_data_off_id = 0x25;
                freq_heap_out->payload_data_off = 0;

                /* Update output index */
                index_out->cpu_gpu_buf[g_iHeapOut].heap_valid = 1;
                index_out->cpu_gpu_buf[g_iHeapOut].heap_cntr = g_iTotHeapOut;
                index_out->cpu_gpu_buf[g_iHeapOut].heap_rcvd_mjd =
                         index_in->cpu_gpu_buf[g_iFirstHeapIn].heap_rcvd_mjd ;

                //iRet = get_accumulated_spectrum_from_device(payload_addr_out);
                if (iRet != VEGAS_OK)
                {
                    (void) fprintf(stderr, "ERROR: Getting accumulated spectrum failed!\n");
                    run = 0;
                    break;
                }

                ++g_iHeapOut;
                ++g_iTotHeapOut;

                /* zero accumulators */
                zero_accumulator();
                /* reset time */
                if ((g_iNumChanBlocks - 1) == g_iAccID)
                {
                    g_iSpecPerAcc = 0;
                }
            }

            /* Calculate input heap addresses for the next round of processing */
            heap_in += num_in_heaps_per_proc;
            heap_addr_in = (char*)(vegas_databuf_data(db_in, curblock_in) +
                                sizeof(struct time_spead_heap) * heap_in);
            if (0 == g_iSpecPerAcc)
            {
                first_time_heap_in_accum = (struct time_spead_heap*)(heap_addr_in);
                g_iFirstHeapIn = heap_in;
                g_dFirstHeapRcvdMJD = index_in->cpu_gpu_buf[g_iFirstHeapIn].heap_rcvd_mjd;
            }

            /* if output block is full */
            if (g_iHeapOut == g_iMaxNumHeapOut)
            {
                /* Set the number of heaps written to this block */
                index_out->num_heaps = g_iHeapOut;

                /* Mark output buffer as filled */
                vegas_databuf_set_filled(db_out, g_iPFBCurBlockOut);

                printf("Debug: vegas_gpu_thread going to next output block\n");

                /* Note current output block */
                pthread_cleanup_push((void (*) (void *))&vegas_status_unlock, (void *) &st);
                vegas_status_lock(&st);
                hputi4(st.buf, "PFBBLKOU", g_iPFBCurBlockOut);
                vegas_status_unlock(&st);
                pthread_cleanup_pop(0);

                /*  Wait for next output block */
                g_iPFBCurBlockOut = (g_iPFBCurBlockOut + 1) % db_out->n_block;
                while ((vegas_databuf_wait_free(db_out, g_iPFBCurBlockOut)!=0) && run) {
                    pthread_cleanup_push((void (*)(void *))&vegas_status_unlock, (void *) &st);
                    vegas_status_lock(&st);

                    hputs(st.buf, STATUS_KEY, "blocked");
                    vegas_status_unlock(&st);
                    pthread_cleanup_pop(0);
                }

                g_iHeapOut = 0;

                hdr_out = vegas_databuf_header(db_out, g_iPFBCurBlockOut);
                index_out = (struct databuf_index*)vegas_databuf_index(db_out, g_iPFBCurBlockOut);
                memcpy(hdr_out, vegas_databuf_header(db_in, curblock_in),
                        VEGAS_STATUS_SIZE);

                /* Set basic params in output index */
                index_out->heap_size = sizeof(struct freq_spead_heap) + (g_iNumSubBands * g_nchan * sizeof(float4));
            }

            /* update the accumulator ID */
            ++g_iAccID;

            fprintf(stderr, "%ld\n", g_pc4DataRead_d);
            if ((g_pc4DataRead_d - g_pc4Data_d) == g_iBlockInDataSize)
            {
                is_proc_done = 1;
                fprintf(stderr, "****************PROC DONE!********************\n");
                break;
            }
        }
    }

    return;
}

void read_gpu_input_data()
{
    /* write new data to the write buffer */
    /* strided copy of g_iNumConcFFT channels (* 2 polarizations) */
    CUDASafeCall(cudaMemcpy2D(g_pc4DataStage_d,
                              g_iNumConcFFT * sizeof(char4),     /* dest. pitch */
                              g_pc4DataRead_d,
                              g_iNumSubBands * sizeof(char4),    /* src. pitch */
                              g_iNumConcFFT * sizeof(char4),
                              g_nchan,
                              cudaMemcpyDeviceToDevice));
    /* update the read pointer to where data needs to be read in from, in the
       next read */
    if (g_iNumConcFFT == g_iNumSubBands)
    {
        g_pc4DataRead_d += (g_iNumSubBands * g_nchan);
    }
    else
    {
        if ((g_iNumChanBlocks - 1) == g_iReadID)
        {
            g_pc4DataRead_d += ((g_iNumConcFFT % g_iNumSubBands));
            g_pc4DataRead_d += (g_iNumSubBands * (g_nchan - 1));
        }
        else
        {
            g_pc4DataRead_d += g_iNumConcFFT;
        }
    }

    g_iReadID = (g_iReadID + 1) % g_iNumChanBlocks;

    return;
}

/* function that performs the FFT */
int do_fft()
{
    cufftResult iCUFFTRet = CUFFT_SUCCESS;

    /* execute plan */
    iCUFFTRet = cufftExecC2C(g_stPlan,
                             (cufftComplex*) g_pf4FFTIn_d,
                             (cufftComplex*) g_pf4FFTOut_d,
                             CUFFT_FORWARD);
    if (iCUFFTRet != CUFFT_SUCCESS)
    {
        (void) fprintf(stderr, "ERROR: FFT failed!");
        run = 0;
        return VEGAS_ERR_GEN;
    }

    return VEGAS_OK;
}

int accumulate()
{
    cudaError_t iCUDARet = cudaSuccess;

    Accumulate<<<g_dimGAccum, g_dimBAccum>>>(g_pf4FFTOut_d,
                                             g_ppf4SumStokes_d[g_iAccID]);
    CUDASafeCall(cudaThreadSynchronize());
    iCUDARet = cudaGetLastError();
    if (iCUDARet != cudaSuccess)
    {
        (void) fprintf(stderr, cudaGetErrorString(iCUDARet));
        run = 0;
        return VEGAS_ERR_GEN;
    }

    return VEGAS_OK;
}

void zero_accumulator()
{
    CUDASafeCall(cudaMemset(g_ppf4SumStokes_d[g_iAccID],
                            '\0',
                            (g_iNumConcFFT
                             * g_nchan
                             * sizeof(float4))));

    return;
}

int get_accumulated_spectrum_from_device(char *out)
{
    CUDASafeCall(cudaMemcpy(out,
                            g_ppf4SumStokes_d[g_iAccID],
                            (g_iNumSubBands
                             * g_nchan
                             * sizeof(float4)),
                            cudaMemcpyDeviceToHost));

    return VEGAS_OK;
}

/*
 * function to be used to check if any heap within the current PFB is invalid,
 * in which case, the entire PFB should be discarded.
 * NOTE: this function does not check ALL heaps - it returns at the first
 * invalid heap.
 */
int is_valid(int heap_start, int num_heaps)
{
    for (int i = heap_start; i < (heap_start + num_heaps); ++i)
    {
        if (!g_auiHeapValid[i])
        {
            return FALSE;
        }
    }

    return TRUE;
}


void __CUDASafeCall(cudaError_t iCUDARet,
                    const char* pcFile,
                    const int iLine,
                    void (*pCleanUp)(void))
{
    if (iCUDARet != cudaSuccess)
    {
        (void) fprintf(stderr,
                       "ERROR: File <%s>, Line %d: %s\n",
                       pcFile,
                       iLine,
                       cudaGetErrorString(iCUDARet));
        run = 0;
        return;
    }

    return;
}

/* 
 * Frees up any allocated memory.
 */
void cleanup_gpu()
{
    /* free memory */
    if (g_pc4InBuf != NULL)
    {
        free(g_pc4InBuf);
        g_pc4InBuf = NULL;
    }
    if (g_pc4Data_d != NULL)
    {
        (void) cudaFree(g_pc4Data_d);
        g_pc4Data_d = NULL;
    }
    if (g_pc4DataStage_d != NULL)
    {
        (void) cudaFree(g_pc4DataStage_d);
        g_pc4DataStage_d = NULL;
    }
    if (g_pf4FFTIn_d != NULL)
    {
        (void) cudaFree(g_pf4FFTIn_d);
        g_pf4FFTIn_d = NULL;
    }
    if (g_pf4FFTOut_d != NULL)
    {
        (void) cudaFree(g_pf4FFTOut_d);
        g_pf4FFTOut_d = NULL;
    }
    for (int i = 0; i < g_iNumChanBlocks; ++i)
    {
        if (g_ppf4SumStokes[i] != NULL)
        {
            free(g_ppf4SumStokes[i]);
            g_ppf4SumStokes[i] = NULL;
        }
        if (g_ppf4SumStokes_d[i] != NULL)
        {
            (void) cudaFree(g_ppf4SumStokes_d[i]);
            g_ppf4SumStokes_d[i] = NULL;
        }
    }
    if (g_ppf4SumStokes != NULL)
    {
        free(g_ppf4SumStokes);
        g_ppf4SumStokes = NULL;
    }
    if (g_ppf4SumStokes_d != NULL)
    {
        free(g_ppf4SumStokes_d);
        g_ppf4SumStokes_d = NULL;
    }

    /* destroy plan */
    /* TODO: check if plan exists */
    (void) cufftDestroy(g_stPlan);

    return;
}

