#include <cstdio>
#include <iostream>
#include <cuda_runtime.h>

#include "calibration.cuh"

#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

#define MAXTHREADS 1024
// This is the flagging mask application code for the GPU

/* The assumed shape is as follows:
 *     Visibilities:
 *         (channels, baselines, polarizations) - float, float
 *     Mask - same as visibilities:
 *         (channels, baselines, polarizations) - bit
 */

// GPU kernel declaration
/* Thread usage
 *
 * Have a block? for each channel -> the first thread checks if the channel is
 * flagged
 * If it is flagged, then all the threads in the block are each assigned to one
 * visibility in the channel to flag it. Not sure if need to flag all
 * polarizations? Then reassigned until for loop runs out.
 */
__global__ void flag_mask_kernel(
    int nchan,
    int nbaseline,
    int npol,
    const bool* mask,
    float* vis
) {
    // TODO: store mask or vis in shared memory for quicker access
    int mchannel = blockIdx.x * nbaseline * npol;
    int channel = blockIdx.x * nbaseline * npol * CM;

    for (int i = threadIdx.x; i < nbaseline; i += blockDim.x) {
        int mbaseline = (i * npol);
        int baseline = (i * npol) * CM;
        // each polarization will be set separately

        // use bool to make temp float.
        float m1 = static_cast<float>(!mask[mchannel + mbaseline]);
        vis[channel + baseline] *= m1;
        vis[channel + baseline + IM] *= m1;
        float m2 = static_cast<float>(!mask[mchannel + mbaseline + 1]);
        vis[channel + baseline + 2] *= m2;
        vis[channel + baseline + 2 + IM] *= m2;
        float m3 = static_cast<float>(!mask[mchannel + mbaseline + 2]);
        vis[channel + baseline + 4] *= m3;
        vis[channel + baseline + 4 + IM] *= m3;
        float m4 = static_cast<float>(!mask[mchannel + mbaseline + 3]);
        vis[channel + baseline + 6] *= m4;
        vis[channel + baseline + 6 + IM] *= m4;

    }

    return;
}

/*  The assumed shape is as follows:
 *      Visibilities:
 *          (channels, baselines, polarizations) - float, float
 *      Antennas:
 *          (baselines, antennas) - int
 *      Jones:
 *          (channels, antennas, polarizations) - float, float
 */

__device__ void mat_mul_complex(
    float answer[],
    float left[],
    float right[]
) {
    answer[0] = ((left[0] * right[0]) - (left[1] * right[1])) +
        ((left[2] * right[4]) - (left[3] * right[5]));
    answer[1] = ((left[0] * right[1]) + (left[1] * right[0])) +
        ((left[2] * right[5]) + (left[3] * right[4]));

    answer[2] = ((left[0] * right[2]) - (left[1] * right[3])) +
        ((left[2] * right[6]) - (left[3] * right[7]));
    answer[3] = ((left[0] * right[3]) + (left[1] * right[2])) +
        ((left[2] * right[7]) + (left[3] * right[6]));

    answer[4] = ((left[4] * right[0]) - (left[5] * right[1])) +
        ((left[6] * right[4]) - (left[7] * right[5]));
    answer[5] = ((left[4] * right[1]) + (left[5] * right[0])) +
        ((left[6] * right[5]) + (left[7] * right[4]));

    answer[6] = ((left[4] * right[2]) - (left[5] * right[3])) +
        ((left[6] * right[6]) - (left[7] * right[7]));
    answer[7] = ((left[4] * right[3]) + (left[5] * right[2])) +
        ((left[6] * right[7]) + (left[7] * right[6]));
}

__global__ void jones_kernel(
    int nchan,
    int nbaseline,
    int npol,
    int nant,
    float* vis,
    int* ant,
    float* jones
) {
    // TODO: put antennas and/or jones into shared mem for faster access
    int jones_chan = blockIdx.x * nant * npol * CM;
    int vis_chan = blockIdx.x * nbaseline * npol * CM;

    for (int i = threadIdx.x; i < nbaseline; i += blockDim.x) {
        // [0+1i  2+3i]
        // [4+5i  6+7i]

        // access first matrix for matrixmul mat1 * mat2
        float mat1[8];
        // TODO: check order that reference is called
        float* matrix = &jones[jones_chan + (ant[(i * CM)] * npol * CM)];
        for (int j = 0; j < npol * CM; j++) {
            mat1[j] = matrix[j];
        }

        float mat2[8];
        matrix = &vis[vis_chan + (i * npol * CM)];
        for (int j = 0; j < npol * CM; j++) {
            mat2[j] = matrix[j];
        }
        float mat3[8];

        mat_mul_complex(mat3, mat1, mat2);

        // access second matrix for matrixmul.
        // Also need to conjugate transpose mat1
        matrix = &jones[jones_chan + (ant[(i * CM) + 1] * npol * CM)];
        for (int j = 0; j < npol * CM; j++) {
            mat1[j] = matrix[j];
        }
        mat1[1] = -mat1[1];
        float temp_re = mat1[2];
        float temp_im = mat1[3];
        mat1[2] = mat1[4];
        mat1[3] = -mat1[5];
        mat1[4] = temp_re;
        mat1[5] = -temp_im;
        mat1[7] = -mat1[7];

        mat_mul_complex(mat2, mat3, mat1);

        // copy mat2 back into visibility
        for (int j = 0; j < npol * CM; j++) {
            vis[vis_chan + (i * npol * CM) + j] = mat2[j];
        }// TODO: how to do memcpy for 8 variables?
    }

}


// Make a function to move memory to the GPU, but unneeded with Bifrost as
// everything is already on the GPU
void call_flag_mask_kernel(
    int nchan,
    int nbaseline,
    int npol,
    const bool* mask,
    float* vis
) {
    std::clog << "dim 1: " << nchan << "; dim 2: " << nbaseline << "; dim 3: " << npol << std::endl;
    std::clog << ">>> Moving VIS to GPU" << std::endl;
    int size = nchan * nbaseline * npol;

    float* gpu_vis;
    // &gpu_vis gives reference to piece of memory where pointer is stored
    cudaMalloc((void**)&gpu_vis, size * CM * sizeof(float));
    cudaCheckErrors("cudaMalloc vis fail");
    cudaMemcpy(gpu_vis, vis, size * CM * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpy vis fail");

    bool* gpu_mask;
    cudaMalloc((void**)&gpu_mask, size * sizeof(bool));
    cudaCheckErrors("cudaMalloc mask fail");
    cudaMemcpy(gpu_mask, mask, size * sizeof(bool), cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpy mask fail");

    unsigned int blocks = nchan;
    unsigned int threads_per_block = MAXTHREADS;

    std::clog << ">>> Starting kernel" << std::endl;
    flag_mask_kernel<<<blocks, threads_per_block>>> (nchan, nbaseline, npol, gpu_mask, gpu_vis);

    // Check for errors on kernel call
    cudaError err = cudaGetLastError();
    if (cudaSuccess != err)
        fprintf(stderr, "Error %s\n", cudaGetErrorString(err));
    else
        fprintf(stderr, "No kernel error detected\n");

     cudaMemcpy(vis, gpu_vis, nchan * nbaseline * npol * CM * sizeof(float), cudaMemcpyDeviceToHost);

     cudaFree(gpu_vis);
     cudaFree(gpu_mask);
}

void call_jones_kernel(
    int nchan,
    int nbaseline,
    int npol,
    int nant, 
    float* vis,
    int* ant,
    float* jones
) {
    float* gpu_vis;
    // &gpu_vis gives reference to piece of memory where pointer is stored
    cudaMalloc((void**)&gpu_vis, nchan * nbaseline * npol * CM * sizeof(float));
    cudaMemcpy(gpu_vis, vis, nchan * nbaseline * npol * CM * sizeof(float), cudaMemcpyHostToDevice);

    int* gpu_ant;
    cudaMalloc((void**)&gpu_ant, nbaseline * CM * sizeof(int));
    cudaMemcpy(gpu_ant, ant, nbaseline * CM * sizeof(int), cudaMemcpyHostToDevice);

    float* gpu_jones;
    cudaMalloc((void**)&gpu_jones, nchan * nant * npol * CM * sizeof(float));
    cudaMemcpy(gpu_jones, jones, nchan * nant * npol * CM * sizeof(float), cudaMemcpyHostToDevice);

    unsigned int blocks = nchan;
    unsigned int threads_per_block = MAXTHREADS;

    jones_kernel<<<blocks, threads_per_block>>> (nchan, nbaseline, npol, nant, gpu_vis, gpu_ant, gpu_jones);

    // Check for errors on kernel call
    cudaError err = cudaGetLastError();
    if (cudaSuccess != err)
        fprintf(stderr, "Error %s\n", cudaGetErrorString(err));
    else
        fprintf(stderr, "No kernel error detected\n");

     cudaMemcpy(vis, gpu_vis, nchan * nbaseline * npol * CM * sizeof(int), cudaMemcpyDeviceToHost);

     cudaFree(gpu_vis);
     cudaFree(gpu_ant);
     cudaFree(gpu_jones);
}
