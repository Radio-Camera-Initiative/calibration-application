#include <cstdio>
#include <cuda_runtime.h>

#include "calibration.cuh"

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
    int* vis
) {
    // TODO: store mask or vis in shared memory for quicker access
    int mchannel = blockIdx.x * nbaseline * npol;
    int channel = blockIdx.x * nbaseline * npol * 2;

    for (int i = threadIdx.x; i < nbaseline; i += blockDim.x) {
        int mbaseline = (i * npol);
        int baseline = (i * npol) * 2;
        // each polarization will be set separately

        // if mask bit is 0, we want to keep the same number. make a mask of 1s
        //     and bitwise AND
        // if mask bit is 1, we want to erase the whole thing. make a mask of 0s
        //     and bitwise AND

        // use mask bit to make temp integer.
        int m1 = ((int) mask[mchannel + mbaseline]) - 1;
        vis[channel + baseline] &= m1;
        vis[channel + baseline + IM] &= m1;
        int m2 = ((int) mask[mchannel + mbaseline + 1]) - 1;
        vis[channel + baseline + 2] &= m2;
        vis[channel + baseline + 2 + IM] &= m2;
        int m3 = ((int) mask[mchannel + mbaseline + 2]) - 1;
        vis[channel + baseline + 4] &= m3;
        vis[channel + baseline + 4 + IM] &= m3;
        int m4 = ((int) mask[mchannel + mbaseline + 3]) - 1;
        vis[channel + baseline + 6] &= m4;
        vis[channel + baseline + 6 + IM] &= m4;

    }

    return;
}

// Make a function to move memory to the GPU, but unneeded with Bifrost as
// everything is already on the GPU
void call_flag_mask_kernel(
    int nchan,
    int nbaseline,
    int npol,
    const bool* mask,
    int* vis
) {
    int* gpu_vis;
    // &gpu_vis gives reference to piece of memory where pointer is stored
    cudaMalloc((void**)&gpu_vis, nchan * nbaseline * npol * 2 * sizeof(int));
    cudaMemcpy(gpu_vis, vis, nchan * nbaseline * npol * 2 * sizeof(int), cudaMemcpyHostToDevice);

    bool* gpu_mask;
    cudaMalloc((void**)&gpu_mask, nchan * nbaseline* npol * 2 * sizeof(bool));
    cudaMemcpy(gpu_mask, mask, nchan * nbaseline * npol * 2 * sizeof(bool), cudaMemcpyHostToDevice);

    unsigned int blocks = nchan;
    unsigned int threads_per_block = MAXTHREADS;

    flag_mask_kernel<<<blocks, threads_per_block>>> (nchan, nbaseline, npol, gpu_mask, gpu_vis);

    // Check for errors on kernel call
    cudaError err = cudaGetLastError();
    if (cudaSuccess != err)
        fprintf(stderr, "Error %s\n", cudaGetErrorString(err));
    else
        fprintf(stderr, "No kernel error detected\n");

     cudaMemcpy(vis, gpu_vis, nchan * nbaseline * npol * 2 * sizeof(int), cudaMemcpyDeviceToHost);

     cudaFree(gpu_vis);
     cudaFree(gpu_mask);
}