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

/*  The assumed shape is as follows:
 *      Visibilities:
 *          (channels, baselines, polarizations) - float, float
 *      Antennas:
 *          (baselines, antennas) - int
 *      Jones:
 *          (channels, antennas, polarizations) - float, float
 */

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
    int jones_chan = blockIdx.x * nant * npol * 2;
    int vis_chan = blockIdx.x * nbaseline * npol * 2;

    for (int i = threadIdx.x; i < nbaseline; i += blockDim.x) {
        // [0+1i  2+3i]
        // [4+5i  6+7i]

        // access first matrix for matrixmul mat1 * mat2
        float mat1[8];
        // TODO: check order that reference is called
        float* matrix = &jones[jones_chan + (ant[(i * 2)] * npol * 2)];
        for (int j = 0; j < npol * 2; j++) {
            mat1[j] = matrix[j];
        }

        float mat2[8];
        matrix = &vis[vis_chan + (i * npol * 2)];
        for (int j = 0; j < npol * 2; j++) {
            mat2[j] = matrix[j];
        }
        float mat3[8];

        mat3[0] = ((mat1[0] * mat2[0]) - (mat1[1] * mat2[1])) +
            ((mat1[2] * mat2[4]) - (mat1[3] * mat2[5]));
        mat3[1] = ((mat1[0] * mat2[1]) + (mat1[1] * mat2[0])) +
            ((mat1[2] * mat2[5]) + (mat1[3] * mat2[4]));

        mat3[2] = ((mat1[0] * mat2[2]) - (mat1[1] * mat2[3])) +
            ((mat1[2] * mat2[6]) - (mat1[3] * mat2[7]));
        mat3[3] = ((mat1[0] * mat2[3]) + (mat1[1] * mat2[2])) +
            ((mat1[2] * mat2[7]) + (mat1[3] * mat2[6]));

        mat3[4] = ((mat1[4] * mat2[0]) - (mat1[5] * mat2[1])) +
            ((mat1[6] * mat2[4]) - (mat1[7] * mat2[5]));
        mat3[5] = ((mat1[4] * mat2[1]) + (mat1[5] * mat2[0])) +
            ((mat1[6] * mat2[5]) + (mat1[7] * mat2[4]));

        mat3[6] = ((mat1[4] * mat2[2]) - (mat1[5] * mat2[3])) +
            ((mat1[6] * mat2[6]) - (mat1[7] * mat2[7]));
        mat3[7] = ((mat1[4] * mat2[3]) + (mat1[5] * mat2[2])) +
            ((mat1[6] * mat2[7]) + (mat1[7] * mat2[6]));

        // access second matrix for matrixmul.
        // Also need to conjugate transpose mat1
        matrix = &jones[jones_chan + (ant[(i * 2) + 1] * npol * 2)];
        for (int j = 0; j < npol * 2; j++) {
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

        // mat3 * mat1
        mat2[0] = ((mat3[0] * mat1[0]) - (mat3[1] * mat1[1])) +
            ((mat3[2] * mat1[4]) - (mat3[3] * mat1[5]));
        mat2[1] = ((mat3[0] * mat1[1]) + (mat3[1] * mat1[0])) +
            ((mat3[2] * mat1[5]) + (mat3[3] * mat1[4]));

        mat2[2] = ((mat3[0] * mat1[2]) - (mat3[1] * mat1[3])) +
            ((mat3[2] * mat1[6]) - (mat3[3] * mat1[7]));
        mat2[3] = ((mat3[0] * mat1[3]) + (mat3[1] * mat1[2])) +
            ((mat3[2] * mat1[7]) + (mat3[3] * mat1[6]));

        mat2[4] = ((mat3[4] * mat1[0]) - (mat3[5] * mat1[1])) +
            ((mat3[6] * mat1[4]) - (mat3[7] * mat1[5]));
        mat2[5] = ((mat3[4] * mat1[1]) + (mat3[5] * mat1[0])) +
            ((mat3[6] * mat1[5]) + (mat3[7] * mat1[4]));

        mat2[6] = ((mat3[4] * mat1[2]) - (mat3[5] * mat1[3])) +
            ((mat3[6] * mat1[6]) - (mat3[7] * mat1[7]));
        mat2[7] = ((mat3[4] * mat1[3]) + (mat3[5] * mat1[2])) +
            ((mat3[6] * mat1[7]) + (mat3[7] * mat1[6]));

        // do final transform for iquv (technically need to divide by 2)
        mat1[0] = mat2[0] + mat2[6];
        mat1[1] = mat2[1] + mat2[7];
        mat1[2] = mat2[0] - mat2[6];
        mat1[3] = mat2[1] - mat2[7];
        mat1[4] = mat2[3] + mat2[5];
        mat1[5] = mat2[2] + mat2[4];
        mat1[6] = mat2[3] - mat2[5];
        mat1[7] = mat2[2] - mat2[4];

        // copy mat2 back into visibility
        for (int j = 0; j < npol * 2; j++) {
            vis[vis_chan + (i * npol * 2) + j] = mat1[j];
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
    cudaMalloc((void**)&gpu_vis, nchan * nbaseline * npol * 2 * sizeof(float));
    cudaMemcpy(gpu_vis, vis, nchan * nbaseline * npol * 2 * sizeof(float), cudaMemcpyHostToDevice);

    int* gpu_ant;
    cudaMalloc((void**)&gpu_ant, nbaseline * 2 * sizeof(int));
    cudaMemcpy(gpu_ant, ant, nbaseline * 2 * sizeof(int), cudaMemcpyHostToDevice);

    float* gpu_jones;
    cudaMalloc((void**)&gpu_jones, nchan * nant * npol * 2 * sizeof(float));
    cudaMemcpy(gpu_jones, jones, nchan * nant * npol * 2 * sizeof(float), cudaMemcpyHostToDevice);

    unsigned int blocks = nchan;
    unsigned int threads_per_block = MAXTHREADS;

    jones_kernel<<<blocks, threads_per_block>>> (nchan, nbaseline, npol, nant, gpu_vis, gpu_ant, gpu_jones);

    // Check for errors on kernel call
    cudaError err = cudaGetLastError();
    if (cudaSuccess != err)
        fprintf(stderr, "Error %s\n", cudaGetErrorString(err));
    else
        fprintf(stderr, "No kernel error detected\n");

     cudaMemcpy(vis, gpu_vis, nchan * nbaseline * npol * 2 * sizeof(int), cudaMemcpyDeviceToHost);

     cudaFree(gpu_vis);
     cudaFree(gpu_ant);
     cudaFree(gpu_jones);
}
