// This is the flagging mask application code for the GPU

/* The assumed shape is as follows:
 *     Visibilities:
 *         (channels, baselines, polarizations) - float, float
 *     Mask - same as visibilities:
 *         (channels, baselines, polarizations) - bit
 */

#include<algorithm>
#include<assert.h>
#define IM 1
#define MAXTHREADS 1024

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
    float* vis,
) {
    // TODO: store mask or vis in shared memory for quicker access
    int channel = blockIdx.x * nbaseline * npol * 2;

    for (int i = threadIdx.x; i < nbaseline; i += blockDim.x) {
        // each polarization will be set separately

        // if mask bit is 0, we want to keep the same number. make a mask of 1s
        //     and bitwise AND
        // if mask bit is 1, we want to erase the whole thing. make a mask of 0s
        //     and bitwise AND

        // use mask bit to make temp integer.
        int m1 = ((int) mask[channel + i]) - 1;
        vis[channel + i] &= m1;
        vis[channel + i + IM] &= m1;
        int m2 = ((int) mask[channel + i + 1]) - 1;
        vis[channel + i + 1] &= m2;
        vis[channel + i + 1 + IM] &= m2;
        int m3 = ((int) mask[channel + i + 2]) - 1;
        vis[channel + i + 2] &= m3;
        vis[channel + i + 2 + IM] &= m3;
        int m4 = ((int) mask[channel + i + 3]) - 1;
        vis[channel + i + 3] &= m4;
        vis[channel + i + 3 + IM] &= m4;
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
    float* vis,
) {
    float* gpu_vis;
    // &gpu_vis gives reference to piece of memory where pointer is stored
    cudaMalloc((void**)&gpu_vis, nchan * nbaseline * npol * 2 * sizeof(float));
    cudaMemcpy(gpu_vis, vis, nchan * nbaseline * npol * 2 * sizeof(float), cudaMemcpyHostToDevice);

    bool* gpu_mask;
    cudaMalloc((void**)&gpu_mask, nchan * nbaseline* npol * 2 * sizeof(bool));
    cudaMemcpy(gpu_mask, mask, nchan * nbaseline * npol * 2 * sizeof(bool), cudaMemcpyHostToDevice);

    blocks = nchan;
    threads_per_block = MAXTHREADS;

    flag_mask_kernel<<blocks, threads_per_block>> (nchan, nbaseline, npol, gpu_mask, gpu_vis);

    // Check for errors on kernel call
    cudaError err = cudaGetLastError();
    if (cudaSuccess != err)
        fprintf(stderr, "Error %s\n", cudaGetErrorString(err));
    else
        fprintf(stderr, "No kernel error detected\n");

     cudaMemcpy(vis, gpu_vis, nchan * nbaseline * npol * 2 * sizeof(float), cudaMemcpyDeviceToHost);

     cudaFree(gpu_vis);
     cudaFree(gpu_mask);
}

// main function to test out the GPU function
int main(int argc, char **argv) {
    // set up size and testing arrays

    int nchan = 8;
    int nbaseline = 16;
    int npol = 4;

    bool* mask = new bool [nchan * nbaseline * npol] {0};
    // set all of channel 3, polarization 2 to 1
    for (u_int i = 0; i < nbaseline; i++) {
        mask[(3 * nbaseline * npol) + i + 2] = 1;
    }

    float* vis = new float [nchan * nbaseline * npol * 2];
    // fill with 1s
    std::fill_n(vis, nchan * nbaseline * npol * 2, 1);


    // now that everything is initialized, run the GPU
    call_flag_mask_kernel(nchan, nbaseline, npol, mask, vis);

    // check vis after the kernel to see if correct channel is 0
    assert(vis[nbaseline + 3] == 1);
    assert(vis[5 * nbaseline * npol * 2 + (nbaseline/2)] == 1);
    for (u_int i = 0; i < nbaseline; i++) {
        assert(vis[(3 * nbaseline * npol * 2) + i + 2] == 0);
        assert(vis[(3 * nbaseline * npol * 2) + i + 2 + IM] == 0);
    }

}
