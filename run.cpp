#include <algorithm>
#include <iostream>
#include <assert.h>
#include <cuda_runtime.h>

#include "calibration.cuh"

// main function to test out the GPU function
int main(int argc, char **argv) {
    // set up size and testing arrays

    int nchan = 2;
    int nbaseline = 8;
    int npol = 4;

    bool* mask = new bool [nchan * nbaseline * npol] {false};
    // set all of channel 1, polarization 2 to 1
    for (int i = 0; i < nbaseline; i++) {
        mask[(1 * nbaseline * npol) + (i * npol) + 2] = true;
    }
    mask[0] = true;
    for (int c = 0; c < nchan; c++) {
        for (int b = 0; b < nbaseline; b++) {
            std::cout << "[ ";
            for (int p = 0; p < npol; p++) {
                std::cout << 
                    mask[(c * nbaseline * npol) + (b * npol) + p] 
                    << ' ';
            }
            std::cout << "] ";
        }
        std::cout << std::endl;
    }

    int* vis = new int [nchan * nbaseline * npol * 2];
    // fill with 1s
    std::fill_n(vis, nchan * nbaseline * npol * 2, 1);

    for (int c = 0; c < nchan; c++) {
        for (int b = 0; b < nbaseline; b++) {
            std::cout << "[ ";
            for (int p = 0; p < npol + 2; p += 2) {
                std::cout << 
                    vis[(c * nbaseline * npol * 2) + (b * npol * 2) + p] 
                    << " + " << 
                    vis[(c * nbaseline * npol * 2) + (b * npol * 2) + p + IM] 
                    << 'i' << ' ';
            }
            std::cout << "] ";
        }
        std::cout << std::endl;
    }


    // now that everything is initialized, run the GPU
    call_flag_mask_kernel(nchan, nbaseline, npol, mask, vis);

    // check vis after the kernel to see if correct channel is 0
    
    for (int c = 0; c < nchan; c++) {
        for (int b = 0; b < nbaseline; b++) {
            std::cout << "[ ";
            for (int p = 0; p < npol * 2; p += 2) {
                std::cout << 
                    vis[(c * nbaseline * npol * 2) + (b * npol * 2) + p] 
                    << " + " << 
                    vis[(c * nbaseline * npol * 2) + (b * npol * 2) + p + IM] 
                    << 'i' << ' ';
            }
            std::cout << "] ";
        }
        std::cout << std::endl;
    }

    assert(vis[3] == 1);
    assert(vis[nbaseline] == 1);
    for (int i = 0; i < nbaseline; i++) {
        assert(vis[(1 * nbaseline * npol * 2) + (i * 2 * npol) + 4] == 0);
        assert(vis[(1 * nbaseline * npol * 2) + (i * 2 * npol) + 4 + IM] == 0);
    }

    delete[] vis;
    delete[] mask;

}