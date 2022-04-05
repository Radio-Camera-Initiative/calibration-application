#include <algorithm>
#include <iostream>
#include <assert.h>
#include <cuda_runtime.h>

#include "calibration.cuh"

void print_flag_mask(int nchan, int nbaseline, int npol, bool* mask) {
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
}

void print_visf(int nchan, int nbaseline, int npol, float* vis) {
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
}

void print_jones(int nchan, int nant, int npol, float* jones) {
    std::cout << "JOnes" << std::endl;
    for (int c = 0; c < nchan; c++) {
        for (int b = 0; b < nant; b++) {
            std::cout << "[ ";
            for (int p = 0; p < npol * 2; p += 2) {
                std::cout << 
                    jones[(c * nant * npol * 2) + (b * npol * 2) + p] 
                    << " + " << 
                    jones[(c * nant * npol * 2) + (b * npol * 2) + p + IM] 
                    << 'i' << ' ';
            }
            std::cout << "] ";
        }
        std::cout << std::endl;
    }
}

void test_flagging(int nchan, int nbaseline, int npol) {
    bool* mask = new bool [nchan * nbaseline * npol] {false};
    // set all of channel 1, polarization 2 to 1
    for (int i = 0; i < nbaseline; i++) {
        mask[(1 * nbaseline * npol) + (i * npol) + 2] = true;
    }
    print_flag_mask(nchan, nbaseline, npol, mask);

    float* vis = new float [nchan * nbaseline * npol * 2];
    // fill with 1s
    std::fill_n(vis, nchan * nbaseline * npol * 2, 1.0);

    print_visf(nchan, nbaseline, npol, vis);

    // now that everything is initialized, run the GPU
    call_flag_mask_kernel(nchan, nbaseline, npol, mask, vis);

    // check vis after the kernel to see if correct channel is 0
    
    print_visf(nchan, nbaseline, npol, vis);

    assert(vis[3] == 1.0);
    assert(vis[nbaseline] == 1.0);
    for (int i = 0; i < nbaseline; i++) {
        assert(vis[(1 * nbaseline * npol * 2) + (i * 2 * npol) + 4] == 0.0);
        assert(vis[(1 * nbaseline * npol * 2) + (i * 2 * npol) + 4 + IM] == 0.0);
    }

    delete[] vis;
    delete[] mask;
}

void test_jones_identity(int nchan, int nbaseline, int npol, int nant) {
    float* vis = new float [nchan * nbaseline * npol * 2];
    // fill with 1s
    // std::fill_n(vis, nchan * nbaseline * npol * 2, 1);

    // fill with identity matrices
    for (int c = 0; c < nchan; c++) {
        int channel = (c * npol * nant * 2);
        for (int a = 0; a < nbaseline; a++) {
            int baseline = (a * npol * 2);
            for (int i = 0; i < npol * 2; i++) {
                vis[channel + baseline + i] = (float) i;
            }
        }
    }

    // fill with identity matrices
    // for (int c = 0; c < nchan; c++) {
    //     int channel = (c * npol * nant * 2);
    //     for (int a = 0; a < nbaseline; a++) {
    //         int baseline = (a * npol * 2);
    //         // make the identity matrix for each antenna
    //         vis[channel + baseline] = 1.0f;
    //         vis[channel + baseline + 6] = 1.0f;
    //     }
    // }

    print_visf(nchan, nbaseline, npol, vis);

    // make jones matrices

    float* jones = new float [nchan * nant * npol * 2];
    // fill with 1s
    // std::fill_n(jones, nchan * nant * npol * 2, 1);

    // fill with indicies
    // for (int c = 0; c < nchan; c++) {
    //     int channel = (c * npol * nant * 2);
    //     for (int a = 0; a < nant; a++) {
    //         int antenna = (a * npol * 2);
    //         for (int i = 0; i < npol * 2; i++) {
    //             jones[channel + antenna + i] = (float) i;
    //         }
    //     }
    // }

    // fill with identity matrices
    for (int c = 0; c < nchan; c++) {
        int channel = (c * npol * nant * 2);
        for (int a = 0; a < nant; a++) {
            int antenna = (a * npol * 2);
            // make the identity matrix for each antenna
            jones[channel + antenna] = 1.0f;
            jones[channel + antenna + 6] = 1.0f;
        }
    }
    

    print_jones(nchan, nant, npol, jones);

    int* ants = new int [nbaseline * 2];
    int iter = 0;
    for (int i = 0; i < nant; i++) {
        for (int j = i+1; j < nant; j++) {
            ants[iter] = i;
            assert(iter < nbaseline * 2);
            iter += 1;
            ants[iter] = j;
            assert(iter < nbaseline * 2);
            iter += 1;
        }
    }

    // now that everything is initialized, run the GPU
    call_jones_kernel(nchan, nbaseline, npol, nant, vis, ants, jones);

    print_visf(nchan, nbaseline, npol, vis);

    delete[] vis;
    delete[] ants;
    delete[] jones;
}

// main function to test out the GPU function
int main(int argc, char **argv) {
    // set up size and testing arrays

    int nchan = 2;
    int nant = 3;
    int nbaseline = nant * (nant - 1) / 2;
    int npol = 4;

    test_flagging(nchan, nbaseline, npol);

    test_jones_identity(nchan, nbaseline, npol, nant);

}