#include <algorithm>
#include <iostream>
#include <assert.h>
#include <complex>
#include <cuda_runtime.h>

#include "lender.hpp"
#include "calibration.cuh"

#define CM 2

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
            for (int p = 0; p < npol * CM; p += CM) {
                std::cout << 
                    vis[(c * nbaseline * npol * CM) + (b * npol * CM) + p] 
                    << " + " << 
                    vis[(c * nbaseline * npol * CM) + (b * npol * CM) + p + IM] 
                    << 'i' << ' ';
            }
            std::cout << "] ";
        }
        std::cout << std::endl;
    }
}

void print_jones(int nchan, int nant, int npol, float* jones) {
    std::cout << "Jones" << std::endl;
    for (int c = 0; c < nchan; c++) {
        for (int b = 0; b < nant; b++) {
            std::cout << "[ ";
            for (int p = 0; p < npol * CM; p += CM) {
                std::cout << 
                    jones[(c * nant * npol * CM) + (b * npol * CM) + p] 
                    << " + " << 
                    jones[(c * nant * npol * CM) + (b * npol * CM) + p + IM] 
                    << 'i' << ' ';
            }
            std::cout << "] ";
        }
        std::cout << std::endl;
    }
}

/* ASSUMES vis 1 OR 0 */
void check_flags(int size, bool* mask, float* vis) {
    for (int i = 0; i < size; i++) {
        if (mask[i]) {
            assert(vis[i * CM] == 0.0);
            assert(vis[i * CM + 1] == 0.0);
        } else {
            assert(vis[i * CM] == 1.0);
            assert(vis[i * CM + 1] == 1.0);
        }
    }
}

void flag_chan1_pol2(int nchan, int nbaseline, int npol, bool* mask, float* vis) {

    std::cout << "set all of channel 1, polarization 2 to true" << std::endl;
    for (int i = 0; i < nbaseline; i++) {
        mask[(1 * nbaseline * npol) + (i * npol) + 2] = true;
    }

    // fill with 1s
    std::fill_n(vis, nchan * nbaseline * npol * CM, 1.0);

    // now that everything is initialized, run the GPU
    call_flag_mask_kernel(nchan, nbaseline, npol, mask, vis);

    // check vis after the kernel to see if correct channel is 0
    check_flags(npol * nbaseline * nchan, mask, vis);

    std::cout << "Test Passed" << std::endl;
}

void flag_chan2(int nchan, int nbaseline, int npol, bool* mask, float* vis) {

    std::cout << "set all of channel 2 to true" << std::endl;
    for (int i = 0; i < nbaseline; i++) {
        mask[(1 * nbaseline * npol) + (i * npol)] = true;
        mask[(1 * nbaseline * npol) + (i * npol) + 1] = true;
        mask[(1 * nbaseline * npol) + (i * npol) + 2] = true;
        mask[(1 * nbaseline * npol) + (i * npol) + 3] = true;
    }

    // fill with 1s
    std::fill_n(vis, nchan * nbaseline * npol * CM, 1.0);

    // now that everything is initialized, run the GPU
    call_flag_mask_kernel(nchan, nbaseline, npol, mask, vis);

    // check vis after the kernel to see if correct channel is 0
    check_flags(npol * nbaseline * nchan, mask, vis);

    std::cout << "Test Passed" << std::endl;
}

void flag_pol01(int nchan, int nbaseline, int npol, bool* mask, float* vis) {

    std::cout << "set all of polarizations 0, 1 to true" << std::endl;
    for (int c = 0; c < nchan; c++) {
        for (int i = 0; i < nbaseline; i++) {
            mask[(c * nbaseline * npol) + (i * npol)] = true;
            mask[(c * nbaseline * npol) + (i * npol) + 1] = true;
        }
    }
    // fill with 1s
    std::fill_n(vis, nchan * nbaseline * npol * CM, 1.0);

    // now that everything is initialized, run the GPU
    call_flag_mask_kernel(nchan, nbaseline, npol, mask, vis);

    // check vis after the kernel to see if correct channel is 0
    check_flags(npol * nbaseline * nchan, mask, vis);

    std::cout << "Test Passed" << std::endl;
}

void test_flagging(int nchan, int nbaseline, int npol) {
    std::vector<size_t> shape {nchan, nbaseline, npol};
    std::shared_ptr<library<std::complex<float>>> vis_lib = 
          std::make_shared<library<std::complex<float>>>(shape, 1);
    std::shared_ptr<library<bool>> mask_lib = 
          std::make_shared<library<bool>>(shape, 1);
    auto vis_buf = vis_lib->fill();
    auto mask_buf = mask_lib->fill();

    bool* mask = mask_buf.get();
    memset(mask, 0x00, sizeof(bool)*mask_buf.size);
    float* vis = (float*) vis_buf.get();

    flag_chan1_pol2(nchan, nbaseline, npol, mask, vis);
    flag_chan2(nchan, nbaseline, npol, mask, vis);
    flag_pol01(nchan, nbaseline, npol, mask, vis);
}

void test_jones_identity(int nchan, int nbaseline, int npol, int nant) {
    float* vis = new float [nchan * nbaseline * npol * CM];
    float* orig = new float [nchan * nbaseline * npol * CM];
    // fill with 1s
    // std::fill_n(vis, nchan * nbaseline * npol * CM, 1);

    // fill with indices matrices
    for (int a = 0; a < nbaseline; a++) {
            int baseline = (a * nchan * npol * CM);
        for (int c = 0; c < nchan; c++) {
            int channel = (c * npol * CM); 
            for (int i = 0; i < npol * CM; i++) {
                vis[channel + baseline + i] = (float) (channel + baseline + i);
                orig[channel + baseline + i] = (float) (channel + baseline + i);
            }
        }
    }

    // fill with identity matrices
    // for (int c = 0; c < nchan; c++) {
    //     int channel = (c * npol * nant * CM);
    //     for (int a = 0; a < nbaseline; a++) {
    //         int baseline = (a * npol * CM);
    //         // make the identity matrix for each antenna
    //         vis[channel + baseline] = 1.0f;
    //         vis[channel + baseline + 6] = 1.0f;
    //     }
    // }

    print_visf(nchan, nbaseline, npol, vis);

    // make jones matrices

    float* jones = new float [nchan * nant * npol * CM];
    // fill with 1s
    // std::fill_n(jones, nchan * nant * npol * CM, 1);

    // fill with indicies
    // for (int c = 0; c < nchan; c++) {
    //     int channel = (c * npol * nant * CM);
    //     for (int a = 0; a < nant; a++) {
    //         int antenna = (a * npol * CM);
    //         for (int i = 0; i < npol * CM; i++) {
    //             jones[channel + antenna + i] = (float) i;
    //         }
    //     }
    // }

    // fill with identity matrices
    for (int c = 0; c < nchan; c++) {
        int channel = (c * npol * nant * CM);
        for (int a = 0; a < nant; a++) {
            int antenna = (a * npol * CM);
            // make the identity matrix for each antenna
            jones[channel + antenna] = 1.0f;
            jones[channel + antenna + 1] = 0.0f;
            jones[channel + antenna + 2] = 0.0f;
            jones[channel + antenna + 3] = 0.0f;
            jones[channel + antenna + 4] = 0.0f;
            jones[channel + antenna + 5] = 0.0f;
            jones[channel + antenna + 6] = 1.0f;
            jones[channel + antenna + 7] = 0.0f;
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

    int nchan = 62128;
    int nant = 3;
    int nbaseline = 192;
    int npol = 4;

    test_flagging(nchan, nbaseline, npol);

    nchan = 5;
    nbaseline = nant * (nant - 1) / 2;

    test_jones_identity(nchan, nbaseline, npol, nant);

}