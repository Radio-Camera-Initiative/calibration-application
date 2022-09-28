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

void check_same (int nbaseline, int nchan, int npol, float* vis, float* orig) {
    for (int a = 0; a < nbaseline; a++) {
            int baseline = (a * nchan * npol * CM);
        for (int c = 0; c < nchan; c++) {
            int channel = (c * npol * CM); 
            for (int i = 0; i < npol * CM; i++) {
                assert(vis[channel + baseline + i] == orig[channel + baseline + i]);
            }
        }
    }
}

void test_jones_identity(int nchan, int nbaseline, int npol, int nant) {
    float* vis = new float [nchan * nbaseline * npol * CM];
    float* orig = new float [nchan * nbaseline * npol * CM];

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

    // make jones matrices
    float* jones = new float [nchan * nant * npol * CM];

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

    check_same(nbaseline, nchan, npol, vis, orig);

    std::cout << "Jones Identity test Passed" << std::endl;

    delete[] vis;
    delete[] ants;
    delete[] jones;
    delete[] orig;
}

void test_jones(int nchan, int nbaseline, int npol, int nant) {
    float* vis = new float [nchan * nbaseline * npol * CM];
    float* orig = new float [nchan * nbaseline * npol * CM];

    // fill all vis matrices same way
    for (int m = 0; m < nbaseline * nchan * npol * CM; m += 8) {
        vis[m ] = 2.0f;
        vis[m + 1] = 4.0f;
        vis[m + 2] = 5.0f;
        vis[m + 3] = 3.0f;
        vis[m + 4] = 1.0f;
        vis[m + 5] = 9.0f;
        vis[m + 6] = 6.0f;
        vis[m + 7] = 3.0f;
    }
    for (int m = 0; m < nbaseline * nchan * npol * CM; m += 8) {
        orig[m ] = 110.0f;
        orig[m + 1] = 87.0f;
        orig[m + 2] = 202.0f;
        orig[m + 3] = 411.0f;
        orig[m + 4] = 418.0f;
        orig[m + 5] = 203.0f;
        orig[m + 6] = 958.0f;
        orig[m + 7] = 1135.0f;
    }

    // make jones matrices

    float* jones = new float [nchan * nant * npol * CM];

    for (int c = 0; c < nchan; c++) {
        int channel = (c * npol * nant * CM);
        for (int a = 0; a < nant; a++) {
            int antenna = (a * npol * CM);
            // make the matrix for each antenna
            jones[channel + antenna] = 0.0f;
            jones[channel + antenna + 1] = 1.0f;
            jones[channel + antenna + 2] = 2.0f;
            jones[channel + antenna + 3] = 3.0f;
            jones[channel + antenna + 4] = 4.0f;
            jones[channel + antenna + 5] = 5.0f;
            jones[channel + antenna + 6] = 6.0f;
            jones[channel + antenna + 7] = 7.0f;
        }
    }

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

    check_same(nbaseline, nchan, npol, vis, orig);

    std::cout << "Jones random number test Passed" << std::endl;

    delete[] vis;
    delete[] ants;
    delete[] jones;
    delete[] orig;
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
    test_jones(nchan, nbaseline, npol, nant);

}