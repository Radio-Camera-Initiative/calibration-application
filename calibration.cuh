#define IM 1
#define CM 2

void call_flag_mask_kernel(
    int dim1,
    int dim2,
    int npol,
    const bool* mask,
    float* vis
);

void call_jones_kernel(
    int nchan,
    int nbaseline,
    int npol,
    int nant, 
    float* vis,
    int* ant,
    float* jones
);