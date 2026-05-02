// vfft_cuda.cu
// ---------------------------------------------------------------------
// Full 3/2-grid transforms between spectral UC_full and physical UR_full.
// Uses 2D batched cuFFT plans (all 3 components in one call).
// ---------------------------------------------------------------------

#include "cuda_dns.h"
#include <cuda_runtime.h>
#include <cufft.h>
#include <cstdio>

// Inverse: UC_full (complex, [comp][z][kx]) → UR_full (real, [comp][z][x])
// Processes all 3 components in a single batched 2D C2R call.
void vfft_full_inverse_uc_full_to_ur_full(DnsDeviceState *S)
{
    if (!S) return;

    cufftResult r = cufftExecC2R(
        S->plan_full_c2r_2d,
        reinterpret_cast<cufftComplex*>(S->d_uc_full),
        reinterpret_cast<cufftReal*   >(S->d_ur_full)
    );
    if (r != CUFFT_SUCCESS) {
        printf("vfft_full_inverse: cufftExecC2R failed, code=%d\n", r);
    }
}

// Forward: UR_full (real, [comp][z][x]) → UC_full (complex, [comp][z][kx])
// Processes all 3 components in a single batched 2D R2C call.
void vfft_full_forward_ur_full_to_uc_full(DnsDeviceState *S)
{
    if (!S) return;

    cufftResult r = cufftExecR2C(
        S->plan_full_r2c_2d,
        reinterpret_cast<cufftReal*   >(S->d_ur_full),
        reinterpret_cast<cufftComplex*>(S->d_uc_full)
    );
    if (r != CUFFT_SUCCESS) {
        printf("vfft_full_forward: cufftExecR2C failed, code=%d\n", r);
    }
}

// Forward path for STEP2B nonlinear products. STEP3 consumes only kx < N/2,
// so the z-direction FFT is pruned to those columns after the x R2C pass.
void vfft_step2b_forward_ur_full_to_uc_needed(DnsDeviceState *S)
{
    if (!S) return;

    cufftResult r = cufftExecR2C(
        S->plan_full_r2c_x,
        reinterpret_cast<cufftReal*   >(S->d_ur_full),
        reinterpret_cast<cufftComplex*>(S->d_uc_full)
    );
    if (r != CUFFT_SUCCESS) {
        printf("vfft_step2b_forward: cufftExecR2C x failed, code=%d\n", r);
        return;
    }

    const size_t plane = (size_t)S->NZ_full * (size_t)S->NK_full;
    for (int c = 0; c < 3; ++c) {
        cufftComplex* comp = reinterpret_cast<cufftComplex*>(S->d_uc_full + plane * c);
        r = cufftExecC2C(S->plan_full_c2c_z, comp, comp, CUFFT_FORWARD);
        if (r != CUFFT_SUCCESS) {
            printf("vfft_step2b_forward: cufftExecC2C z failed, code=%d\n", r);
            return;
        }
    }
}
