// step3_cuda.cu — Fortran-style STEP3 in a single fused kernel.
#include "cuda_dns.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

__device__ inline size_t uc_full_idx(int x, int z, int c,
                                     int NK_full, int NZ_full)
{
    return ((size_t)c * (size_t)NZ_full + (size_t)z) * (size_t)NK_full + (size_t)x;
}

__device__ inline int om2_idx(int ix, int iz, int NX_half)
{
    return iz * NX_half + ix;
}

// ---------------------------------------------------------------------
// Fused STEP3 kernel:
//   1. Read UC_full(ix, z_spec, 0..2)  (uiuj from previous STEP2B)
//   2. Compute FN and Crank-Nicolson-update OM2, FNM1
//   3. Reconstruct UC_full(ix, iz, 0..1) from new OM2 (held in register)
// STEP3 geometry values are computed from integer wavenumbers in-register to
// avoid per-point alfa/gamma lookup traffic.
//
// Safety: each thread writes UC_full[ix, iz, 0..1]; reads UC_full[ix, z_spec, 0..2].
// Cross-thread: no thread ever reads what another thread writes in the same
// launch (writes go to components 0/1; reads include component 2 unchanged).
// ---------------------------------------------------------------------
__global__
void k_step3_fused(cplx *om2,
                   cplx *fnm1,
                   cplx *uc_full,
                   float divxz,
                   float visc,
                   float dt,
                   float cnm1,
                   int NX_half,
                   int NZ,
                   int NZ_full,
                   int NK_full)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix >= NX_half || iz >= NZ) return;

    int idx_om  = om2_idx(ix, iz, NX_half);
    cplx om_old = om2[idx_om];
    cplx fn_old = fnm1[idx_om];

    // --- phase 1: update OM2 and FNM1 ------------------------------
    float ax = (float)ix;
    float gz = (iz < (NZ / 2)) ? (float)iz : (float)(iz - NZ);
    float A2 = ax * ax;
    float G2 = gz * gz;
    float K2 = A2 + G2;
    float GA = gz * ax;
    float G2_minus_A2 = G2 - A2;
    int Zf = iz + 1;
    int z_spec = (Zf <= NZ / 2) ? iz : (iz + NZ / 2);

    size_t idx_uc1_in = uc_full_idx(ix, z_spec, 0, NK_full, NZ_full);
    size_t idx_uc2_in = uc_full_idx(ix, z_spec, 1, NK_full, NZ_full);
    size_t idx_uc3_in = uc_full_idx(ix, z_spec, 2, NK_full, NZ_full);

    cplx uc1 = uc_full[idx_uc1_in];
    cplx uc2 = uc_full[idx_uc2_in];
    cplx uc3 = uc_full[idx_uc3_in];

    cplx diff12;
    diff12.x = uc1.x - uc2.x;
    diff12.y = uc1.y - uc2.y;

    cplx FN;
    FN.x = (GA * diff12.x + G2_minus_A2 * uc3.x) * divxz;
    FN.y = (GA * diff12.y + G2_minus_A2 * uc3.y) * divxz;

    float VT  = 0.5f * visc * dt;
    float ARG = VT * K2;
    float DEN = 1.0f + ARG;

    float c1 = 1.0f - ARG;
    float c2 = 0.5f * dt * (2.0f + cnm1);
    float c3 = -0.5f * dt * cnm1;

    cplx num;
    num.x = c1 * om_old.x + c2 * FN.x + c3 * fn_old.x;
    num.y = c1 * om_old.y + c2 * FN.y + c3 * fn_old.y;

    cplx om_new;
    om_new.x = num.x / DEN;
    om_new.y = num.y / DEN;

    om2[idx_om]  = om_new;
    fnm1[idx_om] = FN;

    // --- phase 2: reconstruct UC(ix, iz, 0..1) from om_new ---------
    cplx out1, out2;
    out1.x = out1.y = 0.0f;
    out2.x = out2.y = 0.0f;

    if (ix >= 1) {
        float invK2 = 1.0f / (K2 + 1.0e-30f);

        cplx v;
        v.x = om_new.x * invK2;
        v.y = om_new.y * invK2;

        // UC1: gamma * v, then multiply by -i  → (gy, -gx)
        float gx = gz * v.x;
        float gy = gz * v.y;
        out1.x =  gy;
        out1.y = -gx;

        // UC2: alfa * v, then multiply by +i  → (-axi, axr)
        float axr = ax * v.x;
        float axi = ax * v.y;
        out2.x = -axi;
        out2.y =  axr;
    } else {
        // X = 1 branch
        float invG = (Zf >= 2 && fabsf(gz) > 0.0f) ? (1.0f / gz) : 0.0f;
        cplx v;
        v.x = om_new.x * invG;
        v.y = om_new.y * invG;
        out1.x =  v.y;
        out1.y = -v.x;
    }

    size_t idx_uc1_out = uc_full_idx(ix, iz, 0, NK_full, NZ_full);
    size_t idx_uc2_out = uc_full_idx(ix, iz, 1, NK_full, NZ_full);
    uc_full[idx_uc1_out] = out1;
    uc_full[idx_uc2_out] = out2;
}

// ---------------------------------------------------------------------
// Host entry point
// ---------------------------------------------------------------------
void dnsCudaStep3(DnsDeviceState *S)
{
    if (!S) return;

    int NX_half = S->Nbase / 2;
    int NZ      = S->Nbase;

    dim3 block(64, 4);
    dim3 grid( (NX_half + block.x - 1) / block.x,
               (NZ      + block.y - 1) / block.y );

    DNS_PHASE_TIME(DNS_PHASE_STEP3, {
        k_step3_fused<<<grid, block>>>(
            S->d_om2,
            S->d_fnm1,
            S->d_uc_full,
            S->step3_divxz,
            S->visc,
            S->dt,
            S->cnm1,
            NX_half,
            NZ,
            S->NZ_full,
            S->NK_full
        );
    });

    // Mirror Fortran: CNM1 = CN
    S->cnm1 = S->cn;
}
