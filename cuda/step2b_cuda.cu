// step2b_cuda.cu — Fortran-style STEP2B (uiuj + full-grid FFT + zero middle)
// -------------------------------------------------------------------------
// Fortran reference (visasub.f):
//
//   SUBROUTINE STEP2B(NX,NZ,UC,UR,TFFTXZ,PREX,PREZ)
//     REAL UR(3*NX/2+2,3*NZ/2,3)
//     ...
// C   calculate uiuj and put into UR(x,z,1-3)
//     DO 700 Z=1,3*NZ/2
//        DO 710 X=1,3*NX/2
//           UR(X,Z,3)=UR(X,Z,1)*UR(X,Z,2)
//           UR(X,Z,1)=UR(X,Z,1)*UR(X,Z,1)
//           UR(X,Z,2)=UR(X,Z,2)*UR(X,Z,2)
// 710     CONTINUE
// 700  CONTINUE
//
// C   transform uiuj back to a-g space, i.e. to uiuj_th
//     DO 800 I=1,3
//        CALL VRFFTF(...,3*NX/2,3*NZ/2,...)
//        CALL VCFFTF(...,3*NZ/2,NX/2,...)
// 800  CONTINUE
//
// C   put the 'extra' fourier coeff. to zero (the middle one)
//     DO 900 I=1,3
//        DO 910 X=1,NX/2
//           UC(X,NZ+1,I)=0
// 910     CONTINUE
// 900  CONTINUE
//
// On CUDA we mirror this with:
//
//   • UR_full : real   [3][NZ_full][NX_full]  (3N/2 × 3N/2 grid)
//   • UC_full : complex[3][NZ_full][NK_full] (3N/2 × (3N/4+1) grid)
//
// where layout is [comp][z][x] in both cases.
//
// We rely on vfft_full_forward_ur_full_to_uc_full(S) to perform the
// full-grid VRFFTF + VCFFTF equivalent using CUFFT plans created in
// dnsCudaCreate().
//
// ---------------------------------------------------------------------

#include "cuda_dns.h"
#include <cuda_runtime.h>
#include <cstdio>

// Forward declaration from vfft_cuda.cu (existing full-grid FFT helper)
extern void vfft_full_forward_ur_full_to_uc_full(DnsDeviceState *S);

// ---------------------------------------------------------------------
// Kernel 1: build uiuj on the full 3/2 grid
//
// UR_full layout in memory (float):
//   size per component plane: NX_full * NZ_full
//   index = comp * plane + z * NX_full + x
//
// Before STEP2B:
//   UR(:,:,0) = u(x,z)
//   UR(:,:,1) = w(x,z)
//   UR(:,:,2) = (don't care)
//
// After this kernel (Fortran's UIUJ build):
//   UR(:,:,0) = u^2
//   UR(:,:,1) = w^2
//   UR(:,:,2) = u * w
// ---------------------------------------------------------------------
__global__
void k_step2b_build_uiuj(real *ur_full,
                         int NX_full,
                         int NZ_full)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x; // 0..NX_full-1
    int z = blockIdx.y * blockDim.y + threadIdx.y; // 0..NZ_full-1

    if (x >= NX_full || z >= NZ_full)
        return;

    const size_t plane = (size_t)NX_full * (size_t)NZ_full;
    const size_t base  = (size_t)z * (size_t)NX_full + (size_t)x;

    const size_t idx_u    = 0 * plane + base;  // UR(x,z,1) in Fortran
    const size_t idx_w    = 1 * plane + base;  // UR(x,z,2)
    const size_t idx_uw   = 2 * plane + base;  // UR(x,z,3)

    real u = ur_full[idx_u];
    real w = ur_full[idx_w];

    // Fortran order:
    //   UR(:,:,3) = u*w
    //   UR(:,:,1) = u*u
    //   UR(:,:,2) = w*w
    ur_full[idx_uw] = u * w;
    ur_full[idx_u]  = u * u;
    ur_full[idx_w]  = w * w;
}

// ---------------------------------------------------------------------
// Helper for UC_full indexing: [comp][z][kx]
// ---------------------------------------------------------------------
__device__ inline size_t uc_full_idx(int kx, int z, int c,
                                     int NK_full, int NZ_full)
{
    // comp-major, then z, then kx:
    // idx = (c * NZ_full + z) * NK_full + kx
    return ((size_t)c * (size_t)NZ_full + (size_t)z) * (size_t)NK_full + (size_t)kx;
}

// ---------------------------------------------------------------------
// Kernel 2: zero the "middle" Fourier coefficient UC(X,NZ+1,I)
//           for X <= NX/2 and I = 1..3 (comp = 0..2).
//
// In Fortran:
//   DO I=1,3
//      DO X=1,NX/2
//         UC(X,NZ+1,I) = 0
//      END DO
//   END DO
//
// Here:
//   • NX_half = NX/2  (Nbase/2)
//   • NZ      = N     (Nbase)
//   • Z index for NZ+1 (1-based) is z_mid = NZ (0-based).
// ---------------------------------------------------------------------
__global__
void k_step2b_zero_middle(cplx *uc_full,
                          int NX_half,
                          int NZ,        // base N
                          int NK_full,
                          int NZ_full)
{
    int kx = blockIdx.x * blockDim.x + threadIdx.x; // 0..NX_half-1
    if (kx >= NX_half)
        return;

    const int z_mid = NZ;  // 0-based index for Z = NZ+1 (1-based)

    // I = 1..3  → comp = 0,1,2
    for (int c = 0; c < 3; ++c) {
        size_t idx = uc_full_idx(kx, z_mid, c, NK_full, NZ_full);
        uc_full[idx].x = 0.0f;
        uc_full[idx].y = 0.0f;
    }
}

// ---------------------------------------------------------------------
// Host entry point: dnsCudaStep2B
//
// Mirrors Fortran STEP2B:
//
//   1) Build uiuj in UR(x,z,1..3)
//   2) VRFFTF + VCFFTF on UR(.,.,I) for I=1..3  → UC(.,.,I)
//   3) Zero UC(X,NZ+1,I) for X<=NX/2, I=1..3
// ---------------------------------------------------------------------
void dnsCudaStep2B(DnsDeviceState *S)
{
    if (!S) return;

    const int NX_full = S->NX_full;   // 3*N/2
    const int NZ_full = S->NZ_full;   // 3*N/2

    // 1) Build uiuj (UR(:,:,0..2) = u^2, w^2, u*w)
    dim3 block(32, 8);
    dim3 grid((NX_full + block.x - 1) / block.x,
              (NZ_full + block.y - 1) / block.y);

    k_step2b_build_uiuj<<<grid, block>>>(S->d_ur_full,
                                         NX_full,
                                         NZ_full);

    // 2) Full-grid forward FFT: UR_full → UC_full (3 components)
    //    This is the CUDA equivalent of Fortran's:
    //      VRFFTF (X-direction, 3*NX/2)
    //      VCFFTF (Z-direction, 3*NZ/2)
    //
    //    vfft_full_forward_ur_full_to_uc_full(S) is assumed to:
    //      • use CUFFT_R2C along X for all 3 components, and
    //      • then CUFFT_C2C (forward) along Z for all 3 components.
    vfft_full_forward_ur_full_to_uc_full(S);

    // 3) Zero UC(X,NZ+1,I) for X<=NX/2, I=1..3
    const int NX_half = S->Nbase / 2;  // NX/2
    const int NZ      = S->Nbase;      // NZ

    dim3 block2(64);
    dim3 grid2((NX_half + block2.x - 1) / block2.x);

    k_step2b_zero_middle<<<grid2, block2>>>(S->d_uc_full,
                                            NX_half,
                                            NZ,
                                            S->NK_full,
                                            S->NZ_full);
}
