// calcom_cuda.cu — CUDA version of Fortran CALCOM (no Z-aliasing)
// -----------------------------------------------------------------
//
// Fortran reference (visasub.f):
//
//   SUBROUTINE CALCOM(NX,NZ,ALFA,GAMMA,UC,OM2)
//     IMPLICIT NONE
//     INTEGER NX,NZ
//     COMPLEX UC(3*NX/4+1,3*NZ/2,3),OM2(NX/2,NZ)
//     REAL ALFA(NX/2),GAMMA(NZ)
//     COMPLEX IM
//     PARAMETER (IM=(0.00,1.00))
//     INTEGER X,Z
// C       calculate omega2_th spectrally
//     DO 100 Z=1,NZ
//        DO 110 X=1,NX/2
//           OM2(X,Z)=IM*(GAMMA(Z)*UC(X,Z,1)-ALFA(X)*UC(X,Z,2))
// 110     CONTINUE
// 100  CONTINUE
//
// IMPORTANT:
// ---------
// Fortran CALCOM uses UC(X,Z,*) with *no* special Z-aliasing. It only
// looks at the first NZ rows of UC(3*NX/4+1, 3*NZ/2, 3). Our previous
// CUDA implementation incorrectly applied the STEP3-style Z-mapping
// (Z > NZ/2 -> Z+NZ/2), which caused perfect parity for Z<=NZ/2 and
// large errors for Z>NZ/2.
//
// This version uses the full grid d_uc_full but with a *direct*
// Z-index: z_full = Z-1 for all Z=1..NZ.
//
//   OM2(x,z) = i * ( gamma(z) * UC1(x,z) - alfa(x) * UC2(x,z) )
//
// with i = (0,1) so:
//   i * (a + i b) = (-b) + i a

#include "cuda_dns.h"
#include <cuda_runtime.h>
#include <cufft.h>
#include <math.h>

// cuda_dns.h is expected to provide:
//   typedef float        real;
//   typedef cufftComplex cplx;
//   struct DnsDeviceState { ... };

// ---------------------------------------------------------------------
// Index helpers
// ---------------------------------------------------------------------

// UC_full: [comp][z_full][kx] -> flattened index
__device__ inline size_t uc_full_idx(int x, int z_full, int c,
                                     int NK_full, int NZ_full)
{
    // c = 0,1,2; z_full = 0..NZ_full-1; x = 0..NK_full-1
    return ((size_t)c * (size_t)NZ_full + (size_t)z_full) * (size_t)NK_full + (size_t)x;
}

// OM2: [z][x] with x = 0..NX_half-1, z = 0..NZ-1
__device__ inline int om2_idx(int ix, int iz, int NX_half)
{
    return iz * NX_half + ix;
}

// ---------------------------------------------------------------------
// Kernel: spectral computation of OM2 from UC_full
// ---------------------------------------------------------------------
//
// For each (X,Z) with
//   X = 1..NX/2   -> ix = X-1
//   Z = 1..NZ     -> iz = Z-1
//
//   OM2(X,Z) = i * ( GAMMA(Z)*UC(X,Z,1) - ALFA(X)*UC(X,Z,2) )
//
// UC(X,Z,*) is taken from the full spectral grid UC_full with a
// *direct* Z-index:
//
//   z_full = iz    // 0..NZ-1
//
// i.e. we read UC(X,Z,*) from the first NZ rows of the 3/2 grid,
// exactly as Fortran CALCOM does.

__global__
void k_calcom_from_uc_full(const cplx *uc_full,
                           cplx       *om2,
                           const real *alfa,
                           const real *gamma,
                           int Nbase,     // NX = NZ = Nbase
                           int NX_full,
                           int NZ_full,
                           int NK_full)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x; // 0..NX/2-1
    int iz = blockIdx.y * blockDim.y + threadIdx.y; // 0..NZ-1

    int NX_half = Nbase / 2;
    int NZ      = Nbase;

    if (ix >= NX_half || iz >= NZ) return;

    // Safety: NZ_full should be >= NZ (we only use first NZ rows)
    // (No branch; assume this holds, as in Fortran)

    // Physical indices:
    //   X = ix+1, Z = iz+1
    // Wavenumbers:
    real ax = alfa[ix];   // ALFA(X)
    real gz = gamma[iz];  // GAMMA(Z)

    // Direct Z-index into full UC grid: UC(X,Z,*) := UC_full(ix, iz, *)
    int z_full = iz;  // 0..NZ-1, no aliasing

    // Fetch UC(X,Z,1:2) from full grid
    size_t idx_uc1 = uc_full_idx(ix, z_full, 0, NK_full, NZ_full); // UC(...,1)
    size_t idx_uc2 = uc_full_idx(ix, z_full, 1, NK_full, NZ_full); // UC(...,2)

    cplx u1 = uc_full[idx_uc1];
    cplx u2 = uc_full[idx_uc2];

    // diff = GAMMA(Z)*UC1 - ALFA(X)*UC2
    cplx diff;
    diff.x = gz * u1.x - ax * u2.x;
    diff.y = gz * u1.y - ax * u2.y;

    // Multiply by i = (0,1): i * (a + i b) = (-b) + i a
    cplx om;
    om.x = -diff.y;
    om.y =  diff.x;

    int idx_om = om2_idx(ix, iz, NX_half);
    om2[idx_om] = om;
}

// ---------------------------------------------------------------------
// Host entry point
// ---------------------------------------------------------------------
//
// Called from dnsCudaInit, matching the Fortran sequence:
//
//   CALL PAO   (UC,ALFA,GAMMA,...)
//   CALL INIT  (N,N,ALFA,GAMMA,...)
//   CALL CALCOM(N,N,ALFA,GAMMA,UC,OM2)
//
// We use the full spectral grid (d_uc_full) and write OM2(N/2,N)
// into d_om2, using the same [z][x] layout as the Fortran dump.
//

void dnsCudaCalcom(DnsDeviceState *S)
{
    if (!S) return;

    int NX      = S->Nbase;   // Fortran NX
    int NZ      = S->Nbase;   // Fortran NZ
    int NX_full = S->NX_full;
    int NZ_full = S->NZ_full;
    int NK_full = S->NK_full;

    int NX_half = NX / 2;

    dim3 block(16, 16);
    dim3 grid( (NX_half + block.x - 1) / block.x,
               (NZ      + block.y - 1) / block.y );

    k_calcom_from_uc_full<<<grid, block>>>(
        S->d_uc_full,   // FULL UC from PAO/INIT, Fortran layout
        S->d_om2,
        S->d_alfa,
        S->d_gamma,
        S->Nbase,
        NX_full,
        NZ_full,
        NK_full
    );
}
