// step2a_cuda.cu
// ---------------------------------------------------------------------
// Fortran-style STEP2A on the full 3/2 grid, then downmap UR_full → UR.
//
//   1) Dealias high-kx band on UC_full
//   2) Z-reshuffle the low-kz strip (as in PAO/STEP2A in visasub.f)
//   3) Inverse FFT UC_full → UR_full via vfft_full_inverse_uc_full_to_ur_full
//   4) Copy low N×N block of UR_full into compact UR (N x N)
//
// STEP2B and STEP3 are untouched for now.
// ---------------------------------------------------------------------

#include "cuda_dns.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>

// Dealias: zero high-kx modes on UC_full for comp=0,1
// Fortran logic (1-based):
//   DO I=1,2
//     DO Z=1,3*NZ/2
//       DO X=NX/2+1,3*NX/4+1
//         UC(X,Z,I) = 0
//       END DO
//     END DO
//   END DO
//
// Here Nbase = N, 0-based indices:
//   X = N/2 .. 3N/4
__global__
void k_step2a_full_zero_highkx(cplx *uc_full,
                               int Nbase,
                               int NZ_full,
                               int NK_full)
{
    int N = Nbase;

    int nx_start = N / 2;      // 0-based
    int nx_end   = 3 * N / 4;  // inclusive
    int nx_len   = nx_end - nx_start + 1;

    int tx = blockIdx.x * blockDim.x + threadIdx.x;  // along X band
    int tz = blockIdx.y * blockDim.y + threadIdx.y;  // along Z

    if (tz >= NZ_full || tx >= nx_len) return;

    int kx = nx_start + tx;
    if (kx >= NK_full) return;

    // Only velocity components 0,1 (u,w)
    for (int c = 0; c < 2; ++c) {
        size_t idx = UC_FULL_INDEX(kx, tz, c, NK_full, NZ_full);
        uc_full[idx].x = 0.0f;
        uc_full[idx].y = 0.0f;
    }
}

// Z-reshuffle on UC_full for low-kz strip
// Fortran (1-based):
//   DO I=1,2
//     DO Z=1,NZ/2
//       DO X=1,NX/2
//         UC(X,Z+NZ,I)   = UC(X,Z+NZ/2,I)
//         UC(X,Z+NZ/2,I) = 0
//       END DO
//     END DO
//   END DO
//
// Here Nbase = N, 0-based:
//
//   z_low = 0..N/2-1
//   z_mid = z_low + N/2
//   z_top = z_low + N
//
__global__
void k_step2a_full_reshuffle_z(cplx *uc_full,
                               int Nbase,
                               int NZ_full,
                               int NK_full)
{
    int N = Nbase;

    int tx = blockIdx.x * blockDim.x + threadIdx.x;  // X index
    int tz = blockIdx.y * blockDim.y + threadIdx.y;  // Z-low index

    if (tx >= N/2 || tz >= N/2) return;
    if (tx >= NK_full) return;

    //int z_low = tz;
    int z_mid = tz + N/2;
    int z_top = tz + N;

    if (z_top >= NZ_full) return;

    for (int c = 0; c < 2; ++c) {
        size_t idx_mid = UC_FULL_INDEX(tx, z_mid, c, NK_full, NZ_full);
        size_t idx_top = UC_FULL_INDEX(tx, z_top, c, NK_full, NZ_full);

        cplx v = uc_full[idx_mid];
        uc_full[idx_top] = v;

        // zero mid band
        uc_full[idx_mid].x = 0.0f;
        uc_full[idx_mid].y = 0.0f;
    }
}

__global__
void k_step2a_full_prepare(cplx *uc_full,
                           int Nbase,
                           int NZ_full,
                           int NK_full)
{
    int N = Nbase;

    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int tz = blockIdx.y * blockDim.y + threadIdx.y;

    int nx_start = N / 2;
    int nx_end   = 3 * N / 4;
    int nx_len   = nx_end - nx_start + 1;

    if (tz < NZ_full && tx < nx_len) {
        int kx = nx_start + tx;
        if (kx < NK_full) {
            for (int c = 0; c < 2; ++c) {
                size_t idx = UC_FULL_INDEX(kx, tz, c, NK_full, NZ_full);
                uc_full[idx].x = 0.0f;
                uc_full[idx].y = 0.0f;
            }
        }
    }

    if (tx < N / 2 && tz < N / 2 && tx < NK_full) {
        int z_mid = tz + N / 2;
        int z_top = tz + N;

        if (z_top < NZ_full) {
            for (int c = 0; c < 2; ++c) {
                size_t idx_mid = UC_FULL_INDEX(tx, z_mid, c, NK_full, NZ_full);
                size_t idx_top = UC_FULL_INDEX(tx, z_top, c, NK_full, NZ_full);

                cplx v = uc_full[idx_mid];
                uc_full[idx_top] = v;

                uc_full[idx_mid].x = 0.0f;
                uc_full[idx_mid].y = 0.0f;
            }
        }
    }
}

// ---------------------------------------------------------------------
// Fortran-style STEP2A on full 3/2 grid (debug + physics)
// ---------------------------------------------------------------------
void dnsCudaStep2A_full_debug(DnsDeviceState *S)
{
    if (!S) return;

    int N       = S->Nbase;
    int NZ_full = S->NZ_full;
    int NK_full = S->NK_full;

    // 1) Dealias high-kx band and z-reshuffle low-kz strip.
    {
        dim3 block(64, 4);
        int nx_start = N / 2;
        int nx_end   = 3 * N / 4;
        int nx_len   = nx_end - nx_start + 1;
        int work_x   = std::max(nx_len, N / 2);

        dim3 grid((work_x  + block.x - 1) / block.x,
                  (NZ_full + block.y - 1) / block.y);

        DNS_PHASE_TIME(DNS_PHASE_STEP2A_PREPARE, {
            k_step2a_full_prepare<<<grid, block, 0, S->stream>>>(
                S->d_uc_full, N, NZ_full, NK_full
            );
        });
    }

    // 3) Full inverse FFT: UC_full → UR_full (3/2 grid)
    DNS_PHASE_TIME(DNS_PHASE_STEP2A_FFT, {
        vfft_full_inverse_uc_full_to_ur_full(S);
    });
}

// ---------------------------------------------------------------------
// dnsCudaStep2A — used by the main time-stepper
// ---------------------------------------------------------------------
// Uses the full-grid STEP2A path, then downmaps UR_full → UR so that
// compact-grid diagnostics and CFL see a proper physical velocity field.
// ---------------------------------------------------------------------
void dnsCudaStep2A(DnsDeviceState *S)
{
    if (!S) return;

    // Full 3/2-grid STEP2A (dealias + reshuffle + inverse)
    // Populates d_ur_full; compact d_ur is not needed in the hot loop.
    dnsCudaStep2A_full_debug(S);
}
