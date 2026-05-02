// om2phys_cuda.cu
#include "cuda_dns.h"
#include <cuda_runtime.h>
#include <cufft.h>
#include <cstdio>

// We assume UC_FULL_INDEX(kx,z,comp,nk,nz) is defined in cuda_dns.h,
// and that:
//   S->d_om2     : [z][x], size NZ * (NX/2)
//   S->d_uc_full : [comp][z][kx], comp=0,1,2, size 3 * NZ_full * NK_full
//   S->d_ur_full : [comp][z][x],  comp=0,1,2, size 3 * NZ_full * NX_full

// ---------------------------------------------------------------------
// Kernel 1: clear UC_full(:,:,3) on full 3/2 grid
// ---------------------------------------------------------------------
__global__
void k_clear_uc3_full(cplx *uc_full, int NK_full, int NZ_full)
{
    int kx = blockIdx.x * blockDim.x + threadIdx.x;
    int z  = blockIdx.y * blockDim.y + threadIdx.y;
    if (kx >= NK_full || z >= NZ_full) return;

    size_t idx = UC_FULL_INDEX(kx, z, 2, NK_full, NZ_full); // comp=2 ↔ 3rd
    uc_full[idx].x = 0.0f;
    uc_full[idx].y = 0.0f;
}

// ---------------------------------------------------------------------
// Kernel 2: copy compact OM2(kx,z) → UC_full(kx,z,3) for kx < NX/2, z < N
//   Fortran: UC(X,Z,3) = OM2(X,Z),  X=1..NX/2, Z=1..NZ
// ---------------------------------------------------------------------
__global__
void k_copy_om2_to_uc3(const cplx *om2,
                       cplx       *uc_full,
                       int NX, int NZ,
                       int NK_full, int NZ_full)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x; // 0..NX/2-1
    int iz = blockIdx.y * blockDim.y + threadIdx.y; // 0..NZ-1

    int NX2 = NX / 2;
    if (ix >= NX2 || iz >= NZ) return;

    int idx_om  = iz * NX2 + ix;                           // OM2(X,Z)
    size_t idx_uc3 = UC_FULL_INDEX(ix, iz, 2, NK_full, NZ_full);

    uc_full[idx_uc3] = om2[idx_om];
}

// ---------------------------------------------------------------------
// Kernel 3: Z-reshuffle, Fortran:
//
//   DO Z=1,NZ/2
//     DO X=1,NX/2
//       UC(X,Z+NZ,3)   = UC(X,Z+NZ/2,3)
//       UC(X,Z+NZ/2,3) = 0
//   ...
//
// 0-based:
//   z in [0,NZ/2-1], x in [0,NX/2-1]
//   z_src = z + NZ/2
//   z_dst = z + NZ
// ---------------------------------------------------------------------
__global__
void k_reshuffle_om2_z(cplx *uc_full,
                       int NX, int NZ,
                       int NK_full, int NZ_full)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;

    int NX2 = NX / 2;
    int NZ2 = NZ / 2;

    if (ix >= NX2 || iz >= NZ2) return;

    int z_src = iz + NZ2;   // Z+NZ/2
    int z_dst = iz + NZ;    // Z+NZ

    size_t idx_src = UC_FULL_INDEX(ix, z_src, 2, NK_full, NZ_full);
    size_t idx_dst = UC_FULL_INDEX(ix, z_dst, 2, NK_full, NZ_full);

    cplx v = uc_full[idx_src];
    uc_full[idx_dst] = v;
    uc_full[idx_src].x = 0.0f;
    uc_full[idx_src].y = 0.0f;
}

// ---------------------------------------------------------------------
// Host: OM2 → phys UR_full(:,:,3), Fortran OM2PHYS equivalent
// ---------------------------------------------------------------------
void dnsCudaOm2Phys(DnsDeviceState *S)
{
    if (!S) return;

    const int N        = S->Nbase;    // NX = NZ = N
    const int NX       = N;
    const int NZ       = N;
    const int NX_full  = S->NX_full;  // 3N/2
    const int NZ_full  = S->NZ_full;  // 3N/2
    const int NK_full  = S->NK_full;  // 3N/4+1

    // 1) Clear UC_full(:,:,3)
    {
        dim3 block(16, 16);
        dim3 grid((NK_full + block.x - 1) / block.x,
                  (NZ_full + block.y - 1) / block.y);
        k_clear_uc3_full<<<grid, block>>>(S->d_uc_full, NK_full, NZ_full);
    }

    // 2) Copy compact OM2 → low band of UC_full(:,:,3)
    {
        dim3 block(16, 16);
        dim3 grid((NX/2 + block.x - 1) / block.x,
                  (NZ    + block.y - 1) / block.y);
        k_copy_om2_to_uc3<<<grid, block>>>(
            S->d_om2,
            S->d_uc_full,
            NX, NZ,
            NK_full, NZ_full
        );
    }

    // 3) Z-reshuffle like the Fortran loop 210/220
    {
        dim3 block(16, 16);
        dim3 grid((NX/2      + block.x - 1) / block.x,
                  (NZ/2      + block.y - 1) / block.y);
        k_reshuffle_om2_z<<<grid, block>>>(
            S->d_uc_full,
            NX, NZ,
            NK_full, NZ_full
        );
    }

    // 4) Inverse FFT: UC_full(:,:,3) → UR_full(:,:,3)

    const size_t plane_cplx = (size_t)NK_full * NZ_full;
    const size_t plane_real = (size_t)NX_full * NZ_full;

    cplx *uc_comp = S->d_uc_full + plane_cplx * 2;  // comp=2
    real *ur_comp = S->d_ur_full + plane_real * 2;  // comp=2

    cufftResult rx = cufftExecC2R(
        S->plan_full_c2r_2d_one,
        reinterpret_cast<cufftComplex*>(uc_comp),
        reinterpret_cast<cufftReal*  >(ur_comp)
    );
    if (rx != CUFFT_SUCCESS) {
        std::printf("dnsCudaOm2Phys: cufftExecC2R failed, code=%d\n", rx);
        return;
    }

}

// ---------------------------------------------------------------------
// Kernel: build streamfunction spectrum from OM2
//   Fortran:
//      K2 = ALFA(X)**2 + GAMMA(Z)**2
//      UC(X,Z,4) = OM2(X,Z) / (K2 + 1E-30)
//      UC(1,1,4) = 0
//
//   Here we write it into UC_full(:,:,comp=2).
// ---------------------------------------------------------------------
__global__
void k_streamfunc_from_om2(const cplx *om2,
                           cplx       *uc_full,
                           const real *alfa,
                           const real *gamma,
                           int NX, int NZ,
                           int NK_full, int NZ_full)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x; // 0..NX/2-1
    int iz = blockIdx.y * blockDim.y + threadIdx.y; // 0..NZ-1

    int NX2 = NX / 2;
    if (ix >= NX2 || iz >= NZ) return;

    // Get wave numbers
    real ax = alfa[ix];   // ALFA(X)
    real gz = gamma[iz];  // GAMMA(Z)

    real k2 = ax*ax + gz*gz + (real)1.0e-30f;

    int idx_om  = iz * NX2 + ix;                           // OM2(X,Z)
    size_t idx_uc4 = UC_FULL_INDEX(ix, iz, 2, NK_full, NZ_full); // comp=2 ~ axis 4

    cplx v = om2[idx_om];

    v.x /= k2;
    v.y /= k2;

    // UC(1,1,4) = 0
    if (ix == 0 && iz == 0) {
        v.x = 0.0f;
        v.y = 0.0f;
    }

    uc_full[idx_uc4] = v;
}

// ---------------------------------------------------------------------
// STREAMFUNC on GPU: build streamfunction ψ(x,z) from OM2.
// Result: UR_full(:,:,2) holds ψ in physical space (3rd component).
// Mirrors Fortran STREAMFUNC (axis=4) using comp=2 here.
// ---------------------------------------------------------------------
void dnsCudaStreamFunc(DnsDeviceState *S)
{
    if (!S) return;

    const int N       = S->Nbase;    // NX = NZ = N
    const int NX      = N;
    const int NZ      = N;
    const int NX_full = S->NX_full;  // 3N/2
    const int NZ_full = S->NZ_full;  // 3N/2
    const int NK_full = S->NK_full;  // 3N/4+1

    // 1) Clear UC_full(:,:,stream) — reuse OM2PHYS helper
    {
        dim3 block(16, 16);
        dim3 grid((NK_full + block.x - 1) / block.x,
                  (NZ_full + block.y - 1) / block.y);
        k_clear_uc3_full<<<grid, block>>>(S->d_uc_full, NK_full, NZ_full);
    }

    // 2) Fill low band from OM2 / k^2 (Fortran DO 10/20)
    {
        dim3 block(16, 16);
        dim3 grid((NX/2 + block.x - 1) / block.x,
                  (NZ    + block.y - 1) / block.y);

        k_streamfunc_from_om2<<<grid, block>>>(
            S->d_om2,
            S->d_uc_full,
            S->d_alfa,
            S->d_gamma,
            NX, NZ,
            NK_full, NZ_full
        );
    }

    // 3) Z-reshuffle like DO 210/220 in Fortran
    {
        dim3 block(16, 16);
        dim3 grid((NX/2      + block.x - 1) / block.x,
                  (NZ/2      + block.y - 1) / block.y);
        k_reshuffle_om2_z<<<grid, block>>>(
            S->d_uc_full,
            NX, NZ,
            NK_full, NZ_full
        );
    }

    cudaDeviceSynchronize();

    // 4) Inverse FFT: UC_full(:,:,stream) → UR_full(:,:,stream)

    const size_t plane_cplx = (size_t)NK_full * NZ_full;
    const size_t plane_real = (size_t)NX_full * NZ_full;

    cplx *uc_comp = S->d_uc_full + plane_cplx * 2;  // comp=2
    real *ur_comp = S->d_ur_full + plane_real * 2;  // comp=2

    cufftResult rx = cufftExecC2R(
        S->plan_full_c2r_2d_one,
        reinterpret_cast<cufftComplex*>(uc_comp),
        reinterpret_cast<cufftReal*  >(ur_comp)
    );
    if (rx != CUFFT_SUCCESS) {
        std::printf("dnsCudaStreamFunc: cufftExecC2R failed, code=%d\n", rx);
        return;
    }

    cudaDeviceSynchronize();
}

// Kernel: K(x,z) = sqrt( U^2 + W^2 ) on full 3/2 grid
// UR_full layout: [comp][z][x], comp=0 -> U, comp=1 -> W, comp=2 -> K
__global__
void k_kinetic_from_ur_full(const real* ur_full,
                            real*       ur_full_out,
                            int NX_full,
                            int NZ_full)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int z = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= NX_full || z >= NZ_full) return;

    const size_t plane = (size_t)NX_full * (size_t)NZ_full;

    size_t idx = (size_t)z * (size_t)NX_full + (size_t)x;

    // comp 0: U, comp 1: W
    real u = ur_full[0 * plane + idx];
    real w = ur_full[1 * plane + idx];

    real U2 = u * u;
    real W2 = w * w;
    real K  = sqrtf(U2 + W2);

    // write into comp 2 plane
    ur_full_out[2 * plane + idx] = K;
}

// Host wrapper: compute kinetic magnitude into UR_full(:,:,2)
void dnsCudaKinetic(DnsDeviceState* S)
{
    if (!S) return;

    const int NX_full = S->NX_full;  // 3N/2
    const int NZ_full = S->NZ_full;  // 3N/2

    real* d_ur_full = S->d_ur_full;

    dim3 block(16, 16);
    dim3 grid((NX_full + block.x - 1) / block.x,
              (NZ_full + block.y - 1) / block.y);

    k_kinetic_from_ur_full<<<grid, block>>>(d_ur_full, d_ur_full,
                                            NX_full, NZ_full);
}

// We assume in cuda_dns.h:
//   #define UC_FULL_INDEX(kx,z,comp,nk,nz)  ((comp)*(nz)*(nk) + (z)*(nk) + (kx))

// ---------------------------------------------------------------------
// Kernel: velocity potential from the current full-grid velocity spectrum.
//
// Mathematical definition:
//   div(u,w)_k = i * (kx*u_k + kz*w_k)
//   -|k|^2 * phi_k = div(u,w)_k
//   phi_k = -i * (kx*u_k + kz*w_k) / |k|^2
//
// The input is generated immediately before this kernel by a full-grid cuFFT
// forward transform of UR_full, so z uses cuFFT's natural ordering:
//   kz = z                 for z <= NZ_full/2
//   kz = z - NZ_full       otherwise
// ---------------------------------------------------------------------
__global__
void k_phi_from_uc_full(const cplx *uc_full_vel,
                        cplx       *uc_full,
                        int NK_full, int NZ_full)
{
    int kx = blockIdx.x * blockDim.x + threadIdx.x; // 0..NK_full-1
    int zf = blockIdx.y * blockDim.y + threadIdx.y; // 0..NZ_full-1

    if (kx >= NK_full || zf >= NZ_full) return;

    real ax = (real)kx;
    int kz_i = (zf <= NZ_full / 2) ? zf : zf - NZ_full;
    real gz = (real)kz_i;

    size_t idx_u1 = UC_FULL_INDEX(kx, zf, 0, NK_full, NZ_full);
    size_t idx_u2 = UC_FULL_INDEX(kx, zf, 1, NK_full, NZ_full);
    size_t idx_phi = UC_FULL_INDEX(kx, zf, 2, NK_full, NZ_full);

    cplx u1 = uc_full_vel[idx_u1];
    cplx u2 = uc_full_vel[idx_u2];

    real k2 = ax * ax + gz * gz;
    if (k2 <= 0.0f) {
        uc_full[idx_phi].x = 0.0f;
        uc_full[idx_phi].y = 0.0f;
        return;
    }

    real inv_k2 = 1.0f / k2;
    cplx div;
    div.x = (ax * u1.x + gz * u2.x) * inv_k2;
    div.y = (ax * u1.y + gz * u2.y) * inv_k2;

    // -i * (a + i b) = b - i a
    uc_full[idx_phi].x = div.y;
    uc_full[idx_phi].y = -div.x;
}

// Host wrapper: build φ(x,z) into UR_full(:,:,2) from UC and ALFA/GAMMA
void dnsCudaPhiPhys(DnsDeviceState *S)
{
    if (!S) return;

    const int NX_full = S->NX_full;  // 3N/2
    const int NZ_full = S->NZ_full;  // 3N/2
    const int NK_full = S->NK_full;  // full (3N/4+1)

    // 1) Start from the actual current physical velocity field.  This avoids
    // compact/full-layout assumptions and computes PHI from first principles.
    vfft_full_forward_ur_full_to_uc_full(S);
    cudaDeviceSynchronize();

    // 2) Clear UC_full(:,:,φ) — reuse OM2/streamfunc helper
    {
        dim3 block(16, 16);
        dim3 grid((NK_full + block.x - 1) / block.x,
                  (NZ_full + block.y - 1) / block.y);
        k_clear_uc3_full<<<grid, block>>>(S->d_uc_full, NK_full, NZ_full);
    }

    // 3) Solve Poisson in spectral space:
    //      phi_k = -i * (kx*u_k + kz*w_k) / (kx^2 + kz^2)
    {
        dim3 block(16, 16);
        dim3 grid((NK_full + block.x - 1) / block.x,
                  (NZ_full + block.y - 1) / block.y);

        k_phi_from_uc_full<<<grid, block>>>(
            S->d_uc_full,   // current full UC velocity (comp=0,1)
            S->d_uc_full,   // full UC_full, we use comp=2
            NK_full, NZ_full
        );
    }

    cudaDeviceSynchronize();

    // 4) Inverse FFT: UC_full(:,:,phi) → UR_full(:,:,phi)

    const size_t plane_cplx = (size_t)NK_full * NZ_full;
    const size_t plane_real = (size_t)NX_full * NZ_full;

    cplx *uc_comp = S->d_uc_full + plane_cplx * 2;  // comp=2
    real *ur_comp = S->d_ur_full + plane_real * 2;  // comp=2

    cufftResult rx = cufftExecC2R(
        S->plan_full_c2r_2d_one,
        reinterpret_cast<cufftComplex*>(uc_comp),
        reinterpret_cast<cufftReal*  >(ur_comp)
    );
    if (rx != CUFFT_SUCCESS) {
        std::printf("dnsCudaPhiPhys: cufftExecC2R failed, code=%d\n", rx);
        return;
    }

    cudaDeviceSynchronize();
}
