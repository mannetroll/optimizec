//
// dns_cuda.cu — Option A (Fortran-physics focused driver)
// -------------------------------------------------------
// Orchestrates:
//   • dnsCudaCreate / dnsCudaDestroy
//   • dnsCudaPaoHostInit  (PAO, full Fortran physics)
//   • dnsCudaStep2A / 2B / 3
//   • CFL-based next_dt_gpu
//   • Simple debug printing
//

#include "cuda_dns.h"
#include <cuda_runtime.h>
#include <cufft.h>

#include <cstdio>
#include <cmath>
#include <vector>
#include <cstring>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ======================================================================
// Small helpers to get max|.| on GPU
// ======================================================================

__global__
void k_debug_stats_real(const float *a, int n, float *out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float v = 0.0f;

    if (idx < n) {
        float x = a[idx];
        v = fabsf(x);
    }

    // warp-level max
    for (int offset = 16; offset > 0; offset >>= 1)
        v = fmaxf(v, __shfl_down_sync(0xffffffff, v, offset));

    if ((threadIdx.x & 31) == 0)
        out[blockIdx.x] = v;
}

static float gpu_maxabs_real(const float *d, int n)
{
    int B = 256;
    int G = (n + B - 1) / B;
    if (G < 1) G = 1;

    float *d_out = nullptr;
    cudaMalloc(&d_out, G * sizeof(float));

    k_debug_stats_real<<<G,B>>>(d, n, d_out);

    std::vector<float> tmp(G, 0.0f);
    cudaMemcpy(tmp.data(), d_out, G * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_out);

    float m = 0.0f;
    for (float v : tmp) m = fmaxf(m, v);
    return m;
}

static float gpu_maxabs_cplx(const cplx *d, int n_cplx)
{
    // treat as 2*n real values
    return gpu_maxabs_real(reinterpret_cast<const float*>(d), 2*n_cplx);
}

// ======================================================================
// Life-cycle: Create / Destroy
// ======================================================================
bool dnsCudaCreate(DnsDeviceState *S, int N, real Re, real K0)
{
    if (!S) return false;
    std::memset(S, 0, sizeof(*S));

    // ------------------------------------------------------------------
    // Geometry / parameters
    // ------------------------------------------------------------------
    S->Nbase = N;           // logical base N (128)
    S->N     = N;           // keep == Nbase for now

    // Current working grid (minimal model — unchanged behaviour)
    S->NX    = N;
    S->NZ    = N;
    S->NK    = 3*S->Nbase/4 + 1;   // already adopted

    // Future full 3/2 grid (Fortran-style). We just store these now.
    S->NX_full = 3*S->Nbase/2;      // 3N/2 (e.g. 192)
    S->NZ_full = 3*S->Nbase/2;      // 3N/2
    S->NK_full = 3*S->Nbase/4 + 1;  // same spectral width as NK

    S->Re   = Re;
    S->K0   = K0;
    S->visc = 0.0f;

    S->t    = 0.0f;
    S->dt   = 0.0f;
    S->cn   = 1.0f;
    S->cnm1 = 0.0f;     // <-- must start at 0 to match Fortran
    S->cflnum = 0.75f;  // CFLNUM
    S->it   = 0;
    S->ifn  = 1;        // IFN = 1

    // ------------------------------------------------------------------
    // Allocate device arrays: current working grid (N x N)
    // ------------------------------------------------------------------
    size_t n_om = (size_t)S->NZ * (S->NX/2);      // vorticity

    auto alloc = [](void **ptr, size_t bytes, const char *name) -> bool {
        cudaError_t err = cudaMalloc(ptr, bytes);
        if (err != cudaSuccess) {
            std::fprintf(stderr, "cudaMalloc(%s, %.3f GiB) failed: %s\n",
                         name,
                         bytes / (1024.0 * 1024.0 * 1024.0),
                         cudaGetErrorString(err));
            return false;
        }
        return true;
    };
    auto checkCufft = [](cufftResult r, const char *name) -> bool {
        if (r != CUFFT_SUCCESS) {
            std::fprintf(stderr, "%s failed: CUFFT code %d\n", name, (int)r);
            return false;
        }
        return true;
    };
    auto fail = [&]() -> bool {
        dnsCudaDestroy(S);
        return false;
    };

    // The current mem_cuda path runs on d_ur_full/d_uc_full.  The compact
    // d_ur/d_uc buffers are legacy debug/snapshot storage; at N=20480 they
    // cost about 12.5 GiB, so do not allocate them here.
    S->d_ur = nullptr;
    S->d_uc = nullptr;

    if (!alloc(reinterpret_cast<void**>(&S->d_om2),   n_om * sizeof(cplx), "d_om2")) return fail();
    if (!alloc(reinterpret_cast<void**>(&S->d_fnm1),  n_om * sizeof(cplx), "d_fnm1")) return fail();

    if (!alloc(reinterpret_cast<void**>(&S->d_alfa),  (S->NX/2) * sizeof(real), "d_alfa")) return fail();
    if (!alloc(reinterpret_cast<void**>(&S->d_gamma),  S->NZ    * sizeof(real), "d_gamma")) return fail();

    // ------------------------------------------------------------------
    // Allocate device arrays: future full 3/2 grid (NOT USED YET)
    // ------------------------------------------------------------------
    size_t n_ur_full = (size_t)S->NX_full * S->NZ_full * 3;
    size_t n_uc_full = (size_t)S->NZ_full * S->NK_full * 3;

    if (!alloc(reinterpret_cast<void**>(&S->d_ur_full), n_ur_full * sizeof(real), "d_ur_full")) return fail();
    if (!alloc(reinterpret_cast<void**>(&S->d_uc_full), n_uc_full * sizeof(cplx), "d_uc_full")) return fail();

    // CFLM GPU reduction scratch: one float per block (block size 256)
    int cflm_total = S->NX_full * S->NZ_full;
    S->cflm_num_blocks = (cflm_total + 255) / 256;
    if (!alloc(reinterpret_cast<void**>(&S->d_cflm_scratch),
               (size_t)S->cflm_num_blocks * sizeof(real),
               "d_cflm_scratch")) return fail();

    const size_t sigma_total = (size_t)S->NX_full * (size_t)S->NZ_full;
    const size_t sigma_min_blocks = (sigma_total + 255u) / 256u;
    S->sigma_num_blocks = (int)std::min<size_t>(1024u,
                                                std::max<size_t>(1u, sigma_min_blocks));
    if (!alloc(reinterpret_cast<void**>(&S->d_sigma_minmax),
               (size_t)S->sigma_num_blocks * 2u * sizeof(real),
               "d_sigma_minmax")) return fail();
    if (!alloc(reinterpret_cast<void**>(&S->d_sigma_sums),
               (size_t)S->sigma_num_blocks * 2u * sizeof(unsigned long long),
               "d_sigma_sums")) return fail();

    S->step3_divxz = 1.0f / (float(S->NX_full) * float(S->NZ_full));

    // ------------------------------------------------------------------
    // FULL 3/2 GRID PLANS (UC_full ↔ UR_full)
    // ------------------------------------------------------------------
    // Only the 2D batched plans are used by vfft_cuda.cu.  The older 1D
    // plans consumed large work areas and made N=20480 fail on 80 GB cards.

    // ------------------------------------------------------------------
    // 2D batched FFT plans: all 3 components in ONE cufftExec call.
    // UR_full layout [comp][z][x]  →  UC_full layout [comp][z][kx]
    //   n  = (NZ_full, NX_full) — logical 2D transform size
    //   batch = 3 (components)
    //   idist/odist = per-component plane stride
    // ------------------------------------------------------------------
    int n_2d[2] = { S->NZ_full, S->NX_full };
    int idist_r2c = S->NZ_full * S->NX_full;
    int odist_r2c = S->NZ_full * S->NK_full;
    int n_x[1] = { S->NX_full };
    int inembed_x[1] = { S->NX_full };
    int onembed_x[1] = { S->NK_full };
    int n_z[1] = { S->NZ_full };
    int embed_z[1] = { S->NZ_full };

    S->owns_plans = true;

    if (!checkCufft(cufftCreate(&S->plan_full_r2c_x),
                    "cufftCreate(plan_full_r2c_x)")) return fail();
    if (!checkCufft(cufftSetAutoAllocation(S->plan_full_r2c_x, 0),
                    "cufftSetAutoAllocation(plan_full_r2c_x)")) return fail();
    if (!checkCufft(cufftCreate(&S->plan_full_c2c_z),
                    "cufftCreate(plan_full_c2c_z)")) return fail();
    if (!checkCufft(cufftSetAutoAllocation(S->plan_full_c2c_z, 0),
                    "cufftSetAutoAllocation(plan_full_c2c_z)")) return fail();
    if (!checkCufft(cufftCreate(&S->plan_full_r2c_2d),
                    "cufftCreate(plan_full_r2c_2d)")) return fail();
    if (!checkCufft(cufftSetAutoAllocation(S->plan_full_r2c_2d, 0),
                    "cufftSetAutoAllocation(plan_full_r2c_2d)")) return fail();

    int inembed_c2r[2] = { S->NZ_full, S->NK_full };
    int onembed_c2r[2] = { S->NZ_full, S->NX_full };
    int idist_c2r = S->NZ_full * S->NK_full;
    int odist_c2r = S->NZ_full * S->NX_full;

    if (!checkCufft(cufftCreate(&S->plan_full_c2r_2d),
                    "cufftCreate(plan_full_c2r_2d)")) return fail();
    if (!checkCufft(cufftSetAutoAllocation(S->plan_full_c2r_2d, 0),
                    "cufftSetAutoAllocation(plan_full_c2r_2d)")) return fail();
    if (!checkCufft(cufftCreate(&S->plan_full_c2r_2d_one),
                    "cufftCreate(plan_full_c2r_2d_one)")) return fail();
    if (!checkCufft(cufftSetAutoAllocation(S->plan_full_c2r_2d_one, 0),
                    "cufftSetAutoAllocation(plan_full_c2r_2d_one)")) return fail();

    size_t work_r2c = 0;
    size_t work_r2c_x = 0;
    size_t work_c2c_z = 0;
    size_t work_c2r = 0;
    size_t work_c2r_one = 0;
    if (!checkCufft(cufftMakePlanMany(S->plan_full_r2c_x,
                                      1, n_x,
                                      inembed_x, 1, S->NX_full,
                                      onembed_x, 1, S->NK_full,
                                      CUFFT_R2C, 3 * S->NZ_full,
                                      &work_r2c_x),
                    "cufftMakePlanMany(plan_full_r2c_x)")) return fail();
    if (!checkCufft(cufftMakePlanMany(S->plan_full_c2c_z,
                                      1, n_z,
                                      embed_z, S->NK_full, 1,
                                      embed_z, S->NK_full, 1,
                                      CUFFT_C2C, S->NX / 2,
                                      &work_c2c_z),
                    "cufftMakePlanMany(plan_full_c2c_z)")) return fail();
    if (!checkCufft(cufftMakePlanMany(S->plan_full_r2c_2d,
                                      2, n_2d,
                                      nullptr, 1, idist_r2c,
                                      nullptr, 1, odist_r2c,
                                      CUFFT_R2C, 3,
                                      &work_r2c),
                    "cufftMakePlanMany(plan_full_r2c_2d)")) return fail();

    // Inverse only needs comps 0,1 (u,w); comp 2 is overwritten by STEP2B.
    if (!checkCufft(cufftMakePlanMany(S->plan_full_c2r_2d,
                                      2, n_2d,
                                      inembed_c2r, 1, idist_c2r,
                                      onembed_c2r, 1, odist_c2r,
                                      CUFFT_C2R, 2,
                                      &work_c2r),
                    "cufftMakePlanMany(plan_full_c2r_2d)")) return fail();
    if (!checkCufft(cufftMakePlanMany(S->plan_full_c2r_2d_one,
                                      2, n_2d,
                                      inembed_c2r, 1, idist_c2r,
                                      onembed_c2r, 1, odist_c2r,
                                      CUFFT_C2R, 1,
                                      &work_c2r_one),
                    "cufftMakePlanMany(plan_full_c2r_2d_one)")) return fail();

    S->fft_work_size = std::max(std::max(work_r2c, work_r2c_x),
                                std::max(work_c2c_z, std::max(work_c2r, work_c2r_one)));
    std::printf(" cuFFT work: R2C=%.2f MiB R2Cx=%.2f MiB C2Cz=%.2f MiB C2R2=%.2f MiB C2R1=%.2f MiB shared=%.2f MiB\n",
                work_r2c / (1024.0 * 1024.0),
                work_r2c_x / (1024.0 * 1024.0),
                work_c2c_z / (1024.0 * 1024.0),
                work_c2r / (1024.0 * 1024.0),
                work_c2r_one / (1024.0 * 1024.0),
                S->fft_work_size / (1024.0 * 1024.0));
    if (S->fft_work_size > 0) {
        if (!alloc(&S->d_fft_work, S->fft_work_size, "d_fft_work")) return fail();
        if (!checkCufft(cufftSetWorkArea(S->plan_full_r2c_x, S->d_fft_work),
                        "cufftSetWorkArea(plan_full_r2c_x)")) return fail();
        if (!checkCufft(cufftSetWorkArea(S->plan_full_c2c_z, S->d_fft_work),
                        "cufftSetWorkArea(plan_full_c2c_z)")) return fail();
        if (!checkCufft(cufftSetWorkArea(S->plan_full_r2c_2d, S->d_fft_work),
                        "cufftSetWorkArea(plan_full_r2c_2d)")) return fail();
        if (!checkCufft(cufftSetWorkArea(S->plan_full_c2r_2d, S->d_fft_work),
                        "cufftSetWorkArea(plan_full_c2r_2d)")) return fail();
        if (!checkCufft(cufftSetWorkArea(S->plan_full_c2r_2d_one, S->d_fft_work),
                        "cufftSetWorkArea(plan_full_c2r_2d_one)")) return fail();
    }


    return true;
}

void dnsCudaDestroy(DnsDeviceState *S)
{
    if (!S) return;

    // Working grid
    cudaFree(S->d_ur);
    cudaFree(S->d_uc);
    cudaFree(S->d_om2);
    cudaFree(S->d_fnm1);
    cudaFree(S->d_alfa);
    cudaFree(S->d_gamma);

    // Full 3/2 grid (not used yet, but allocated)
    cudaFree(S->d_ur_full);
    cudaFree(S->d_uc_full);
    cudaFree(S->d_fft_work);

    // CFLM reduction scratch
    cudaFree(S->d_cflm_scratch);
    cudaFree(S->d_sigma_minmax);
    cudaFree(S->d_sigma_sums);

    if (S->owns_plans) {
        if (S->plan_full_c2r_x) cufftDestroy(S->plan_full_c2r_x);
        if (S->plan_full_c2c_z) cufftDestroy(S->plan_full_c2c_z);
        if (S->plan_full_r2c_x) cufftDestroy(S->plan_full_r2c_x);
        if (S->plan_full_r2c_2d) cufftDestroy(S->plan_full_r2c_2d);
        if (S->plan_full_c2r_2d) cufftDestroy(S->plan_full_c2r_2d);
        if (S->plan_full_c2r_2d_one) cufftDestroy(S->plan_full_c2r_2d_one);
    }
}

// ======================================================================
// CFL-based dt, mirroring Fortran NEXTDT on the 3/2 grid
// ======================================================================

__global__
void k_cfl_scan(const float *ur, int Ncells,
                float inv_dx, float inv_dz,
                float visc, float *out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float m = 0.0f;

    if (idx < Ncells) {
        // UR layout: [z][x][comp], comp 0=u, 1=w, 2=extra
        float u = fabsf(ur[3*idx + 0]);
        float w = fabsf(ur[3*idx + 1]);

        float adv = fmaxf(u*inv_dx, w*inv_dz);
        m = adv;
    }

    // warp max
    for (int s=16; s>0; s>>=1)
        m = fmaxf(m, __shfl_down_sync(0xffffffff, m, s));

    if ((threadIdx.x & 31) == 0)
        out[blockIdx.x] = m;
}

__global__
void k_cfl_scan_full(const float *ur, int Ncells,
                     float inv_dx, float inv_dz,
                     float *out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float m = 0.0f;

    if (idx < Ncells) {
        // UR layout: [z][x][comp], comp 0=u, 1=w, 2=extra
        float u = fabsf(ur[3*idx + 0]);
        float w = fabsf(ur[3*idx + 1]);

        // Fortran: CFLM = max( |u|/dx + |w|/dz )
        float adv = u * inv_dx + w * inv_dz;
        m = adv;
    }

    // warp max
    for (int s = 16; s > 0; s >>= 1)
        m = fmaxf(m, __shfl_down_sync(0xffffffff, m, s));

    if ((threadIdx.x & 31) == 0)
        out[blockIdx.x] = m;
}

// ======================================================================
// GPU reduction for CFLM on UR_full (layout [comp][z][x])
// Computes max over all plane points of |u|*inv_dx + |w|*inv_dz.
// ======================================================================

__global__
void k_cflm_reduce_ur_full(const float *ur_full, size_t plane,
                           float dx, float dz,
                           float *out_blocks)
{
    __shared__ float s_max[256];

    int tid = threadIdx.x;
    size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)tid;

    float m = 0.0f;
    if (idx < plane) {
        float u = fabsf(ur_full[0 * plane + idx]);
        float w = fabsf(ur_full[1 * plane + idx]);
        // IEEE-compliant division, bypassing --use_fast_math approximation
        m = __fadd_rn(__fdiv_rn(u, dx), __fdiv_rn(w, dz));
    }

    // warp-level max
    for (int s = 16; s > 0; s >>= 1)
        m = fmaxf(m, __shfl_down_sync(0xffffffff, m, s));

    s_max[tid] = ((tid & 31) == 0) ? m : 0.0f;
    __syncthreads();

    if (tid < 8) {
        m = s_max[tid * 32];
    } else {
        m = 0.0f;
    }

    for (int s = 4; s > 0; s >>= 1) {
        m = fmaxf(m, __shfl_down_sync(0xffffffff, m, s));
    }

    if (tid == 0) {
        out_blocks[blockIdx.x] = m;
    }
}

static float gpu_compute_cflm(DnsDeviceState *S, float dx, float dz)
{
    const size_t plane = (size_t)S->NX_full * (size_t)S->NZ_full;
    const int block = 256;
    const int grid  = (int)((plane + (size_t)block - 1) / (size_t)block);

    k_cflm_reduce_ur_full<<<grid, block>>>(
        S->d_ur_full, plane, dx, dz, S->d_cflm_scratch);

    std::vector<float> block_max((size_t)grid);
    cudaMemcpy(block_max.data(), S->d_cflm_scratch, block_max.size() * sizeof(float),
               cudaMemcpyDeviceToHost);

    float h_cflm = 0.0f;
    for (float v : block_max) {
        h_cflm = fmaxf(h_cflm, v);
    }
    return h_cflm;
}

// ======================================================================
// Fortran-like NEXTDT, operating on UR_full (3N/2 x 3N/2).
// CFLM reduction is done on the GPU; only a single float is copied back.
// ======================================================================
void next_dt_gpu(DnsDeviceState *S)
{
    const float PI = 3.14159265358979323846f;

    // DX, DZ computed exactly like Fortran: 2*PI/NX, 2*PI/NZ
    const float DX = 2.0f * PI / float(S->NX);
    const float DZ = 2.0f * PI / float(S->NZ);

    // Initial DT branch: IF ((IT.EQ.0).AND.(DT.EQ.0))
    if (S->it == 0 && S->dt == 0.0f) {
        float CFLM = gpu_compute_cflm(S, DX, DZ);
        S->dt   = S->cflnum / CFLM / PI;
        S->cflm = CFLM;

        printf(" [NEXTDT INIT] CFLM=%12.7f  DT=%12.7f  CN=%12.7f\n",
               CFLM, S->dt, S->cn);
        return;
    }

    // Regular update: IF (MOD(IT,IFN).EQ.0)
    if (S->ifn > 0 && (S->it % S->ifn) == 0) {
        float CFLM = gpu_compute_cflm(S, DX, DZ);
        float CFL  = CFLM * S->dt * PI;
        S->cn   = 0.8f + 0.2f * (S->cflnum / CFL);
        S->dt   = S->dt * S->cn;
        S->cflm = CFLM;
    }
}

float compute_cflm(const DnsDeviceState *S)
{
    return S->cflm;
}

// ======================================================================
// Initialization
// ======================================================================

bool dnsCudaInit(DnsDeviceState *S)
{
    // PAO: build initial velocity spectrum UC, alfa/gamma, viscosity
    if (!dnsCudaPaoHostInit(S)) {
        return false;
    }

    //dnsCudaDumpUCFullCsv(S, 0, "UC_u_cuda_pao.csv");
    //dnsCudaDumpUCFullCsv(S, 1, "UC_v_cuda_pao.csv");

    // CALCOM: build spectral vorticity OM2 from UC
    dnsCudaCalcom(S);

    // Do NOT call next_dt_gpu here; Fortran calls NEXTDT
    // only after the first STEP2A has produced UR.
    return true;
}

void gpu_debug_real(const real *d_ptr, int N, const char *label)
{
    std::vector<float> h(N);
    cudaMemcpy(h.data(), d_ptr, N * sizeof(float), cudaMemcpyDeviceToHost);

    float minv =  1.0e30f;
    float maxv = -1.0e30f;
    unsigned nan_count = 0;
    unsigned inf_count = 0;

    for (int i = 0; i < N; ++i) {
        float x = h[i];

        if (isnan(x)) {
            ++nan_count;
            continue;
        }
        if (isinf(x)) {
            ++inf_count;
            continue;
        }
        if (x < minv) minv = x;
        if (x > maxv) maxv = x;
    }

    if (nan_count + inf_count == (unsigned)N) {
        minv = 0.0f;
        maxv = 0.0f;
    }

    printf("[%s] min=%12.7f max=%12.7f nan=%u inf=%u\n",
           label, (double)minv, (double)maxv, nan_count, inf_count);
}

static void computeFieldRange(const std::vector<float>& h,
                              bool clipped,
                              double lo_pct,
                              double hi_pct,
                              float& minv,
                              float& maxv,
                              float& lo,
                              float& hi)
{
    minv =  1.0e30f;
    maxv = -1.0e30f;
    size_t finite_count = 0;

    for (float val : h) {
        if (!std::isfinite(val)) continue;
        if (val < minv) minv = val;
        if (val > maxv) maxv = val;
        ++finite_count;
    }

    if (finite_count == 0) {
        minv = 0.0f;
        maxv = 0.0f;
    }

    lo = minv;
    hi = maxv;
    if (!clipped || finite_count == 0 || maxv - minv <= 1.0e-12f) {
        return;
    }

    lo_pct = std::max(0.0, std::min(1.0, lo_pct));
    hi_pct = std::max(0.0, std::min(1.0, hi_pct));
    if (!(lo_pct < hi_pct)) {
        return;
    }

    constexpr int bins = 8192;
    std::vector<size_t> hist(bins, 0);
    const double inv_range = (double)(bins - 1) / (double)(maxv - minv);

    for (float val : h) {
        if (!std::isfinite(val)) continue;
        int b = (int)(((double)val - (double)minv) * inv_range);
        if (b < 0) b = 0;
        if (b >= bins) b = bins - 1;
        ++hist[(size_t)b];
    }

    const size_t lo_rank = (size_t)std::floor(lo_pct * (double)(finite_count - 1));
    const size_t hi_rank = (size_t)std::floor(hi_pct * (double)(finite_count - 1));
    int lo_bin = 0;
    int hi_bin = bins - 1;
    size_t acc = 0;

    for (int b = 0; b < bins; ++b) {
        acc += hist[(size_t)b];
        if (acc > lo_rank) {
            lo_bin = b;
            break;
        }
    }

    acc = 0;
    for (int b = 0; b < bins; ++b) {
        acc += hist[(size_t)b];
        if (acc > hi_rank) {
            hi_bin = b;
            break;
        }
    }

    const double bin_width = (double)(maxv - minv) / (double)bins;
    lo = (float)((double)minv + bin_width * (double)lo_bin);
    hi = (float)((double)minv + bin_width * (double)(hi_bin + 1));
    if (!(hi > lo)) {
        lo = minv;
        hi = maxv;
    }
}

static void dumpFieldAsPGMFullWithRange(DnsDeviceState *S,
                                        int comp,
                                        const char *filename,
                                        bool clipped,
                                        double lo_pct,
                                        double hi_pct)
{
    const int nx = S->NX_full;
    const int nz = S->NZ_full;
    const size_t planeSize = (size_t)nx * (size_t)nz;

    std::vector<float> h(planeSize);

    cudaMemcpy(h.data(), S->d_ur_full + planeSize * (size_t)comp,
               h.size() * sizeof(float), cudaMemcpyDeviceToHost);

    float minv, maxv, lo, hi;
    computeFieldRange(h, clipped, lo_pct, hi_pct, minv, maxv, lo, hi);

    FILE *f = fopen(filename, "wb");
    if (!f) { perror("fopen"); return; }

    fprintf(f, "P5\n%d %d\n255\n", nx, nz);

    float range = hi - lo;

    // ------------------------------------------------------------
    // Map [lo, hi] → [1, 255]
    // If nearly constant field, use mid-grey 128
    // ------------------------------------------------------------
    if (range <= 1e-12f) {
        // field is essentially constant
        unsigned char c = (unsigned char)128;
        for (int j = 0; j < nz; ++j) {
            for (int i = 0; i < nx; ++i) {
                fwrite(&c, 1, 1, f);
            }
        }
    } else {
        for (int j = 0; j < nz; ++j) {
            for (int i = 0; i < nx; ++i) {
                size_t idx = (size_t)j * (size_t)nx + (size_t)i;
                float val = h[idx];

                // normalize to [0,1]
                float norm = std::isfinite(val) ? (val - lo) / range : 0.5f;

                // scale to [1,255]
                float pixf = 1.0f + norm * 254.0f;
                int   pix  = (int)(pixf + 0.5f);
                if (pix < 1)   pix = 1;
                if (pix > 255) pix = 255;

                unsigned char c = (unsigned char)pix;
                fwrite(&c, 1, 1, f);
            }
        }
    }

    fclose(f);
    if (clipped) {
        printf("[DUMP] Wrote %s (PGM, %dx%d, comp=%d, min=%g, max=%g, clip=%g..%g)\n",
               filename, nx, nz, comp, (double)minv, (double)maxv,
               (double)lo, (double)hi);
    } else {
        printf("[DUMP] Wrote %s (PGM, %dx%d, comp=%d, min=%g, max=%g)\n",
               filename, nx, nz, comp, (double)minv, (double)maxv);
    }
}

void dnsCudaDumpFieldAsPGMFull(DnsDeviceState *S, int comp,
                               const char *filename)
{
    dumpFieldAsPGMFullWithRange(S, comp, filename, false, 0.0, 1.0);
}

void dnsCudaDumpFieldAsPGMFullClipped(DnsDeviceState *S,
                                      int comp,
                                      const char *filename,
                                      double lo_pct,
                                      double hi_pct)
{
    dumpFieldAsPGMFullWithRange(S, comp, filename, true, lo_pct, hi_pct);
}

// Dump OM2(kx,z) as CSV with Re,Im per line, Fortran-compatible ordering.
// Fortran loop was: DO I = 1,N ; DO K = 1,N/2  (K=kx, I=z)
void dnsCudaDumpOM2Csv(DnsDeviceState *S, int step, const char *base)
{
    int NX2 = S->NX / 2;   // N/2
    int NZ  = S->NZ;       // N

    int n = NX2 * NZ;
    std::vector<cplx> h(n);

    cudaMemcpy(h.data(),
               S->d_om2,
               n * sizeof(cplx),
               cudaMemcpyDeviceToHost);

    char fname[64];
    std::snprintf(fname, sizeof(fname), "%s_step%d.csv", base, step);

    FILE *f = std::fopen(fname, "w");
    if (!f) {
        std::perror("dnsCudaDumpOM2Csv fopen");
        return;
    }

    // Match Fortran order I=1..N (z), K=1..N/2 (kx)
    for (int iz = 0; iz < NZ; ++iz) {
        for (int ik = 0; ik < NX2; ++ik) {
            int idx = ik + NX2 * iz;         // [z][kx]
            cplx v  = h[idx];
            std::fprintf(f, "% .14e,% .14e\n", v.x, v.y);
        }
    }

    std::fclose(f);
    std::printf("[DUMP] Wrote %s (OM2, NX/2=%d, NZ=%d)\n",
                fname, NX2, NZ);
}

// ======================================================================
// One full time step: 2B → 3 → 2A → NEXTDT
// (mirrors Fortran VISASUB ordering; still uses compact grid for now)
// ======================================================================

void dnsCudaStep(DnsDeviceState *S, int step)
{
    int n_ur_compact = S->NX * S->NZ * 3;
    int n_uc_compact = S->NZ * S->NK * 3;
    int n_om         = S->NZ * (S->NX / 2);

    printf("************ ITERATION: %d\n", step);

    // UR (compact) as input from previous step (or PAO for step=0)
    gpu_debug_real(S->d_ur, n_ur_compact, "UR BEFORE 2B");

    // 1) STEP2B: quadratic / linear term in spectral space
    dnsCudaStep2B(S);
    gpu_debug_real((real*)S->d_uc, n_uc_compact, " UC AFTER 2B");

    // 2) STEP3: one-diagonal solve with current DT/CN (updates UC, OM2)
    dnsCudaStep3(S);
    gpu_debug_real((real*)S->d_uc,  n_uc_compact, " UC AFTER 3");
    gpu_debug_real((real*)S->d_om2, 2 * n_om,     "OM2 AFTER 3");

    // Dump OM2 AFTER STEP3, like Fortran DUMP_OM2_CSV
    dnsCudaDumpOM2Csv(S, step, "OM2_cuda");

    // Optional: Fortran prints "Max |UC| before STEP2A" here
    float max_uc_before_2A = gpu_maxabs_cplx(S->d_uc, S->NZ * S->NK);
    printf("Max |UC| before STEP2A = %12.7f\n", max_uc_before_2A);

    // 3) STEP2A: UC -> UR (nonlinear part, phys space)
    dnsCudaStep2A(S);
    gpu_debug_real(S->d_ur,        n_ur_compact, " UR AFTER 2A");
    gpu_debug_real((real*)S->d_uc, n_uc_compact, " UC AFTER 2A");
    dnsCudaDumpFieldAsPGMFull(S, 0, "pao_u_cuda_full.pgm");
    dnsCudaDumpFieldAsPGMFull(S, 1, "pao_v_cuda_full.pgm");

    // 4) NEXTDT: compute DT from UR (for NEXT iteration)
    next_dt_gpu(S);
    // printf("  [NEXTDT] DT=%12.7f  CN=%12.7f\n", S->dt, S->cn);

    // Time update: same semantics as before (STEP3 in Fortran updates T;
    // here we keep the old behaviour and advance T after NEXTDT).
    S->t  += S->dt;
    S->it += 1;
}

// ======================================================================
// Snapshot (simple implementation; comp=0,1,2)
// ======================================================================

void dnsCudaSnapshot(const DnsDeviceState *S,
                     int comp,
                     real *host_plane,
                     int nx,
                     int ny)
{
    if (nx != S->NX || ny != S->NZ) {
        // simple, no fancy checks — Option A: "just make it work"
        return;
    }

    size_t total = (size_t)S->NX * S->NZ * 3;
    std::vector<real> tmp(total);
    cudaMemcpy(tmp.data(), S->d_ur, total*sizeof(real), cudaMemcpyDeviceToHost);

    for (int z=0; z<S->NZ; ++z) {
        for (int x=0; x<S->NX; ++x) {
            int idx_ur = UR_INDEX(x,z,comp,S->NX);
            int idx_pl = x + nx*z;
            host_plane[idx_pl] = tmp[idx_ur];
        }
    }
}

// ======================================================================
// Run loop with simple debug prints
// ======================================================================

void dnsCudaRun(DnsDeviceState *S, int nsteps)
{
    int Ncells = S->NX * S->NZ;

    float max_ur0 = gpu_maxabs_real(S->d_ur, Ncells*3);
    printf("Max |UR| before STEP2A = %12.7f\n", max_ur0);

    for (int i=0; i<nsteps; ++i)
    {
        dnsCudaStep(S, i);

        float max_uc = gpu_maxabs_cplx(S->d_uc, S->NZ*S->NK*3);
        float max_om = gpu_maxabs_cplx(S->d_om2, S->NZ*(S->NX/2));

        printf(" ITERATION %6d T=%12.10f DT=%10.8f CN=%10.8f CFLM=%.6f\n",
               i, S->t, S->dt, S->cn, compute_cflm(S));
        printf("max|UC|=%12.7f  max|OM2|=%12.7f\n", max_uc, max_om);
    }
}

// ======================================================================
// Dump UR component as a PGM image (Fortran FIELD2PIX-style)
// ======================================================================
void dnsCudaDumpFieldAsPGM(DnsDeviceState* S, int comp, const char* filename)
{
    if (!S) return;

    const int nx = S->NX;   // 128
    const int nz = S->NZ;   // 128

    // Copy compact ur (spectral 1× grid) to host
    std::vector<float> h_ur(nx * nz * 3);
    cudaMemcpy(h_ur.data(),
               S->d_ur,
               h_ur.size() * sizeof(float),
               cudaMemcpyDeviceToHost);

    FILE* fp = std::fopen(filename, "wb");
    if (!fp) {
        std::perror("dnsCudaDumpFieldAsPGM: fopen");
        return;
    }

    // PGM header: P5 = binary grayscale, width, height, maxval
    std::fprintf(fp, "P5\n%d %d\n255\n", nx, nz);

    const float enfac = 75.0f;  // same as Fortran FIELD2PIX

    for (int j = 0; j < nz; ++j) {
        for (int i = 0; i < nx; ++i) {
            const int idx = (j * nx + i) * 3 + comp;  // <--- key line
            float v = h_ur[idx];

            int gray = static_cast<int>(std::floor(enfac * v) + 128.0f);
            if (gray < 1)   gray = 1;
            if (gray > 255) gray = 255;

            unsigned char c = static_cast<unsigned char>(gray);
            std::fwrite(&c, 1, 1, fp);
        }
    }

    std::fclose(fp);
    std::printf("[DUMP] Wrote %s (PGM, %dx%d, comp=%d)\n",
                filename, nx, nz, comp);
}

// ----------------------------------------------------------------------
// Dump UR_full(:,:,comp) as raw float32 binary [nz, nx] (row-major)
// ----------------------------------------------------------------------
// Dump UR_full (3/2 grid, 192x192) for a single component as raw float32
void dnsCudaDumpUrFullBinary(DnsDeviceState* S, int comp, const char* filename)
{
    if (!S) return;

    const int nx = S->NX_full;  // 3N/2 = 192
    const int nz = S->NZ_full;  // 3N/2 = 192
    const size_t plane = (size_t)nx * (size_t)nz;

    std::vector<float> h_ur_full(plane * 3);
    cudaMemcpy(h_ur_full.data(),
               S->d_ur_full,
               h_ur_full.size() * sizeof(float),
               cudaMemcpyDeviceToHost);

    FILE* fp = std::fopen(filename, "wb");
    if (!fp) {
        std::perror("dnsCudaDumpUrFullBinary: fopen");
        return;
    }

    for (int j = 0; j < nz; ++j) {
        for (int i = 0; i < nx; ++i) {
            size_t lin = (size_t)j * (size_t)nx + (size_t)i;
            float v = h_ur_full[(size_t)comp * plane + lin];
            std::fwrite(&v, sizeof(float), 1, fp);
        }
    }

    std::fclose(fp);
    std::printf("[DUMP] wrote %s (float32 %dx%d, comp=%d)\n",
                filename, nx, nz, comp);
}

// ----------------------------------------------------------------------
// Dump UC_full(:,:,comp) to ASCII CSV, Fortran DNSDUMPCSV-style
//
// Fortran side:
//   REAL    UR(NX3D2+2, NZ3D2, 3)
//   CALL DNSDUMPCSV('UC_ur_u_fortran.csv', UC, N3D2, N3D2, 1)
//   ...
//
// There UC is complex; passing it to a REAL routine effectively writes
// only the REAL part of UC(I,J,AXIS). Here we mirror that:
//
//   • CSV size: NX_full x NZ_full  (N3D2 x N3D2)
//   • Loops:    I ↔ x, J ↔ z
//   • Value:    Re( UC_full(I,J,comp) ) for I < NK_full, else 0.
// ----------------------------------------------------------------------
void dnsCudaDumpUCFullCsv(const DnsDeviceState* S, int comp, const char* fname)
{
    if (!S) return;

    const int N      = S->Nbase;    // logical base N (e.g. 16, 128, ...)
    const int nx     = S->NX_full;  // 3*N/2, Fortran N3D2
    const int nz     = S->NZ_full;  // 3*N/2
    const int nk     = S->NK_full;  // spectral width (3*N/4+1)
    const int nd2    = N / 2;       // ND2 in Fortran

    // Pull UC_full from device: shape [nz][nk][3]
    std::vector<cplx> h_uc_full((size_t)nz * (size_t)nk * 3);
    cudaMemcpy(h_uc_full.data(),
               S->d_uc_full,
               h_uc_full.size() * sizeof(cplx),
               cudaMemcpyDeviceToHost);

    FILE* fp = std::fopen(fname, "w");
    if (!fp) {
        std::perror("dnsCudaDumpUCFullCsv: fopen");
        return;
    }

    // Fortran DNSDUMPCSV (with EQUIVALENCE(UC,UR)) effectively lays out:
    //
    //   row 0   → Re( UC(kx=0, z, comp) )
    //   row 1   → Im( UC(kx=0, z, comp) )
    //   row 2   → Re( UC(kx=1, z, comp) )
    //   row 3   → Im( UC(kx=1, z, comp) )
    //   ...
    //   rows >= 2*ND2 → zero
    //
    // Our previous CUDA version only wrote Re(UC) at row i = kx.
    // Here we explicitly interleave Re/Im so that the CSV matches Fortran.

    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < nz; ++j) {

            float val = 0.0f;

            if (i < 2 * nd2) {
                int kx  = i / 2;        // 0..ND2-1
                bool imag_row = (i & 1);

                if (kx < nk) {
                    size_t idx = UC_FULL_INDEX(kx, j, comp, nk, nz);
                    cplx v  = h_uc_full[idx];

                    val = imag_row ? v.y : v.x;   // even row → Re, odd row → Im
                }
            }

            std::fprintf(fp, "% .10E", val);
            if (j < nz - 1) {
                std::fprintf(fp, ",");
            }
        }
        std::fprintf(fp, "\n");
    }

    std::fclose(fp);
    std::printf("[CSV] Wrote %s (UC_full, %dx%d, comp=%d, Fortran layout)\n",
                fname, nx, nz, comp);
}

// ----------------------------------------------------------------------
// Dump UR_full(:,:,comp) to ASCII CSV, Fortran DNSDUMPCSV-style
//
// Fortran side:
//   REAL    UR(NX3D2+2, NZ3D2, 3)
//   DO I = 1, NX3D2
//     DO J = 1, NZ3D2
//        WRITE(...) UR(I,J,AXIS)
//     ...
//
// Here we mirror that logical indexing, but using our C layout:
//   UR_FULL_INDEX(x,z,c,NX_full,NZ_full)
// ----------------------------------------------------------------------
void dnsCudaDumpURFullCsv(const DnsDeviceState* S, int comp, const char* fname)
{
    if (!S) return;

    const int nx = S->NX_full;  // 3*N/2, corresponds to NX3D2 in Fortran
    const int nz = S->NZ_full;  // 3*N/2, corresponds to NZ3D2

    // Pull full-grid UR from device
    std::vector<float> h_ur_full((size_t)nx * (size_t)nz * 3);
    cudaMemcpy(h_ur_full.data(),
               S->d_ur_full,
               h_ur_full.size() * sizeof(float),
               cudaMemcpyDeviceToHost);

    FILE* fp = std::fopen(fname, "w");
    if (!fp) {
        std::perror("dnsCudaDumpURFullCsv: fopen");
        return;
    }

    // Fortran loop order:
    //   DO I = 1, NX3D2
    //      DO J = 1, NZ3D2
    //         UR(I,J,AXIS)
    // I ↔ x, J ↔ z, AXIS ↔ comp
    for (int i = 0; i < nx; ++i) {      // I = 1..NX3D2
        for (int j = 0; j < nz; ++j) {  // J = 1..NZ3D2
            size_t idx = UR_FULL_INDEX(i, j, comp, nx, nz);
            float v = h_ur_full[idx];

            // E18.10 style
            std::fprintf(fp, "% .10E", v);
            if (j < nz - 1) {
                std::fprintf(fp, ",");
            }
        }
        std::fprintf(fp, "\n");
    }

    std::fclose(fp);
    std::printf("[CSV] Wrote %s (UR_full, %dx%d, comp=%d)\n",
                fname, nx, nz, comp);
}
