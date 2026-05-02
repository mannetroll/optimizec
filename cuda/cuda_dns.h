#pragma once

#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>

// ===============================================================
// Type aliases
// ===============================================================
using real = float;
using cplx = cufftComplex;

// ===============================================================
// Optional phase timing hooks
// ===============================================================
enum DnsPhaseId
{
    DNS_PHASE_STEP2B_BUILD = 0,
    DNS_PHASE_STEP2B_FFT,
    DNS_PHASE_STEP2B_ZERO_MIDDLE,
    DNS_PHASE_STEP3,
    DNS_PHASE_STEP2A_PREPARE,
    DNS_PHASE_STEP2A_FFT,
    DNS_PHASE_NEXTDT_CFLM,
    DNS_PHASE_OM2PHYS,
    DNS_PHASE_DISPLAY_SIGMA,
    DNS_PHASE_COUNT
};

extern bool g_dns_phase_timing_enabled;
bool dnsPhaseTimingEnabled();
void dnsPhaseTimingSetEnabled(bool enabled);
void dnsPhaseTimingReset();
void dnsPhaseTimingAdd(DnsPhaseId id, float ms);
void dnsPhaseTimingReport();

#define DNS_PHASE_TIME(ID, ...)                                             \
    do {                                                                     \
        if (dnsPhaseTimingEnabled()) {                                       \
            cudaEvent_t _dns_phase_start;                                    \
            cudaEvent_t _dns_phase_stop;                                     \
            cudaEventCreate(&_dns_phase_start);                              \
            cudaEventCreate(&_dns_phase_stop);                               \
            cudaEventRecord(_dns_phase_start, 0);                            \
            do { __VA_ARGS__ } while (0);                                    \
            cudaEventRecord(_dns_phase_stop, 0);                             \
            cudaEventSynchronize(_dns_phase_stop);                           \
            float _dns_phase_ms = 0.0f;                                      \
            cudaEventElapsedTime(&_dns_phase_ms,                             \
                                 _dns_phase_start,                           \
                                 _dns_phase_stop);                           \
            dnsPhaseTimingAdd((ID), _dns_phase_ms);                          \
            cudaEventDestroy(_dns_phase_stop);                               \
            cudaEventDestroy(_dns_phase_start);                              \
        } else {                                                             \
            do { __VA_ARGS__ } while (0);                                    \
        }                                                                    \
    } while (0)

// ===============================================================
// Row-major indexing macros (Option B.1 layout = YOUR CURRENT LAYOUT)
// ---------------------------------------------------------------
// UR[z][x][comp]  → linear: (comp) + 3*((x) + NX*(z))
// UC[z][kx][comp] → linear: (comp) + 3*((kx) + NK*(z))
//
// NOTE:
//   • UC stores spectral Ux,U z,NONLINEAR in comps 0,1,2
//   • UR stores physical Ux,Uz, and temporary nonlinear (comp=2)
// ===============================================================

// Compact N×N (AoS): [z][x][comp], comp fastest
#define UR_INDEX(x,z,c,NX)   ((c) + 3 * ((x) + (NX) * (z)))
#define UC_INDEX(kx,z,c,NK)  ((c) + 3 * ((kx) + (NK) * (z)))

// Full 3/2-grid UC layout: [comp][z][kx], x/kx fastest (SoA)
#define UC_FULL_INDEX(kx,z,c,NK_full,NZ_full) \
    ((size_t)(kx) + (size_t)(NK_full) * ((size_t)(z) + (size_t)(NZ_full) * (size_t)(c)))

// Full 3/2-grid UR layout: [comp][z][x], x fastest (SoA)
#define UR_FULL_INDEX(x,z,c,NX_full,NZ_full) \
    ((size_t)(x) + (size_t)(NX_full) * ((size_t)(z) + (size_t)(NZ_full) * (size_t)(c)))

// ===============================================================
// DnsDeviceState — DEVICE ARRAYS + FFT PLANS + PARAMETERS
// ===============================================================
struct DnsDeviceState
{
    // Logical resolution and current working grid
    int Nbase;      // logical base N (128)
    int N;          // kept for compatibility (== Nbase)
    int NX, NZ;     // current working grid (still == Nbase for now)
    int NK;         // current spectral width (3*Nbase/4+1)

    // --- Future full 3/2 grid (not used yet) ------------------------
    int NX_full;    // 3*Nbase/2  (physical X, Fortran UR dimension)
    int NZ_full;    // 3*Nbase/2  (physical Z)
    int NK_full;    // 3*Nbase/4+1 (same as NK, but for clarity)

    real Re;
    real K0;
    real visc;

    real t;
    real dt;
    real cn;
    real cnm1;
    real cflnum;
    real cflm;
    int  it;
    int  ifn;

    // device pointers: current working grid
    real *d_ur;     // NX * NZ * 3
    cplx *d_uc;     // NZ * NK * 3
    cplx *d_om2;    // NZ * (NX/2)
    cplx *d_fnm1;   // same shape as om2

    real *d_alfa;   // NX/2
    real *d_gamma;  // NZ

    // device pointers: future full grid (3/2), not used yet
    real *d_ur_full;   // NX_full * NZ_full * 3
    cplx *d_uc_full;   // NZ_full * NK_full * 3

    // Per-block CFLM reduction scratch (one float per block)
    real *d_cflm_scratch;
    int   cflm_num_blocks;

    // Per-block display sigma reduction scratch.
    real *d_sigma_minmax;              // sigma_num_blocks mins, then maxes
    unsigned long long *d_sigma_sums;  // sigma_num_blocks sums, then sumsq
    int   sigma_num_blocks;

    // STEP3 scalar.
    real  step3_divxz;        // scalar DIVXZ = 1/(NX_full*NZ_full)

    // FFT plans (full 3/2 grid) — 1D per-direction, per-component (legacy)
    cufftHandle plan_full_c2r_x;
    cufftHandle plan_full_c2c_z;
    cufftHandle plan_full_r2c_x;

    // 2D batched FFT plans (all 3 components in one call)
    cufftHandle plan_full_r2c_2d;
    cufftHandle plan_full_c2r_2d;
    cufftHandle plan_full_c2r_2d_one;
    void *d_fft_work;
    size_t fft_work_size;

    bool owns_plans;
};

// ===============================================================
// Life-cycle functions
// ===============================================================
bool dnsCudaCreate(DnsDeviceState *S, int N, real Re, real K0);
void dnsCudaDestroy(DnsDeviceState *S);

// ===============================================================
// Initialization
// ===============================================================

/// EXACT Fortran PAO — sets ALFA/GAMMA, generates initial UC (vel),
/// computes viscosity = SQRT(Q2**2 / (RE * W2)), reshuffles.
bool dnsCudaPaoHostInit(DnsDeviceState *S);
void dnsCudaStep2A_full_debug(DnsDeviceState *S);

void dnsCudaCalcom(DnsDeviceState *S);

/// Zeros UR, UC, OM2, FNM1 and loads ALFA/GAMMA (after PAO)
bool dnsCudaInit(DnsDeviceState *S);

// ===============================================================
// Time stepping: FULL STEP = 2A → 2B → 3 → dt update
// ===============================================================
void dnsCudaStep(DnsDeviceState *S);
void dnsCudaRun(DnsDeviceState *S, int nsteps);

// ===============================================================
// Snapshot API
// ===============================================================
/// Copies UR[:,:,comp] → host_plane[z][x]
void dnsCudaSnapshot(const DnsDeviceState *S,
                     int comp,
                     real *host_plane,
                     int nx,
                     int ny);

// ===============================================================
// FFT interface — EXACT FORTRAN RULES:
//   • Forward:  NO SCALING
//   • Inverse: scale by 1/(NX*NZ) ONLY
// ===============================================================
void vfft_full_inverse_uc_full_to_ur_full(DnsDeviceState *S);
void vfft_full_forward_ur_full_to_uc_full(DnsDeviceState *S);
void vfft_step2b_forward_ur_full_to_uc_needed(DnsDeviceState *S);


// ===============================================================
// Next time step size (CFL rule)
// ===============================================================
void next_dt_gpu(DnsDeviceState *S);
float compute_cflm(const DnsDeviceState *S);

// ===============================================================
// Low-level DNS steps
// ===============================================================
void dnsCudaStep2A(DnsDeviceState *S);
void dnsCudaStep2B(DnsDeviceState *S);
void dnsCudaStep3(DnsDeviceState *S);

void dnsCudaDumpUCFullCsv(const DnsDeviceState* S, int comp, const char* fname);
void dnsCudaDumpFieldAsPGM(DnsDeviceState* S, int comp, const char* filename);
void dnsCudaDumpFieldAsPGMFull(DnsDeviceState *S, int comp, const char *filename);
void dnsCudaDumpFieldAsPGMFullClipped(DnsDeviceState *S,
                                      int comp,
                                      const char *filename,
                                      double lo_pct,
                                      double hi_pct);

int dnsCudaMovieScaleF(int N);
bool dnsCudaSaveFieldInfernoPng(DnsDeviceState *S,
                                int comp,
                                const char *filename,
                                int scale_f);

void dnsCudaOm2Phys(DnsDeviceState *S);
void dnsCudaStreamFunc(DnsDeviceState *S);
void dnsCudaKinetic(DnsDeviceState* S);
void dnsCudaPhiPhys(DnsDeviceState *S);
