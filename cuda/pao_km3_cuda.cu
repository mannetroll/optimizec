//
// pao_km3_cuda.cu — CUDA PAO initializer with k^-3 spectral reshaping
// --------------------------------------------------
// Re-implements PAO(UR,ALFA,GAMMA,N,NE,RE,K0,VISC) *without* any FFT.
// It builds a random isotropic spectral field and computes VISC.
// The field is left in spectral space (in d_uc), just like PAO.f
// leaves UC filled and the FFTs are done later in the DNS pipeline.
//

#include "cuda_dns.h"
#include <cuda_runtime.h>

#include <vector>
#include <complex>
#include <cmath>
#include <cstdio>

__global__
void k_scatter_pao_uc_to_full(const cplx *uc_compact,
                              cplx *uc_full,
                              int ND2,
                              int NE,
                              int NK_full,
                              int NZ_full)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int z = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= ND2 || z >= NE) return;

    const size_t compact_base = ((size_t)x + (size_t)ND2 * (size_t)z) * 2;
    for (int c = 0; c < 2; ++c) {
        uc_full[UC_FULL_INDEX(x, z, c, NK_full, NZ_full)] = uc_compact[compact_base + c];
    }
}

// Fortran-style LCG random:
//   ISEED = MOD(ISEED*IMM + IT, ID)
static inline float frand(int &seed)
{
    const int IMM = 420029;
    const int IT  = 2017;
    const int ID  = 5011;

    seed = (seed*IMM + IT) % ID;
    return float(seed) / float(ID);
}

static inline double pao_hash01(int x, int z, int seed, int salt)
{
    const double a =
        (double(x) + 1.0) * 12.9898 +
        (double(z) + 1.0) * 78.233 +
        (double(seed) + 1.0) * 0.010001 +
        (double(salt) + 1.0) * 37.719;
    const double s = std::sin(a) * 43758.5453123;
    return s - std::floor(s);
}

bool dnsCudaPaoHostInit(DnsDeviceState *S)
{
    const int   N   = S->NX;   // base grid
    const int   NE  = S->NZ;   // "NE" as in Fortran
    const int   ND2 = N/2;
    const int   NED2= NE/2;
    const float PI  = 3.14159265358979f;

    const float DXZ  = 2.0f * PI / float(N);
    const float K0   = S->K0;
    const float NORM = PI * K0 * K0;
    const float AMP2_FLOOR = 1.0e-40f;

    // ------------------------------------------------------------------
    // Build ALFA(N/2) and GAMMA(N)  (Fortran DALFA, DGAMMA, E1, E3)
    // ------------------------------------------------------------------
    std::vector<float> alfa(ND2);
    std::vector<float> gamma(NE);

    float E1 = 1.0f;
    float E3 = 1.0f / E1;

    float DALFA  = 1.0f / E1;
    float DGAMMA = 1.0f / E3;

    for (int x=0; x<NED2; ++x)
        alfa[x] = float(x) * DALFA;

    gamma[0] = 0.0f;
    for (int z=1; z<=NED2; ++z)
    {
        gamma[z]      = float(z) * DGAMMA;
        gamma[NE-z]   = -gamma[z];
    }

    // ship to device for later use in STEP2A/STEP3
    cudaMemcpy(S->d_alfa,  alfa.data(),  ND2*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(S->d_gamma, gamma.data(), NE *sizeof(float), cudaMemcpyHostToDevice);

    // ------------------------------------------------------------------
    // Host spectral UR: complex field UR(kx,z,comp)
    // comp=0 → u1, comp=1 → u3 (Fortran components 1 and 2)
    //
    // We only need X=1..ND2, Z=1..NE, I=1..2 for PAO’s sums and reshuffle,
    // so we store exactly that subset as:
    //
    //   UR[x,z,c]  where  x ∈ [0..ND2-1], z ∈ [0..NE-1], c ∈ {0,1}
    // ------------------------------------------------------------------
    std::vector<std::complex<float>> UR((size_t)ND2 * (size_t)NE * 2);
    auto URat = [&](int x,int z,int c) -> std::complex<float>& {
        return UR[(size_t)c + 2 * ((size_t)x + (size_t)ND2 * (size_t)z)];
    };

    for (auto &v : UR) v = std::complex<float>(0.0f,0.0f);

    // ------------------------------------------------------------------
    // Fortran random vector RANVEC(97)
    // ------------------------------------------------------------------
    const int seed_init = 1;
    int   seed   = seed_init;       // mimics ISEED SAVE; we bump it immediately
    float RANVEC[97];

    // "warm-up" 97 calls
    for (int i=0; i<97; ++i)
        frand(seed);

    // fill RANVEC
    for (int i=0; i<97; ++i)
        RANVEC[i] = frand(seed);

    auto random_from_vec = [&](float r) -> float {
        int idx = int(r * 97.0f);
        if (idx < 0)   idx = 0;
        if (idx > 96)  idx = 96;
        float v = RANVEC[idx];
        RANVEC[idx] = r;
        return v;
    };

    // ------------------------------------------------------------------
    // Generate isotropic random spectrum (Fortran DO 500/510 loops)
    // ------------------------------------------------------------------
    for (int z=0; z<NE; ++z)
    {
        for (int x=0; x<NED2; ++x)
        {
            float r   = frand(seed);
            float th  = 2.0f * PI * random_from_vec(r);

            std::complex<float> ARG(std::cos(th), std::sin(th));

            float ax = alfa[x];
            float gz = gamma[z];
            float K  = std::sqrt(ax*ax + gz*gz);

            if (ax == 0.0f)
            {
                // ALFA(X) == 0: purely u1 mode
                URat(x,z,1) = std::complex<float>(0.0f,0.0f);

                float ABSU2 = std::exp(-(K/K0)*(K/K0)) / NORM;
                if (ABSU2 < AMP2_FLOOR)
                    ABSU2 = AMP2_FLOOR;
                float amp   = std::sqrt(ABSU2);
                URat(x,z,0) = amp * ARG;
            }
            else
            {
                float denom = 1.0f + (gz*gz)/(ax*ax);
                float ABSW2 = std::exp(-(K/K0)*(K/K0)) / (denom * NORM);
                if (ABSW2 < AMP2_FLOOR)
                    ABSW2 = AMP2_FLOOR;
                float ampw  = std::sqrt(ABSW2);

                std::complex<float> w = ampw * ARG;
                std::complex<float> u = -(gz/ax) * w;  // -GAMMA/ALFA * UR(.,.,2)

                URat(x,z,1) = w;
                URat(x,z,0) = u;
            }
        }
    }

    // Special zero modes (UR(1,1,1)=0, UR(1,1,2)=0 in 1-based Fortran)
    URat(0,0,0) = std::complex<float>(0.0f,0.0f);
    URat(0,0,1) = std::complex<float>(0.0f,0.0f);

    // ------------------------------------------------------------------
    // Hermitian symmetry in Z (Fortran DO 600)
    // ------------------------------------------------------------------
    for (int z=1; z<NED2; ++z)
    {
        URat(0, NE - z, 0) = std::conj(URat(0, z, 0));
        URat(0, NE - z, 1) = std::conj(URat(0, z, 1));
    }

    // Zero at Z=NED2+1 (index NED2 in 0-based)
    for (int x=0; x<ND2; ++x)
    {
        URat(x, NED2, 0) = std::complex<float>(0.0f,0.0f);
        URat(x, NED2, 1) = std::complex<float>(0.0f,0.0f);
    }

    // ------------------------------------------------------------------
    // PAO-k^-3 start spectrum:
    //   keep the PAO divergence-free relation, but redistribute each
    //   integer shell toward E(k) ~ k^-3 through k/k_Nyquist <= 1.
    //   This mirrors turbo_simulator.py, including deterministic
    //   per-mode jitter and hash phases.
    // ------------------------------------------------------------------
    {
        const double k_nyquist = double(ND2);
        const int kmax_shell = ND2;
        std::vector<double> shell_energy((size_t)kmax_shell + 1, 0.0);
        std::vector<double> shell_weight((size_t)kmax_shell + 1, 0.0);
        std::vector<double> target_shell((size_t)kmax_shell + 1, 0.0);

        for (int x=0; x<ND2; ++x)
        {
            const double m = (x == 0) ? 1.0 : 2.0;
            const double ax2 = double(alfa[x]) * double(alfa[x]);

            for (int z=0; z<NE; ++z)
            {
                const double gz = double(gamma[z]);
                const double gz2 = gz * gz;
                const double k = std::sqrt(ax2 + gz2);
                const int shell = int(k + 0.5);

                const std::complex<float> U1 = URat(x,z,0);
                const std::complex<float> U3 = URat(x,z,1);
                const double u1u1 = double(U1.real()) * double(U1.real()) +
                                     double(U1.imag()) * double(U1.imag());
                const double u3u3 = double(U3.real()) * double(U3.real()) +
                                     double(U3.imag()) * double(U3.imag());

                if (k > 0.0 && k <= k_nyquist && shell <= kmax_shell)
                {
                    int z_hash = z;
                    if (x == 0 && z > NED2)
                        z_hash = NE - z;

                    const double mode_jitter =
                        0.75 + 0.50 * pao_hash01(x, z_hash, seed_init, 17);

                    shell_energy[(size_t)shell] += m * (u1u1 + u3u3);
                    shell_weight[(size_t)shell] += m * mode_jitter;
                }
            }
        }

        const double k0 = std::max(1.0, double(K0));

        for (int shell=1; shell<=kmax_shell; ++shell)
        {
            const double k = double(shell);
            double shape = 0.0;

            if (k < k0)
            {
                const double low = k / k0;
                shape = (low * low * low * low) / (k0 * k0 * k0);
            }
            else
            {
                shape = 1.0 / (k * k * k);
            }

            target_shell[(size_t)shell] = shape;
        }

        double original_energy = 0.0;
        double target_energy = 0.0;
        for (int shell=1; shell<=kmax_shell; ++shell)
        {
            original_energy += shell_energy[(size_t)shell];
            target_energy += target_shell[(size_t)shell];
        }

        if (original_energy > 0.0 && target_energy > 0.0)
        {
            const double norm = original_energy / target_energy;
            for (int x=0; x<ND2; ++x)
            {
                const double ax = double(alfa[x]);
                const double ax2 = ax * ax;

                for (int z=0; z<NE; ++z)
                {
                    const double gz = double(gamma[z]);
                    const double gz2 = gz * gz;
                    const double k = std::sqrt(ax2 + gz2);
                    const int shell = int(k + 0.5);

                    if (k > 0.0 &&
                        k <= k_nyquist &&
                        shell <= kmax_shell &&
                        shell_weight[(size_t)shell] > 0.0)
                    {
                        int z_hash = z;
                        double phase_sign = 1.0;
                        if (x == 0 && z > NED2)
                        {
                            z_hash = NE - z;
                            phase_sign = -1.0;
                        }

                        const double mode_jitter =
                            0.75 + 0.50 * pao_hash01(x, z_hash, seed_init, 17);
                        const double mode_energy =
                            (norm * target_shell[(size_t)shell]) *
                            mode_jitter / shell_weight[(size_t)shell];
                        const double th =
                            phase_sign * 2.0 * 3.14159265358979323846 *
                            pao_hash01(x, z_hash, seed_init, 53);
                        const std::complex<float> phase(
                            float(std::cos(th)), float(std::sin(th)));

                        if (ax == 0.0)
                        {
                            URat(x,z,0) = float(std::sqrt(mode_energy)) * phase;
                            URat(x,z,1) = std::complex<float>(0.0f,0.0f);
                        }
                        else
                        {
                            const double denom = 1.0 + gz2 / ax2;
                            const std::complex<float> w =
                                float(std::sqrt(mode_energy / denom)) * phase;
                            const std::complex<float> u = float(-gz / ax) * w;
                            URat(x,z,0) = u;
                            URat(x,z,1) = w;
                        }
                    }
                    else
                    {
                        URat(x,z,0) = std::complex<float>(0.0f,0.0f);
                        URat(x,z,1) = std::complex<float>(0.0f,0.0f);
                    }
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // Compute averages A(1..7), E110, Q2, W2, VISC (Fortran DO 800/810)
    // ------------------------------------------------------------------
    double A1=0, A2=0, A3=0, A4=0, A5=0, A6=0, A7=0;
    double E110 = 0;

    for (int x=0; x<ND2; ++x)
    {
        bool x1 = (x == 0);

        for (int z=0; z<N; ++z)
        {
            std::complex<float> U1 = URat(x,z,0);
            std::complex<float> U3 = URat(x,z,1);

            double u1u1 = std::norm(U1);
            double u3u3 = std::norm(U3);

            double ax2 = double(alfa[x])  * double(alfa[x]);
            double gz2 = double(gamma[z]) * double(gamma[z]);
            double K2  = ax2 + gz2;

            double m = x1 ? 1.0 : 2.0;

            A1 += m * u1u1;
            A2 += m * u3u3;
            A3 += m * u1u1 * ax2;
            A4 += m * u1u1 * gz2;
            A5 += m * u3u3 * ax2;
            A6 += m * u3u3 * gz2;
            A7 += m * (u1u1 + u3u3) * K2*K2;

            if (x1) E110 += u1u1;
        }
    }

    double Q2   = A1 + A2;
    double W2   = A3 + A4 + A5 + A6;
    double visc = 1.0 / S->Re;

    S->visc = float(visc);

    // ------------------------------------------------------------------
    // Extra diagnostics (Fortran WRITE block)
    // ------------------------------------------------------------------
    double EP   = visc * W2;
    double De   = 2.0 * visc*visc * A7;
    double KOL  = std::pow(visc*visc*visc / EP, 0.25);
    double NLAM = 0.0;
    if (E110 != 0.0)
        NLAM = 2.0 * A1 / E110;

    // a11, e11, length scales, etc.
    double a11   = 2.0 * A1 / Q2 - 1.0;
    double e11   = 2.0 * (A3 + A4) / W2 - 1.0;
    double tscale= 0.5 * Q2 / EP;
    double dxKol = DXZ / KOL;
    double Lux   = 2.0 * PI / std::sqrt(2.0 * A1 / A3);
    double Luz   = 2.0 * PI / std::sqrt(2.0 * A1 / A4);
    double Lwx   = 2.0 * PI / std::sqrt(2.0 * A2 / A5);
    double Lwz   = 2.0 * PI / std::sqrt(2.0 * A2 / A6);
    double Ceps2 = 0.5 * Q2 * De / (EP*EP);

    std::printf(" Reynolds n. =%12.1f\n", double(S->Re));
    std::printf(" Energy      =%12.7f\n", Q2);
    std::printf(" WiWi        =%12.5f\n", W2);
    std::printf(" Epsilon     =%12.7f\n", EP);
    std::printf(" a11         =%12.7f\n", a11);
    std::printf(" e11         =%12.7f\n", e11);
    std::printf(" Time scale  =%12.4f\n", tscale);
    std::printf(" Kolmogorov  =%12.7f\n", KOL);
    std::printf(" Viscosity   =%12.7f\n", visc);
    std::printf(" dx/Kol.     =%12.7f\n", dxKol);
    std::printf(" 2Pi/Nlamda  =%12.7f\n", NLAM);
    std::printf(" 2Pi/Lux     =%12.7f\n", Lux);
    std::printf(" 2Pi/Luz     =%12.7f\n", Luz);
    std::printf(" 2Pi/Lwx     =%12.7f\n", Lwx);
    std::printf(" 2Pi/Lwz     =%12.7f\n", Lwz);
    std::printf(" Deps.       =%12.7f\n", De);
    std::printf(" Ceps2       =%12.7f\n", Ceps2);
    std::printf(" E1          =%12.7f\n", double(E1));
    std::printf(" E3          =%12.7f\n", double(E3));
    std::printf(" PAO seed    =%12d\n", seed);

    // ------------------------------------------------------------------
    // Reshuffle (Fortran DO 1000 block)
    // ------------------------------------------------------------------
    for (int comp=0; comp<2; ++comp)
    {
        for (int z=NED2-1; z>=0; --z)
        {
            for (int x=0; x<ND2; ++x)
            {
                // UR(X,N-NED2+Z,I) = UR(X,Z+NED2,I)
                URat(x, N - NED2 + z, comp) = URat(x, z + NED2, comp);

                // IF(Z.LE.(N-NE)) UR(X,Z+NED2,I) = NOLL
                if (z <= (N - NE - 1))
                    URat(x, z + NED2, comp) = std::complex<float>(0.0f,0.0f);
            }
        }
    }

    // ------------------------------------------------------------
    // Build full 3/2-grid UC_full on device.  Large cases cannot afford
    // a dense host-side UC_full staging array.
    // ------------------------------------------------------------
    const int NK_full = S->NK_full;
    const int NZ_full = S->NZ_full;
    const size_t full_count = (size_t)NK_full * (size_t)NZ_full * 3;

    cudaMemset(S->d_uc_full, 0, full_count * sizeof(cplx));

    cplx *d_pao_uc = nullptr;
    const size_t compact_bytes = UR.size() * sizeof(std::complex<float>);
    cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&d_pao_uc), compact_bytes);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "PAO: cudaMalloc(d_pao_uc, %.3f GiB) failed: %s\n",
                     compact_bytes / (1024.0 * 1024.0 * 1024.0),
                     cudaGetErrorString(err));
        return false;
    }

    err = cudaMemcpy(d_pao_uc,
                     reinterpret_cast<const cplx*>(UR.data()),
                     compact_bytes,
                     cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "PAO: cudaMemcpy(d_pao_uc) failed: %s\n",
                     cudaGetErrorString(err));
        cudaFree(d_pao_uc);
        return false;
    }

    {
        dim3 block(16, 16);
        dim3 grid((ND2 + block.x - 1) / block.x,
                  (NE  + block.y - 1) / block.y);
        k_scatter_pao_uc_to_full<<<grid, block>>>(d_pao_uc,
                                                  S->d_uc_full,
                                                  ND2,
                                                  NE,
                                                  NK_full,
                                                  NZ_full);
    }
    cudaFree(d_pao_uc);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "PAO: k_scatter_pao_uc_to_full launch failed: %s\n",
                     cudaGetErrorString(err));
        return false;
    }

    printf(" PAO-kminus3 initialization OK. VISC=%12.7f\n", S->visc);
    return true;
}
