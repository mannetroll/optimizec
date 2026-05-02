// mem_cuda.cu
// ---------------------------------------------------------------------
// Standalone CUDA DNS benchmark driver (Option A: Fortran-style physics)
// Mirrors dns_fps.f by default:
//   N = 2048
//   Re = 100000
//   K0 = 100
//   STEPS = 1000000
//   Loop: STEP2B -> STEP3 -> STEP2A -> NEXTDT
//   Timing + FPS, final diagnostics.
// ---------------------------------------------------------------------

#include "cuda_dns.h"
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cerrno>
#include <csignal>
#include <cmath>
#include <cstdint>
#include <cctype>
#include <cstring>
#include <ctime>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#include <sys/types.h>
#endif

// next_dt_gpu / PGM dump come from dns_cuda.cu
extern void next_dt_gpu(DnsDeviceState *S);
extern void dnsCudaDumpFieldAsPGM(DnsDeviceState* S, int comp, const char* filename);

static volatile std::sig_atomic_t g_exit_requested = 0;
static volatile std::sig_atomic_t g_exit_signal = 0;
static volatile std::sig_atomic_t g_save_requested = 0;
static volatile std::sig_atomic_t g_save_restart_requested = 0;
static volatile std::sig_atomic_t g_pause_requested = 0;
static volatile std::sig_atomic_t g_resume_requested = 0;

static void requestExit(int sig)
{
    g_exit_requested = 1;
    g_exit_signal = sig;
}

static void requestSave(int)
{
    g_save_requested = 1;
}

static void requestSaveRestart(int)
{
    g_save_restart_requested = 1;
}

static void requestPause(int)
{
    g_pause_requested = 1;
}

static void requestResume(int)
{
    g_resume_requested = 1;
}

static void installExitSignalHandlers()
{
#ifndef _WIN32
    std::signal(SIGHUP, requestExit);
    std::signal(SIGUSR1, requestSave);
    std::signal(SIGUSR2, requestPause);
    std::signal(SIGCONT, requestResume);
#ifdef SIGRTMIN
    std::signal(SIGRTMIN, requestSaveRestart);
#endif
#endif
    std::signal(SIGINT, requestExit);
    std::signal(SIGTERM, requestExit);
}

static const char* signalName(int sig)
{
#ifndef _WIN32
#ifdef SIGRTMIN
    if (sig == SIGRTMIN) return "SIGRTMIN";
#endif
#endif
    switch (sig) {
#ifndef _WIN32
        case SIGHUP: return "SIGHUP";
        case SIGUSR1: return "SIGUSR1";
        case SIGUSR2: return "SIGUSR2";
        case SIGCONT: return "SIGCONT";
#endif
        case SIGINT: return "SIGINT";
        case SIGTERM: return "SIGTERM";
        default: return "signal";
    }
}

static bool signalShouldDump(int sig)
{
#ifndef _WIN32
    if (sig == SIGHUP) {
        return true;
    }
#endif
    return sig == SIGTERM;
}

static void ignoreExitSignalHandlers()
{
#ifndef _WIN32
    std::signal(SIGHUP, SIG_IGN);
    std::signal(SIGUSR1, SIG_IGN);
    std::signal(SIGUSR2, SIG_IGN);
    std::signal(SIGCONT, SIG_IGN);
#ifdef SIGRTMIN
    std::signal(SIGRTMIN, SIG_IGN);
#endif
#endif
    std::signal(SIGINT, SIG_IGN);
    std::signal(SIGTERM, SIG_IGN);
}

static std::string sciNoPlus(real x)
{
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.0E", (double)x);

    std::string s(buf);
    const std::string eplus = "E+";
    const size_t pos = s.find(eplus);
    if (pos != std::string::npos) {
        s.replace(pos, eplus.size(), "E");
    }
    return s;
}

static bool makeDirIfNeeded(const std::string& path)
{
#ifdef _WIN32
    const int rc = _mkdir(path.c_str());
#else
    const int rc = mkdir(path.c_str(), 0775);
#endif

    if (rc == 0 || errno == EEXIST) {
        return true;
    }

    std::fprintf(stderr, "Failed to create output folder %s: %s\n",
                 path.c_str(), std::strerror(errno));
    return false;
}

static std::string pgmPath(const std::string& folder, const char* filename)
{
    return folder + "/" + filename;
}

static std::string joinPath(const std::string& folder, const std::string& name)
{
    if (folder.empty()) {
        return name;
    }
    const char last = folder[folder.size() - 1];
    if (last == '/' || last == '\\') {
        return folder + name;
    }
    return folder + "/" + name;
}

static bool makeDirRecursive(const std::string& path)
{
    if (path.empty()) {
        return false;
    }

    size_t i = 0;
    if (path[0] == '/' || path[0] == '\\') {
        i = 1;
    } else if (path.size() >= 2 && path[1] == ':') {
        i = 2;
    }

    for (; i <= path.size(); ++i) {
        if (i != path.size() && path[i] != '/' && path[i] != '\\') {
            continue;
        }

        const std::string part = path.substr(0, i);
        if (part.empty() || part == "/" || part == "\\" ||
            (part.size() == 2 && part[1] == ':')) {
            continue;
        }
        if (!makeDirIfNeeded(part)) {
            return false;
        }
    }
    return true;
}

static bool stringEqualsIgnoreCase(const char* a, const char* b)
{
    if (!a || !b) {
        return false;
    }
    while (*a && *b) {
        const unsigned char ca = (unsigned char)*a;
        const unsigned char cb = (unsigned char)*b;
        if (std::toupper(ca) != std::toupper(cb)) {
            return false;
        }
        ++a;
        ++b;
    }
    return *a == '\0' && *b == '\0';
}

static bool parseEnabledFlag(const char* raw)
{
    if (!raw || raw[0] == '\0') {
        return false;
    }
    if (stringEqualsIgnoreCase(raw, "true") ||
        stringEqualsIgnoreCase(raw, "yes") ||
        stringEqualsIgnoreCase(raw, "on")) {
        return true;
    }
    return std::atoi(raw) != 0;
}

static bool parseMovArg(int argc, char** argv, int first_optional_arg)
{
    const char* raw = std::getenv("MOV");
    for (int i = first_optional_arg; i < argc; ++i) {
        if (stringEqualsIgnoreCase(argv[i], "MOV")) {
            const char* env = std::getenv("MOV");
            raw = (env && env[0] != '\0') ? env : "1";
        }
    }
    return parseEnabledFlag(raw);
}

static std::string compactFloat(real x)
{
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.3f", (double)x);
    return buf;
}

static std::string makeMovieFolderName(int N,
                                       real K0,
                                       real Re,
                                       real CFL,
                                       int update_interval)
{
    const char* root_env = std::getenv("MOV_DIR");
    const std::string root = (root_env && root_env[0] != '\0')
                             ? root_env
                             : "/home/tobbe/movies";
    const std::string suffix = "cuda_" +
                               std::to_string(N) + "_" +
                               std::to_string((int)K0) + "_" +
                               sciNoPlus(Re) + "_" +
                               compactFloat(CFL) + "_U" +
                               std::to_string(update_interval);
    return joinPath(root, suffix);
}

static std::string makeMovieFramePath(const std::string& folder, int frame_index)
{
    char name[64];
    std::snprintf(name, sizeof(name), "omega_inferno_%04d.png", frame_index);
    return joinPath(folder, name);
}

static std::vector<std::string> splitCsvLine(const std::string& line)
{
    std::vector<std::string> cols;
    size_t begin = 0;
    while (begin <= line.size()) {
        const size_t comma = line.find(',', begin);
        if (comma == std::string::npos) {
            cols.push_back(line.substr(begin));
            break;
        }
        cols.push_back(line.substr(begin, comma - begin));
        begin = comma + 1;
    }
    return cols;
}

static bool readLine(FILE* fp, std::string& line)
{
    line.clear();
    char buf[4096];
    if (!std::fgets(buf, sizeof(buf), fp)) {
        return false;
    }
    line = buf;
    while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) {
        line.pop_back();
    }
    return true;
}

struct RestartFolderArgs
{
    int N = 0;
    real Re = 0;
    real K0 = 0;
    real CFL = 0;
    int UPDATE = 100;
    std::string restart_path;
};

static bool loadRestartFolderArgs(const char* folder, RestartFolderArgs& args)
{
    if (!folder || folder[0] == '\0') {
        std::fprintf(stderr, "RESTART requires an output folder\n");
        return false;
    }

    const std::string folder_path = folder;
    const std::string meta_path = pgmPath(folder_path, "meta.csv");
    FILE* fp = std::fopen(meta_path.c_str(), "r");
    if (!fp) {
        std::perror("loadRestartFolderArgs: fopen meta.csv");
        return false;
    }

    std::string header;
    std::string row;
    const bool ok_read = readLine(fp, header) && readLine(fp, row);
    std::fclose(fp);
    if (!ok_read) {
        std::fprintf(stderr, "loadRestartFolderArgs: invalid %s\n", meta_path.c_str());
        return false;
    }

    const std::vector<std::string> cols = splitCsvLine(row);
    if (cols.size() < 9) {
        std::fprintf(stderr, "loadRestartFolderArgs: expected at least 9 columns in %s\n",
                     meta_path.c_str());
        return false;
    }

    args.N = std::atoi(cols[1].c_str());
    args.K0 = (real)std::atof(cols[2].c_str());
    args.Re = (real)std::atof(cols[3].c_str());
    args.CFL = (real)std::atof(cols[4].c_str());
    args.UPDATE = std::atoi(cols[8].c_str());
    if (args.N <= 0 || args.K0 <= 0 || args.Re <= 0 || args.CFL <= 0 || args.UPDATE <= 0) {
        std::fprintf(stderr, "loadRestartFolderArgs: invalid run parameters in %s\n",
                     meta_path.c_str());
        return false;
    }

    args.restart_path = pgmPath(folder_path, "restart.bin");
    FILE* restart_fp = std::fopen(args.restart_path.c_str(), "rb");
    if (!restart_fp) {
        std::perror("loadRestartFolderArgs: fopen restart.bin");
        return false;
    }
    std::fclose(restart_fp);
    return true;
}

static std::string makeOutputFolderName(const DnsDeviceState* S,
                                        int N,
                                        real K0,
                                        real Re,
                                        real CFL)
{
    char cfl_buf[64];
    std::snprintf(cfl_buf, sizeof(cfl_buf), "%.2f", (double)CFL);
    const std::string suffix = std::to_string(N) + "_" +
                               std::to_string((int)K0) + "_" +
                               sciNoPlus(Re) + "_" +
                               cfl_buf + "_" +
                               std::to_string(S->it);
    return "output_" + suffix;
}

struct MetricsRow
{
    int N;
    int K0;
    double Re;
    double CFL;
    double VISC;
    int STEPS;
    int PALIN;
    int SIG;
    double TIME;
    double MINUTES;
    double FPS;
};

struct AdaptViscState
{
    double e_int = 0.0;
    double e_prev = 0.0;
    double target = 20.0;
    double palin_filt = 0.0;
    double palin_filt_1 = 0.0;
};

static double reFromNK0(int N, double K0)
{
    const double a = 1.066240;
    const double b = -0.083911;
    const double c = 1.286396;
    const double log10Re = a * std::log10((double)N) + b * std::log10(K0) + c;
    return std::pow(10.0, log10Re);
}

static double palinFilter2nd(AdaptViscState& A, double p_raw)
{
    const double palin_tau = 5.0;
    const double alpha = 1.0 / (palin_tau + 1.0);

    if (A.palin_filt == 0.0) {
        A.palin_filt_1 = p_raw;
        A.palin_filt = p_raw;
    } else {
        A.palin_filt_1 += alpha * (p_raw - A.palin_filt_1);
        A.palin_filt += alpha * (A.palin_filt_1 - A.palin_filt);
    }

    return A.palin_filt;
}

static void adaptVisc(DnsDeviceState* S, AdaptViscState& A, double pal_over_ens_kmax2)
{
    if (!S) return;

    const double deadband = 0.001;
    const double max_frac = 0.01;
    const double Kp = 0.25;
    const double Ki = 0.0;
    const double Kd = 25.0;

    const double p_raw = 10000.0 * pal_over_ens_kmax2;
    const double p = palinFilter2nd(A, p_raw);
    const double e = (p - A.target) / A.target;

    if (std::fabs(e) < deadband) {
        return;
    }

    if (Ki != 0.0) {
        A.e_int += e;
    }
    const double de = e - A.e_prev;
    A.e_prev = e;

    double u = Kp * e + Ki * A.e_int + Kd * de;
    u = std::max(-max_frac, std::min(max_frac, u));

    const double nu = (double)S->visc;
    const double nu_new = nu * std::exp(u);

    const double Re0 = reFromNK0(S->Nbase, (double)S->K0);
    const double factor = 1000.0;
    const double Re_eff = std::max(Re0 / factor,
                                   std::min(factor * Re0, 1.0 / nu_new));
    S->Re = (real)Re_eff;
    S->visc = (real)(1.0 / Re_eff);
}

static MetricsRow makeMetricsRow(const DnsDeviceState* S,
                                 double pal_over_ens_kmax2,
                                 double sigma,
                                 double elapsed_seconds,
                                 int start_iter)
{
    const int steps = S->it - start_iter;
    const double fps = (elapsed_seconds > 0.0) ? (double)steps / elapsed_seconds : 0.0;
    return MetricsRow{
        S->Nbase,
        (int)S->K0,
        (double)S->Re,
        (double)S->cflnum,
        (double)S->visc,
        S->it,
        (int)(10000.0 * pal_over_ens_kmax2),
        (int)sigma,
        (double)S->t,
        elapsed_seconds / 60.0,
        fps,
    };
}

static bool saveMetricsCsv(const std::vector<MetricsRow>& rows, const char* filename)
{
    FILE* fp = std::fopen(filename, "w");
    if (!fp) {
        std::perror("saveMetricsCsv: fopen");
        return false;
    }

    std::fprintf(fp, "N,K0,Re,CFL,VISC,STEPS,PALIN,SIG,TIME,MINUTES,FPS\n");
    for (const MetricsRow& r : rows) {
        std::fprintf(fp,
                     "%d,%d,%.17g,%.17g,%.17g,%d,%d,%d,%.17g,%.17g,%.17g\n",
                     r.N, r.K0, r.Re, r.CFL, r.VISC, r.STEPS,
                     r.PALIN, r.SIG, r.TIME, r.MINUTES, r.FPS);
    }

    std::fclose(fp);
    std::printf("[CSV] Wrote %s (metrics, %zu rows)\n", filename, rows.size());
    return true;
}

static bool saveEnergySpectrumUvCsv(DnsDeviceState* S, const char* filename)
{
    if (!S) return false;

    vfft_full_forward_ur_full_to_uc_full(S);
    cudaDeviceSynchronize();

    const int nx = S->NX_full;
    const int nz = S->NZ_full;
    const int nk = S->NK_full;
    const size_t plane = (size_t)nk * (size_t)nz;
    const int nbins = std::max(32, 2 * std::min(nx, nz));
    const double r_max = std::sqrt(2.0);
    const double k_nyq = 0.5 * (double)std::min(nx, nz);

    std::vector<cplx> h_uc_full((size_t)plane * 3);
    cudaError_t err = cudaMemcpy(h_uc_full.data(),
                                 S->d_uc_full,
                                 h_uc_full.size() * sizeof(cplx),
                                 cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "saveEnergySpectrumUvCsv: cudaMemcpy failed: %s\n",
                     cudaGetErrorString(err));
        return false;
    }

    std::vector<double> esum(nbins, 0.0);
    std::vector<int> cnt(nbins, 0);

    for (int z = 0; z < nz; ++z) {
        const int kz = (z <= nz / 2) ? z : z - nz;

        for (int kx = 0; kx < nk; ++kx) {
            if (kx == 0 && kz == 0) {
                continue;
            }

            const double r = std::sqrt((double)(kx * kx + kz * kz)) / k_nyq;
            int bin = (int)std::floor((r / r_max) * nbins);
            if (bin < 0) bin = 0;
            if (bin >= nbins) bin = nbins - 1;

            const cplx u = h_uc_full[UC_FULL_INDEX(kx, z, 0, nk, nz)];
            const cplx v = h_uc_full[UC_FULL_INDEX(kx, z, 1, nk, nz)];
            double p = (double)u.x * u.x + (double)u.y * u.y +
                       (double)v.x * v.x + (double)v.y * v.y;

            if (kx > 0 && kx < nx / 2) {
                p *= 2.0;
            }

            esum[bin] += p;
            cnt[bin] += (kx > 0 && kx < nx / 2) ? 2 : 1;
        }
    }

    FILE* fp = std::fopen(filename, "w");
    if (!fp) {
        std::perror("saveEnergySpectrumUvCsv: fopen");
        return false;
    }

    std::fprintf(fp, "normalized_radius,shell_sum_energy,count\n");
    for (int i = 0; i < nbins; ++i) {
        if (cnt[i] > 0 && esum[i] > 0.0) {
            const double r0 = r_max * (double)i / (double)nbins;
            const double r1 = r_max * (double)(i + 1) / (double)nbins;
            std::fprintf(fp, "%.17g,%.17g,%d\n", 0.5 * (r0 + r1), esum[i], cnt[i]);
        }
    }

    std::fclose(fp);
    std::printf("[CSV] Wrote %s (energy spectrum, %d bins)\n", filename, nbins);
    return true;
}

static double computeDisplaySigma(DnsDeviceState* S, int comp)
{
    if (!S) return 0.0;

    const int nx = S->NX_full;
    const int nz = S->NZ_full;
    const size_t plane = (size_t)nx * (size_t)nz;
    std::vector<real> h(plane);

    cudaError_t err = cudaMemcpy(h.data(),
                                 S->d_ur_full + (size_t)plane * comp,
                                 h.size() * sizeof(real),
                                 cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "computeDisplaySigma: cudaMemcpy failed: %s\n",
                     cudaGetErrorString(err));
        return 0.0;
    }

    float minv = h[0];
    float maxv = h[0];
    for (float v : h) {
        minv = std::min(minv, v);
        maxv = std::max(maxv, v);
    }

    const float rng = maxv - minv;
    double sum = 0.0;
    double sum2 = 0.0;

    if (std::fabs(rng) <= 1.0e-12f) {
        sum = 128.0 * plane;
        sum2 = 128.0 * 128.0 * plane;
    } else {
        for (float v : h) {
            float pixf = 1.0f + ((v - minv) / rng) * 254.0f;
            if (pixf < 1.0f) pixf = 1.0f;
            if (pixf > 255.0f) pixf = 255.0f;
            const int pix = (int)pixf;
            sum += pix;
            sum2 += (double)pix * pix;
        }
    }

    const double mean = sum / (double)plane;
    const double var = std::max(0.0, sum2 / (double)plane - mean * mean);
    return std::sqrt(var);
}

static double computePalOverEnsKmax2(DnsDeviceState* S)
{
    if (!S) return 0.0;

    const int nx_half = S->NX / 2;
    const int nz = S->NZ;
    const size_t n = (size_t)nx_half * nz;

    std::vector<cplx> om2(n);
    std::vector<real> alfa(nx_half);
    std::vector<real> gamma(nz);

    cudaError_t err = cudaMemcpy(om2.data(), S->d_om2, n * sizeof(cplx),
                                 cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "computePalOverEnsKmax2: om2 copy failed: %s\n",
                     cudaGetErrorString(err));
        return 0.0;
    }

    err = cudaMemcpy(alfa.data(), S->d_alfa, alfa.size() * sizeof(real),
                     cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "computePalOverEnsKmax2: alfa copy failed: %s\n",
                     cudaGetErrorString(err));
        return 0.0;
    }

    err = cudaMemcpy(gamma.data(), S->d_gamma, gamma.size() * sizeof(real),
                     cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "computePalOverEnsKmax2: gamma copy failed: %s\n",
                     cudaGetErrorString(err));
        return 0.0;
    }

    double total_full = 0.0;
    double pal = 0.0;
    double kmax2 = 0.0;
    double k00 = 0.0;

    for (int z = 0; z < nz; ++z) {
        for (int x = 0; x < nx_half; ++x) {
            const size_t idx = (size_t)x + (size_t)nx_half * z;
            double w = 2.0;
            if (x == 0 || x == nx_half - 1) {
                w = 1.0;
            }

            const double ax = (double)alfa[x];
            const double gz = (double)gamma[z];
            const double k2 = ax * ax + gz * gz;
            const double p = (double)om2[idx].x * om2[idx].x +
                             (double)om2[idx].y * om2[idx].y;
            total_full += w * p;
            pal += w * k2 * p;
            kmax2 = std::max(kmax2, k2);
            if (x == 0 && z == 0) {
                k00 = p;
            }
        }
    }

    const double total = total_full - k00;
    if (total <= 0.0 || kmax2 <= 0.0) {
        return 0.0;
    }
    return pal / (total * kmax2);
}

static void currentTimestamp(char* buf, size_t n)
{
    std::time_t now = std::time(nullptr);
    std::tm tm_now{};
#ifdef _WIN32
    localtime_s(&tm_now, &now);
#else
    localtime_r(&now, &tm_now);
#endif
    std::strftime(buf, n, "%Y-%m-%d %H:%M", &tm_now);
}

static bool saveMetaCsv(const DnsDeviceState* S,
                        const char* filename,
                        int update_interval,
                        double sigma,
                        double pal_over_ens_kmax2,
                        double minutes,
                        double fps)
{
    if (!S) return false;

    FILE* fp = std::fopen(filename, "w");
    if (!fp) {
        std::perror("saveMetaCsv: fopen");
        return false;
    }

    char timestamp[32];
    currentTimestamp(timestamp, sizeof(timestamp));

    std::fprintf(fp,
                 "timestamp,N,K0,Re,CFL,visc,T,IT,Update,sigma,10K_pal_over_Zkmax2,minutes,FPS,backend\n");
    std::fprintf(fp,
                 "%s,%d,%.9g,%.9g,%.9g,%.9g,%.9g,%d,%d,%d,%.9g,%.9g,%.9g,CUDA\n",
                 timestamp,
                 S->Nbase,
                 (double)S->K0,
                 (double)S->Re,
                 (double)S->cflnum,
                 (double)S->visc,
                 (double)S->t,
                 S->it,
                 update_interval,
                 (int)sigma,
                 10000.0 * pal_over_ens_kmax2,
                 minutes,
                 fps);

    std::fclose(fp);
    std::printf("[CSV] Wrote %s (metadata)\n", filename);
    return true;
}

static void appendMetricsRow(DnsDeviceState* S,
                             std::vector<MetricsRow>& csv_rows,
                             double pal_over_ens_kmax2,
                             double omega_sigma,
                             double elapsed_seconds,
                             int start_iter)
{
    if (!csv_rows.empty() && csv_rows.back().STEPS == S->it) {
        csv_rows.back() = makeMetricsRow(S,
                                         pal_over_ens_kmax2,
                                         omega_sigma,
                                         elapsed_seconds,
                                         start_iter);
        return;
    }

    csv_rows.push_back(makeMetricsRow(S,
                                      pal_over_ens_kmax2,
                                      omega_sigma,
                                      elapsed_seconds,
                                      start_iter));
}

struct RestartHeader
{
    char magic[16];
    std::uint32_t version;
    std::uint32_t header_bytes;
    std::uint32_t real_bytes;
    std::uint32_t cplx_bytes;
    std::int32_t Nbase;
    std::int32_t N;
    std::int32_t NX;
    std::int32_t NZ;
    std::int32_t NK;
    std::int32_t NX_full;
    std::int32_t NZ_full;
    std::int32_t NK_full;
    std::int32_t it;
    std::int32_t ifn;
    real Re;
    real K0;
    real visc;
    real t;
    real dt;
    real cn;
    real cnm1;
    real cflnum;
    real cflm;
    std::uint64_t uc_full_count;
    std::uint64_t om2_count;
    std::uint64_t fnm1_count;
};

static bool writeAll(FILE* fp, const void* data, size_t bytes, const char* label)
{
    if (bytes == 0) {
        return true;
    }
    if (std::fwrite(data, 1, bytes, fp) != bytes) {
        std::fprintf(stderr, "writeAll: failed while writing %s: %s\n",
                     label, std::strerror(errno));
        return false;
    }
    return true;
}

static bool writeDeviceBytes(FILE* fp, const void* d_ptr, size_t bytes, const char* label)
{
    if (bytes == 0) {
        return true;
    }

    const size_t chunk_bytes = 256u * 1024u * 1024u;
    std::vector<unsigned char> chunk(std::min(chunk_bytes, bytes));
    const unsigned char* src = static_cast<const unsigned char*>(d_ptr);
    size_t offset = 0;

    while (offset < bytes) {
        const size_t n = std::min(chunk.size(), bytes - offset);
        cudaError_t err = cudaMemcpy(chunk.data(), src + offset, n, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            std::fprintf(stderr, "writeDeviceBytes: cudaMemcpy failed for %s: %s\n",
                         label, cudaGetErrorString(err));
            return false;
        }
        if (!writeAll(fp, chunk.data(), n, label)) {
            return false;
        }
        offset += n;
    }

    return true;
}

static bool readAll(FILE* fp, void* data, size_t bytes, const char* label)
{
    if (bytes == 0) {
        return true;
    }
    if (std::fread(data, 1, bytes, fp) != bytes) {
        std::fprintf(stderr, "readAll: failed while reading %s\n", label);
        return false;
    }
    return true;
}

static bool readDeviceBytes(FILE* fp, void* d_ptr, size_t bytes, const char* label)
{
    if (bytes == 0) {
        return true;
    }

    const size_t chunk_bytes = 256u * 1024u * 1024u;
    std::vector<unsigned char> chunk(std::min(chunk_bytes, bytes));
    unsigned char* dst = static_cast<unsigned char*>(d_ptr);
    size_t offset = 0;

    while (offset < bytes) {
        const size_t n = std::min(chunk.size(), bytes - offset);
        if (!readAll(fp, chunk.data(), n, label)) {
            return false;
        }
        cudaError_t err = cudaMemcpy(dst + offset, chunk.data(), n, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            std::fprintf(stderr, "readDeviceBytes: cudaMemcpy failed for %s: %s\n",
                         label, cudaGetErrorString(err));
            return false;
        }
        offset += n;
    }

    return true;
}

static bool saveRestartBin(DnsDeviceState* S, const std::string& folder)
{
    if (!S) return false;

    const std::string path = pgmPath(folder, "restart.bin");
    FILE* fp = std::fopen(path.c_str(), "wb");
    if (!fp) {
        std::perror("saveRestartBin: fopen");
        return false;
    }

    RestartHeader h{};
    std::snprintf(h.magic, sizeof(h.magic), "CTRB_RESTART");
    h.version = 1;
    h.header_bytes = (std::uint32_t)sizeof(RestartHeader);
    h.real_bytes = (std::uint32_t)sizeof(real);
    h.cplx_bytes = (std::uint32_t)sizeof(cplx);
    h.Nbase = S->Nbase;
    h.N = S->N;
    h.NX = S->NX;
    h.NZ = S->NZ;
    h.NK = S->NK;
    h.NX_full = S->NX_full;
    h.NZ_full = S->NZ_full;
    h.NK_full = S->NK_full;
    h.it = S->it;
    h.ifn = S->ifn;
    h.Re = S->Re;
    h.K0 = S->K0;
    h.visc = S->visc;
    h.t = S->t;
    h.dt = S->dt;
    h.cn = S->cn;
    h.cnm1 = S->cnm1;
    h.cflnum = S->cflnum;
    h.cflm = S->cflm;
    h.uc_full_count = (std::uint64_t)S->NK_full * (std::uint64_t)S->NZ_full * 3u;
    h.om2_count = (std::uint64_t)(S->NX / 2) * (std::uint64_t)S->NZ;
    h.fnm1_count = h.om2_count;

    bool ok = writeAll(fp, &h, sizeof(h), "restart header") &&
              writeDeviceBytes(fp, S->d_uc_full, (size_t)h.uc_full_count * sizeof(cplx), "d_uc_full") &&
              writeDeviceBytes(fp, S->d_om2, (size_t)h.om2_count * sizeof(cplx), "d_om2") &&
              writeDeviceBytes(fp, S->d_fnm1, (size_t)h.fnm1_count * sizeof(cplx), "d_fnm1");

    if (std::fclose(fp) != 0) {
        std::perror("saveRestartBin: fclose");
        ok = false;
    }

    if (ok) {
        std::printf("[SAVE] Wrote %s (restart binary)\n", path.c_str());
    }
    return ok;
}

static bool loadRestartBin(DnsDeviceState* S, const char* path)
{
    if (!S || !path) return false;

    FILE* fp = std::fopen(path, "rb");
    if (!fp) {
        std::perror("loadRestartBin: fopen");
        return false;
    }

    RestartHeader h{};
    bool ok = readAll(fp, &h, sizeof(h), "restart header");
    if (ok && std::strncmp(h.magic, "CTRB_RESTART", 12) != 0) {
        std::fprintf(stderr, "loadRestartBin: bad magic in %s\n", path);
        ok = false;
    }
    if (ok && (h.version != 1 ||
               h.header_bytes != sizeof(RestartHeader) ||
               h.real_bytes != sizeof(real) ||
               h.cplx_bytes != sizeof(cplx))) {
        std::fprintf(stderr, "loadRestartBin: unsupported restart format in %s\n", path);
        ok = false;
    }
    if (ok && (h.Nbase != S->Nbase ||
               h.N != S->N ||
               h.NX != S->NX ||
               h.NZ != S->NZ ||
               h.NK != S->NK ||
               h.NX_full != S->NX_full ||
               h.NZ_full != S->NZ_full ||
               h.NK_full != S->NK_full)) {
        std::fprintf(stderr, "loadRestartBin: restart dimensions do not match this run\n");
        ok = false;
    }

    const std::uint64_t uc_full_count = (std::uint64_t)S->NK_full * (std::uint64_t)S->NZ_full * 3u;
    const std::uint64_t om_count = (std::uint64_t)(S->NX / 2) * (std::uint64_t)S->NZ;
    if (ok && (h.uc_full_count != uc_full_count ||
               h.om2_count != om_count ||
               h.fnm1_count != om_count)) {
        std::fprintf(stderr, "loadRestartBin: restart array sizes do not match this run\n");
        ok = false;
    }

    if (ok) {
        S->Re = h.Re;
        S->K0 = h.K0;
        S->visc = h.visc;
        S->t = h.t;
        S->dt = h.dt;
        S->cn = h.cn;
        S->cnm1 = h.cnm1;
        S->cflnum = h.cflnum;
        S->cflm = h.cflm;
        S->it = h.it;
        S->ifn = h.ifn;

        ok = readDeviceBytes(fp, S->d_uc_full, (size_t)h.uc_full_count * sizeof(cplx), "d_uc_full") &&
             readDeviceBytes(fp, S->d_om2, (size_t)h.om2_count * sizeof(cplx), "d_om2") &&
             readDeviceBytes(fp, S->d_fnm1, (size_t)h.fnm1_count * sizeof(cplx), "d_fnm1");
    }

    if (std::fclose(fp) != 0) {
        std::perror("loadRestartBin: fclose");
        ok = false;
    }

    if (ok) {
        std::printf("[LOAD] Loaded restart binary %s at iteration %d\n", path, S->it);
    }
    return ok;
}

static bool dumpRunOutputsToFolder(DnsDeviceState* S,
                                   const std::vector<MetricsRow>& csv_rows,
                                   const std::string& folder,
                                   int update_interval,
                                   double elapsed_seconds,
                                   double fps)
{
    if (!makeDirIfNeeded(folder)) {
        return false;
    }

    std::printf("[SAVE] Dumping fields to folder: %s\n", folder.c_str());

    const std::string u_path = pgmPath(folder, "u_velocity.pgm");
    dnsCudaDumpFieldAsPGMFull(S, 0, u_path.c_str());
    const std::string v_path = pgmPath(folder, "v_velocity.pgm");
    dnsCudaDumpFieldAsPGMFull(S, 1, v_path.c_str());
    dnsCudaKinetic(S);
    const std::string kinetic_path = pgmPath(folder, "kinetic.pgm");
    dnsCudaDumpFieldAsPGMFull(S, 2, kinetic_path.c_str());
    dnsCudaOm2Phys(S);
    const std::string omega_path = pgmPath(folder, "omega.pgm");
    dnsCudaDumpFieldAsPGMFullClipped(S, 2, omega_path.c_str(), 0.001, 0.999);
    const double omega_sigma = csv_rows.empty() ? computeDisplaySigma(S, 2)
                                                : (double)csv_rows.back().SIG;
    const double pal_over_ens_kmax2 = csv_rows.empty() ? computePalOverEnsKmax2(S)
                                                       : ((double)csv_rows.back().PALIN / 10000.0);
    dnsCudaStreamFunc(S);
    const std::string stream_path = pgmPath(folder, "stream.pgm");
    dnsCudaDumpFieldAsPGMFull(S, 2, stream_path.c_str());
    const std::string spectrum_path = pgmPath(folder, "energy_spectrum.csv");
    saveEnergySpectrumUvCsv(S, spectrum_path.c_str());
    const std::string metrics_path = pgmPath(folder, "metrics.csv");
    saveMetricsCsv(csv_rows, metrics_path.c_str());
    const std::string meta_path = pgmPath(folder, "meta.csv");
    saveMetaCsv(S, meta_path.c_str(), update_interval, omega_sigma, pal_over_ens_kmax2,
                elapsed_seconds / 60.0, fps);

    return true;
}

static bool dumpRunOutputs(DnsDeviceState* S,
                           const std::vector<MetricsRow>& csv_rows,
                           int N,
                           real K0,
                           real Re,
                           real CFL,
                           int update_interval,
                           double elapsed_seconds,
                           double fps)
{
    const std::string folder = makeOutputFolderName(S, N, K0, Re, CFL);
    return dumpRunOutputsToFolder(S,
                                  csv_rows,
                                  folder,
                                  update_interval,
                                  elapsed_seconds,
                                  fps);
}

static bool saveSnapshotAndContinue(DnsDeviceState* S,
                                    std::vector<MetricsRow>& csv_rows,
                                    int N,
                                    real K0,
                                    real Re,
                                    real CFL,
                                    int update_interval,
                                    double elapsed_seconds,
                                    double fps,
                                    int start_iter,
                                    bool adapt_visc)
{
    if (!S) return false;

    std::printf("[SIGNAL] Received SIGUSR1; saving current state and continuing.\n");

    dnsCudaOm2Phys(S);
    const double omega_sigma = computeDisplaySigma(S, 2);
    const double pal_over_ens_kmax2 = adapt_visc ? computePalOverEnsKmax2(S) : 0.0;
    appendMetricsRow(S,
                     csv_rows,
                     pal_over_ens_kmax2,
                     omega_sigma,
                     elapsed_seconds,
                     start_iter);

    return dumpRunOutputs(S,
                          csv_rows,
                          N,
                          K0,
                          Re,
                          CFL,
                          update_interval,
                          elapsed_seconds,
                          fps);
}

static bool saveSnapshotRestartAndContinue(DnsDeviceState* S,
                                           std::vector<MetricsRow>& csv_rows,
                                           int N,
                                           real K0,
                                           real Re,
                                           real CFL,
                                           int update_interval,
                                           double elapsed_seconds,
                                           double fps,
                                           int start_iter,
                                           bool adapt_visc)
{
    if (!S) return false;

    std::printf("[SIGNAL] Received SIGRTMIN; saving images and restart binary, then continuing.\n");

    const std::string folder = makeOutputFolderName(S, N, K0, Re, CFL);
    if (!makeDirIfNeeded(folder)) {
        return false;
    }
    if (!saveRestartBin(S, folder)) {
        return false;
    }

    dnsCudaOm2Phys(S);
    const double omega_sigma = computeDisplaySigma(S, 2);
    const double pal_over_ens_kmax2 = adapt_visc ? computePalOverEnsKmax2(S) : 0.0;
    appendMetricsRow(S,
                     csv_rows,
                     pal_over_ens_kmax2,
                     omega_sigma,
                     elapsed_seconds,
                     start_iter);

    return dumpRunOutputsToFolder(S,
                                  csv_rows,
                                  folder,
                                  update_interval,
                                  elapsed_seconds,
                                  fps);
}

static double gpuTakenMiB()
{
    size_t freeBytes  = 0;
    size_t totalBytes = 0;
    if (cudaMemGetInfo(&freeBytes, &totalBytes) != cudaSuccess) {
        return -1.0;
    }
    return (totalBytes - freeBytes) / (1024.0 * 1024.0);
}

#ifndef _WIN32
static bool readableFile(const char* path)
{
    FILE* fp = std::fopen(path, "r");
    if (!fp) {
        return false;
    }
    std::fclose(fp);
    return true;
}

static std::string shellQuote(const char* s)
{
    std::string out = "'";
    for (const char* p = s; p && *p; ++p) {
        if (*p == '\'') {
            out += "'\\''";
        } else {
            out += *p;
        }
    }
    out += "'";
    return out;
}

static std::string nvidiaSmiCommand()
{
    const char* env = std::getenv("NVIDIA_SMI");
    if (env && env[0] != '\0') {
        return shellQuote(env);
    }

    const char* wsl_path = "/usr/lib/wsl/lib/nvidia-smi";
    if (readableFile(wsl_path)) {
        return shellQuote(wsl_path);
    }

    return "nvidia-smi";
}

static void trimLineEnd(std::string& line)
{
    while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) {
        line.pop_back();
    }
}

static bool commandOutputLines(const std::string& command,
                               std::vector<std::string>& lines)
{
    FILE* pipe = popen(command.c_str(), "r");
    if (!pipe) {
        return false;
    }

    char buf[1024];
    while (std::fgets(buf, sizeof(buf), pipe)) {
        std::string line(buf);
        trimLineEnd(line);
        if (!line.empty()) {
            lines.push_back(line);
        }
    }

    const int status = pclose(pipe);
    return status == 0 && !lines.empty();
}

static void printGpuSnapshot(const char* phase)
{
    const std::string smi = nvidiaSmiCommand();
    const std::string gpu_query =
        smi +
        " --query-gpu=name,persistence_mode,pstate,clocks.sm,clocks.mem,"
        "temperature.gpu,power.draw,power.limit "
        "--format=csv,noheader,nounits 2>/dev/null";

    std::vector<std::string> lines;
    if (!commandOutputLines(gpu_query, lines)) {
        std::printf("[GPU] %s: nvidia-smi unavailable\n", phase);
        return;
    }

    std::printf("[GPU] %s fields: name, persistence, pstate, sm_clock_mhz, "
                "mem_clock_mhz, temp_c, power_w, power_limit_w\n",
                phase);
    for (const std::string& line : lines) {
        std::printf("[GPU] %s: %s\n", phase, line.c_str());
    }

    if (std::strcmp(phase, "start") == 0) {
        lines.clear();
        const std::string app_query =
            smi +
            " --query-compute-apps=pid,process_name,used_memory "
            "--format=csv,noheader,nounits 2>/dev/null";
        if (commandOutputLines(app_query, lines)) {
            std::printf("[GPU] %s active compute apps: pid, process, used_memory_mib\n",
                        phase);
            for (const std::string& line : lines) {
                std::printf("[GPU] %s app: %s\n", phase, line.c_str());
            }
        }
    }
}
#else
static void printGpuSnapshot(const char*) {}
#endif

// ---------------------------------------------------------------------
// main() – CUDA FPS benchmark like dns_fps.f, but with CLI args
// Usage:
//   ./dns_cuda_fps                # N=1024 Re=1e5 K0=100 STEPS=1001
//   ./dns_cuda_fps 1024          # N=1024 Re=1e5 K0=100 STEPS=1001
//   ./dns_cuda_fps 2048 100000 10 50000 0.25 100
//   ./dns_cuda_fps 2048 100000 10 50000 0.25 100 0  # disable adapt_visc
//   ./dns_cuda_fps 2048 100000 10 50000 0.25 100 0 output_.../restart.bin
//   ./dns_cuda_fps RESTART output_...
// ---------------------------------------------------------------------
int main(int argc, char** argv)
{
    installExitSignalHandlers();

    // Defaults (dns_fps.f-style)
    int   N      = 512;
    real  Re     = (real)10000.0f;
    real  K0     = (real)10.0f;
    int   STEPS  = 20001;
    real  CFL    = (real)0.25f;
    int   UPDATE = 100;
    bool  ADAPT_VISC = false;
    bool  MOV = false;
    const char* RESTART_PATH = nullptr;
    std::string restart_path_storage;

    if (argc > 1 && std::strcmp(argv[1], "RESTART") == 0) {
        if (argc < 3) {
            std::fprintf(stderr, "Usage: %s RESTART output_folder\n", argv[0]);
            return 2;
        }

        RestartFolderArgs restart_args{};
        if (!loadRestartFolderArgs(argv[2], restart_args)) {
            return 1;
        }

        N = restart_args.N;
        Re = restart_args.Re;
        K0 = restart_args.K0;
        CFL = restart_args.CFL;
        UPDATE = restart_args.UPDATE;
        STEPS = 1000000;
        ADAPT_VISC = false;
        restart_path_storage = restart_args.restart_path;
        RESTART_PATH = restart_path_storage.c_str();

        const char* env_steps = std::getenv("RESTART_STEPS");
        if (env_steps && env_steps[0] != '\0') {
            STEPS = std::atoi(env_steps);
        }
        MOV = parseMovArg(argc, argv, 3);
    } else {
        if (argc > 1) N      = std::atoi(argv[1]);
        if (argc > 2) Re     = (real)std::atof(argv[2]);
        if (argc > 3) K0     = (real)std::atof(argv[3]);
        if (argc > 4) STEPS  = std::atoi(argv[4]);
        if (argc > 5) CFL    = (real)std::atof(argv[5]);
        if (argc > 6) UPDATE = std::atoi(argv[6]);
        if (argc > 7) ADAPT_VISC = (std::atoi(argv[7]) != 0);
        int mov_arg_begin = 8;
        if (argc > 8) {
            if (stringEqualsIgnoreCase(argv[8], "MOV")) {
                mov_arg_begin = 8;
            } else {
                RESTART_PATH = argv[8];
                mov_arg_begin = 9;
            }
        }
        MOV = parseMovArg(argc, argv, mov_arg_begin);
    }
    if (UPDATE <= 0) UPDATE = 100;
    if (STEPS <= 0) STEPS = 1000000;

    printf("--- INITIALIZING FPS_CUDA ---\n");
    printf(" N=%d, Re=%d, K0=%d, STEPS=%d, CFL=%.3f, UPDATE=%d, ADAPT_VISC=%d, MOV=%d\n",
           N, (int)Re, (int)K0, STEPS, (float)CFL, UPDATE,
           ADAPT_VISC ? 1 : 0, MOV ? 1 : 0);
    if (RESTART_PATH) {
        printf(" restart=%s\n", RESTART_PATH);
    }
    printGpuSnapshot("start");

    DnsDeviceState S{};

    double taken_before = gpuTakenMiB();

    if (!dnsCudaCreate(&S, N, Re, K0)) {
        std::printf("dnsCudaCreate failed\n");
        return 1;
    }

    double taken_after = gpuTakenMiB();

    if (taken_before >= 0.0 && taken_after >= 0.0) {
        std::printf(" dnsCudaCreate: GPU memory ≈ %.2f MiB\n",
                    taken_after - taken_before);
    }

    // CFLNUM etc.
    S.cflnum = CFL;
    S.ifn    = 1;
    S.it     = 0;
    S.dt     = 0.0f;
    S.cn     = 1.0f;
    S.cnm1   = 0.0f;

    // PAO + CALCOM etc.
    if (!dnsCudaInit(&S)) {
        std::printf("dnsCudaInit failed\n");
        dnsCudaDestroy(&S);
        return 1;
    }

    if (RESTART_PATH) {
        if (!loadRestartBin(&S, RESTART_PATH)) {
            dnsCudaDestroy(&S);
            return 1;
        }
        N = S.Nbase;
        Re = S.Re;
        K0 = S.K0;
        CFL = S.cflnum;
    } else {
        // Initial STEP2A before the loop (like dns_fps.f)
        dnsCudaStep2A(&S);

        // Initial NEXTDT (IT=0, special branch)
        next_dt_gpu(&S);
        printf(" Initial DT=%12.7f  CN=%12.7f\n", S.dt, S.cn);
    }

    // -----------------------------------------------------------------
    // Timing section: STEP2B → STEP3 → STEP2A → NEXTDT
    // -----------------------------------------------------------------
    using clock_type = std::chrono::steady_clock;
    cudaDeviceSynchronize();  // ensure all previous work done
    auto tbegin = clock_type::now();
    const int update_interval = UPDATE;
    int steady_warmup_steps = update_interval;
    const char* env_warmup_steps = std::getenv("BENCH_WARMUP_STEPS");
    if (env_warmup_steps && env_warmup_steps[0] != '\0') {
        steady_warmup_steps = std::atoi(env_warmup_steps);
        if (steady_warmup_steps < 0) {
            steady_warmup_steps = 0;
        }
    }
    const int start_iter = S.it;
    AdaptViscState adapt_state{};
    std::vector<MetricsRow> csv_rows;
    bool interrupted = false;
    double paused_seconds = 0.0;
    bool steady_timer_started = false;
    int steady_start_iter = S.it;
    double steady_start_paused_seconds = 0.0;
    clock_type::time_point steady_begin{};
    int movie_frame_index = 0;
    const int movie_scale_f = dnsCudaMovieScaleF(N);
    std::string movie_folder;

    std::printf("[BENCH] Wall-clock FPS timer starts after initial cudaDeviceSynchronize().\n");
    if (steady_warmup_steps > 0) {
        std::printf("[BENCH] Steady FPS will exclude the first %d completed steps "
                    "(override with BENCH_WARMUP_STEPS).\n",
                    steady_warmup_steps);
    } else {
        steady_timer_started = true;
        steady_begin = tbegin;
        steady_start_iter = S.it;
        steady_start_paused_seconds = paused_seconds;
        std::printf("[BENCH] Steady FPS timer starts immediately "
                    "(BENCH_WARMUP_STEPS=0).\n");
    }

    if (MOV) {
        movie_folder = makeMovieFolderName(N, K0, Re, CFL, update_interval);
        if (!makeDirRecursive(movie_folder)) {
            dnsCudaDestroy(&S);
            return 1;
        }
        std::printf("[MOV] Saving omega_inferno_%%04d.png every %d steps to %s (F=%d)\n",
                    update_interval, movie_folder.c_str(), movie_scale_f);
    }

    for (int it = S.it + 1; it <= STEPS; ++it) {

        S.it = it;
        float dt_old = S.dt;  // capture before next_dt may update it

        dnsCudaStep2B(&S);
        dnsCudaStep3(&S);
        dnsCudaStep2A(&S);

        S.t += dt_old;  // advance by pre-nextdt dt, matching Python

        if (it == 1 || (it % update_interval) == 0 || it == STEPS) {
            next_dt_gpu(&S);

            dnsCudaOm2Phys(&S);
            const double omega_sigma = computeDisplaySigma(&S, 2);
            double pal_over_ens_kmax2 = 0.0;
            if (ADAPT_VISC) {
                pal_over_ens_kmax2 = computePalOverEnsKmax2(&S);
                adaptVisc(&S, adapt_state, pal_over_ens_kmax2);
            }

            auto tnow = clock_type::now();
            std::chrono::duration<double> row_elapsed = tnow - tbegin;
            const double row_active_seconds = row_elapsed.count() - paused_seconds;
            appendMetricsRow(&S,
                             csv_rows,
                             pal_over_ens_kmax2,
                             omega_sigma,
                             row_active_seconds,
                             start_iter);

            printf(" ITERATION %6d T=%12.10f DT=%10.8f CN=%10.8f CFLM=%.6f Re=%12.5e VISC=%12.5e PALIN=%d SIG=%d FPS=%.3f\n",
                   it,
                   S.t,
                   S.dt,
                   S.cn,
                   compute_cflm(&S),
                   (double)S.Re,
                   (double)S.visc,
                   csv_rows.back().PALIN,
                   csv_rows.back().SIG,
                   csv_rows.back().FPS);

            if (MOV && it > 0 && (it % update_interval) == 0) {
                const std::string frame_path = makeMovieFramePath(movie_folder,
                                                                  movie_frame_index + 1);
                if (!dnsCudaSaveFieldInfernoPng(&S,
                                                2,
                                                frame_path.c_str(),
                                                movie_scale_f)) {
                    dnsCudaDestroy(&S);
                    return 1;
                }
                ++movie_frame_index;
            }

            if (!steady_timer_started && it >= steady_warmup_steps) {
                cudaDeviceSynchronize();
                steady_begin = clock_type::now();
                steady_start_iter = S.it;
                steady_start_paused_seconds = paused_seconds;
                steady_timer_started = true;
                std::printf("[BENCH] Steady FPS timer starts after iteration %d.\n",
                            steady_start_iter);
            }
        }

        if (g_pause_requested) {
            g_pause_requested = 0;
            g_resume_requested = 0;
            cudaDeviceSynchronize();
            auto pause_begin = clock_type::now();
            printf("[SIGNAL] Received SIGUSR2; pausing at iteration %d. Send SIGCONT to resume.\n",
                   S.it);
            std::fflush(stdout);

            while (!g_resume_requested && !g_exit_requested) {
                std::this_thread::sleep_for(std::chrono::milliseconds(250));
            }

            auto pause_end = clock_type::now();
            std::chrono::duration<double> pause_elapsed = pause_end - pause_begin;
            paused_seconds += pause_elapsed.count();

            if (g_resume_requested) {
                g_resume_requested = 0;
                printf("[SIGNAL] Received SIGCONT; resuming after %.3f paused seconds.\n",
                       pause_elapsed.count());
                std::fflush(stdout);
            }
        }

        if (g_exit_requested) {
            interrupted = true;
            break;
        }

        if (g_save_restart_requested) {
            g_save_restart_requested = 0;
            cudaDeviceSynchronize();
            auto tsave = clock_type::now();
            std::chrono::duration<double> save_elapsed = tsave - tbegin;
            const int save_completed_steps = S.it - start_iter;
            const double save_seconds = save_elapsed.count() - paused_seconds;
            const double save_fps = (save_seconds > 0.0)
                                    ? (double(save_completed_steps) / save_seconds)
                                    : 0.0;
            if (!saveSnapshotRestartAndContinue(&S,
                                                csv_rows,
                                                N,
                                                K0,
                                                Re,
                                                CFL,
                                                update_interval,
                                                save_seconds,
                                                save_fps,
                                                start_iter,
                                                ADAPT_VISC)) {
                dnsCudaDestroy(&S);
                return 1;
            }
        }

        if (g_save_requested) {
            g_save_requested = 0;
            cudaDeviceSynchronize();
            auto tsave = clock_type::now();
            std::chrono::duration<double> save_elapsed = tsave - tbegin;
            const int save_completed_steps = S.it - start_iter;
            const double save_seconds = save_elapsed.count() - paused_seconds;
            const double save_fps = (save_seconds > 0.0)
                                    ? (double(save_completed_steps) / save_seconds)
                                    : 0.0;
            if (!saveSnapshotAndContinue(&S,
                                         csv_rows,
                                         N,
                                         K0,
                                         Re,
                                         CFL,
                                         update_interval,
                                         save_seconds,
                                         save_fps,
                                         start_iter,
                                         ADAPT_VISC)) {
                dnsCudaDestroy(&S);
                return 1;
            }
        }
    }

    cudaDeviceSynchronize();
    auto tend = clock_type::now();
    std::chrono::duration<double> elapsed = tend - tbegin;
    double elap = elapsed.count() - paused_seconds;
    const int completed_steps = S.it - start_iter;
    double fps = (elap > 0.0) ? (double(completed_steps) / elap) : 0.0;
    double steady_elap = 0.0;
    double steady_fps = 0.0;
    const int steady_completed_steps = S.it - steady_start_iter;
    if (steady_timer_started && steady_completed_steps > 0) {
        std::chrono::duration<double> steady_elapsed = tend - steady_begin;
        steady_elap = steady_elapsed.count() -
                      (paused_seconds - steady_start_paused_seconds);
        steady_fps = (steady_elap > 0.0)
                     ? (double(steady_completed_steps) / steady_elap)
                     : 0.0;
    }

    printf(" Frames per second (FPS)            = %12.7f\n", fps);
    if (steady_timer_started && steady_completed_steps > 0) {
        printf(" Steady frames per second (FPS)     = %12.7f  "
               "(%d steps after warmup)\n",
               steady_fps,
               steady_completed_steps);
    }

    printf(" Elapsed time (s): %12.7f\n", elap);
    printf(" FPS: %12.7f\n", fps);
    if (steady_timer_started && steady_completed_steps > 0) {
        printf(" Steady elapsed time (s): %12.7f\n", steady_elap);
        printf(" Steady FPS: %12.7f\n", steady_fps);
    }
    printf(" Final T=%12.7f  CN=%12.7f  DT=%12.7f  VISC=%12.7f\n",
           S.t, S.cn, S.dt, S.visc);
    printGpuSnapshot("end");
    std::fflush(stdout);

    if (interrupted) {
        ignoreExitSignalHandlers();
        if (!signalShouldDump((int)g_exit_signal)) {
            printf("[SIGNAL] Received %s (%d); exiting without saving.\n",
                   signalName((int)g_exit_signal),
                   (int)g_exit_signal);
            dnsCudaDestroy(&S);
            return 128 + (int)g_exit_signal;
        }

        printf("[SIGNAL] Received %s (%d); dumping current state before exit.\n",
               signalName((int)g_exit_signal),
               (int)g_exit_signal);

        next_dt_gpu(&S);
        dnsCudaOm2Phys(&S);
        const double omega_sigma = computeDisplaySigma(&S, 2);
        const double pal_over_ens_kmax2 = ADAPT_VISC ? computePalOverEnsKmax2(&S) : 0.0;
        appendMetricsRow(&S,
                         csv_rows,
                         pal_over_ens_kmax2,
                         omega_sigma,
                         elap,
                         start_iter);
    }

    if (!dumpRunOutputs(&S, csv_rows, N, K0, Re, CFL, update_interval, elap, fps)) {
        dnsCudaDestroy(&S);
        return 1;
    }

    dnsCudaDestroy(&S);
    return interrupted ? 128 + (int)g_exit_signal : 0;
}
