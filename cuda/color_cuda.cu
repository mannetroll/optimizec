// color_cuda.cu
// ---------------------------------------------------------------------
// CUDA-side color lookup tables and PNG output helpers for movie frames.
// ---------------------------------------------------------------------

#include "cuda_dns.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {

constexpr double DISPLAY_NORM_K_STD = 2.5;
__constant__ unsigned char d_inferno_lut[256 * 3];

struct PixelStats
{
    double sum;
    double sum2;
};

struct MinMaxBlock
{
    float minv;
    float maxv;
    unsigned long long count;
};

static void makeInfernoLut(unsigned char lut[256 * 3])
{
    struct Stop
    {
        float pos;
        unsigned char r;
        unsigned char g;
        unsigned char b;
    };

    const Stop stops[] = {
        {0.00f,   0,   0,   4},
        {0.25f,  87,  15, 109},
        {0.50f, 187,  55,  84},
        {0.75f, 249, 142,   8},
        {1.00f, 252, 255, 164},
    };

    for (int i = 0; i < 256; ++i) {
        const float x = (float)i / 255.0f;
        int s = 0;
        while (s + 1 < (int)(sizeof(stops) / sizeof(stops[0])) &&
               x > stops[s + 1].pos) {
            ++s;
        }
        const Stop& a = stops[s];
        const Stop& b = stops[std::min(s + 1, (int)(sizeof(stops) / sizeof(stops[0])) - 1)];
        const float span = std::max(1.0e-12f, b.pos - a.pos);
        const float t = std::max(0.0f, std::min(1.0f, (x - a.pos) / span));
        lut[3 * i + 0] = (unsigned char)((1.0f - t) * a.r + t * b.r);
        lut[3 * i + 1] = (unsigned char)((1.0f - t) * a.g + t * b.g);
        lut[3 * i + 2] = (unsigned char)((1.0f - t) * a.b + t * b.b);
    }
}

static bool ensureInfernoLut()
{
    static bool initialized = false;
    if (initialized) {
        return true;
    }

    unsigned char lut[256 * 3];
    makeInfernoLut(lut);
    cudaError_t err = cudaMemcpyToSymbol(d_inferno_lut, lut, sizeof(lut));
    if (err != cudaSuccess) {
        std::fprintf(stderr, "ensureInfernoLut: cudaMemcpyToSymbol failed: %s\n",
                     cudaGetErrorString(err));
        return false;
    }

    initialized = true;
    return true;
}

__global__ void reduceMinMaxKernel(const float* field,
                                   size_t n,
                                   MinMaxBlock* blocks)
{
    __shared__ float s_min[256];
    __shared__ float s_max[256];
    __shared__ unsigned long long s_count[256];

    const int tid = threadIdx.x;
    const size_t stride = (size_t)blockDim.x * (size_t)gridDim.x;
    size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)tid;

    float local_min = FLT_MAX;
    float local_max = -FLT_MAX;
    unsigned long long local_count = 0;

    while (idx < n) {
        const float v = field[idx];
        if (isfinite(v)) {
            local_min = fminf(local_min, v);
            local_max = fmaxf(local_max, v);
            ++local_count;
        }
        idx += stride;
    }

    s_min[tid] = local_min;
    s_max[tid] = local_max;
    s_count[tid] = local_count;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            s_min[tid] = fminf(s_min[tid], s_min[tid + offset]);
            s_max[tid] = fmaxf(s_max[tid], s_max[tid + offset]);
            s_count[tid] += s_count[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        blocks[blockIdx.x] = MinMaxBlock{s_min[0], s_max[0], s_count[0]};
    }
}

__global__ void reducePixelStatsKernel(const float* field,
                                       size_t n,
                                       PixelStats* blocks,
                                       float minv,
                                       float inv_range,
                                       bool is_constant)
{
    __shared__ double s_sum[256];
    __shared__ double s_sum2[256];

    const int tid = threadIdx.x;
    const size_t stride = (size_t)blockDim.x * (size_t)gridDim.x;
    size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)tid;

    double local_sum = 0.0;
    double local_sum2 = 0.0;

    while (idx < n) {
        const float val = field[idx];
        int pix = 128;
        if (!is_constant && isfinite(val)) {
            const float norm = (val - minv) * inv_range;
            float pixf = 1.0f + norm * 254.0f;
            if (pixf < 1.0f) pixf = 1.0f;
            if (pixf > 255.0f) pixf = 255.0f;
            pix = (int)pixf;
        }
        const double p = (double)pix;
        local_sum += p;
        local_sum2 += p * p;
        idx += stride;
    }

    s_sum[tid] = local_sum;
    s_sum2[tid] = local_sum2;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            s_sum[tid] += s_sum[tid + offset];
            s_sum2[tid] += s_sum2[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        blocks[blockIdx.x] = PixelStats{s_sum[0], s_sum2[0]};
    }
}

__global__ void renderInfernoKernel(const float* field,
                                    unsigned char* rgb,
                                    int nx,
                                    int nz,
                                    int out_w,
                                    int out_h,
                                    int scale_f,
                                    float minv,
                                    float inv_range,
                                    bool is_constant,
                                    double mean,
                                    double sigma)
{
    const int ox = blockIdx.x * blockDim.x + threadIdx.x;
    const int oy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ox >= out_w || oy >= out_h) {
        return;
    }

    const int x = min(ox * scale_f, nx - 1);
    const int y = min(oy * scale_f, nz - 1);
    const float val = field[(size_t)y * (size_t)nx + (size_t)x];

    int pix = 128;
    if (!is_constant && isfinite(val)) {
        const float norm = (val - minv) * inv_range;
        float pixf = 1.0f + norm * 254.0f;
        if (pixf < 1.0f) pixf = 1.0f;
        if (pixf > 255.0f) pixf = 255.0f;
        pix = (int)pixf;
    }

    int color_idx = pix;
    const double lo = mean - DISPLAY_NORM_K_STD * sigma;
    const double hi = mean + DISPLAY_NORM_K_STD * sigma;
    if (sigma >= 1.0 && hi > lo) {
        const double mapped = ((double)pix - lo) * 255.0 / (hi - lo);
        color_idx = (int)floor(mapped + 0.5);
        if (color_idx < 0) color_idx = 0;
        if (color_idx > 255) color_idx = 255;
    }

    const size_t dst = 3u * ((size_t)oy * (size_t)out_w + (size_t)ox);
    rgb[dst + 0] = d_inferno_lut[3 * color_idx + 0];
    rgb[dst + 1] = d_inferno_lut[3 * color_idx + 1];
    rgb[dst + 2] = d_inferno_lut[3 * color_idx + 2];
}

static std::uint32_t crc32Update(std::uint32_t crc, const unsigned char* data, size_t n)
{
    crc = ~crc;
    for (size_t i = 0; i < n; ++i) {
        crc ^= data[i];
        for (int k = 0; k < 8; ++k) {
            crc = (crc & 1u) ? (0xedb88320u ^ (crc >> 1)) : (crc >> 1);
        }
    }
    return ~crc;
}

static std::uint32_t adler32(const unsigned char* data, size_t n)
{
    const std::uint32_t mod = 65521u;
    std::uint32_t a = 1u;
    std::uint32_t b = 0u;
    for (size_t i = 0; i < n; ++i) {
        a += data[i];
        if (a >= mod) a -= mod;
        b += a;
        b %= mod;
    }
    return (b << 16) | a;
}

static void appendU32BE(std::vector<unsigned char>& out, std::uint32_t v)
{
    out.push_back((unsigned char)((v >> 24) & 255u));
    out.push_back((unsigned char)((v >> 16) & 255u));
    out.push_back((unsigned char)((v >> 8) & 255u));
    out.push_back((unsigned char)(v & 255u));
}

static bool writePngChunk(FILE* fp,
                          const char type[4],
                          const unsigned char* data,
                          size_t n)
{
    unsigned char len[4] = {
        (unsigned char)((n >> 24) & 255u),
        (unsigned char)((n >> 16) & 255u),
        (unsigned char)((n >> 8) & 255u),
        (unsigned char)(n & 255u),
    };
    if (std::fwrite(len, 1, 4, fp) != 4 ||
        std::fwrite(type, 1, 4, fp) != 4) {
        return false;
    }
    if (n > 0 && std::fwrite(data, 1, n, fp) != n) {
        return false;
    }

    std::uint32_t crc = crc32Update(0, reinterpret_cast<const unsigned char*>(type), 4);
    if (n > 0) {
        crc = crc32Update(crc, data, n);
    }
    unsigned char crc_bytes[4] = {
        (unsigned char)((crc >> 24) & 255u),
        (unsigned char)((crc >> 16) & 255u),
        (unsigned char)((crc >> 8) & 255u),
        (unsigned char)(crc & 255u),
    };
    return std::fwrite(crc_bytes, 1, 4, fp) == 4;
}

static bool writeRgbPng(const char* filename,
                        int width,
                        int height,
                        const std::vector<unsigned char>& rgb)
{
    if (!filename || width <= 0 || height <= 0) {
        return false;
    }
    if (rgb.size() != (size_t)width * (size_t)height * 3u) {
        return false;
    }

    std::vector<unsigned char> scanlines;
    scanlines.resize(((size_t)width * 3u + 1u) * (size_t)height);
    for (int y = 0; y < height; ++y) {
        const size_t dst = (size_t)y * ((size_t)width * 3u + 1u);
        const size_t src = (size_t)y * (size_t)width * 3u;
        scanlines[dst] = 0;
        std::memcpy(scanlines.data() + dst + 1u, rgb.data() + src, (size_t)width * 3u);
    }

    std::vector<unsigned char> z;
    z.reserve(scanlines.size() + scanlines.size() / 65535u * 5u + 16u);
    z.push_back(0x78);
    z.push_back(0x01);
    size_t off = 0;
    while (off < scanlines.size()) {
        const size_t n = std::min<size_t>(65535u, scanlines.size() - off);
        const bool final_block = (off + n == scanlines.size());
        z.push_back(final_block ? 1u : 0u);
        z.push_back((unsigned char)(n & 255u));
        z.push_back((unsigned char)((n >> 8) & 255u));
        const std::uint16_t nlen = (std::uint16_t)~(std::uint16_t)n;
        z.push_back((unsigned char)(nlen & 255u));
        z.push_back((unsigned char)((nlen >> 8) & 255u));
        z.insert(z.end(), scanlines.begin() + (ptrdiff_t)off,
                 scanlines.begin() + (ptrdiff_t)(off + n));
        off += n;
    }
    appendU32BE(z, adler32(scanlines.data(), scanlines.size()));

    FILE* fp = std::fopen(filename, "wb");
    if (!fp) {
        std::perror("writeRgbPng: fopen");
        return false;
    }

    const unsigned char sig[8] = {137, 80, 78, 71, 13, 10, 26, 10};
    bool ok = std::fwrite(sig, 1, sizeof(sig), fp) == sizeof(sig);

    unsigned char ihdr[13];
    ihdr[0] = (unsigned char)((width >> 24) & 255);
    ihdr[1] = (unsigned char)((width >> 16) & 255);
    ihdr[2] = (unsigned char)((width >> 8) & 255);
    ihdr[3] = (unsigned char)(width & 255);
    ihdr[4] = (unsigned char)((height >> 24) & 255);
    ihdr[5] = (unsigned char)((height >> 16) & 255);
    ihdr[6] = (unsigned char)((height >> 8) & 255);
    ihdr[7] = (unsigned char)(height & 255);
    ihdr[8] = 8;
    ihdr[9] = 2;
    ihdr[10] = 0;
    ihdr[11] = 0;
    ihdr[12] = 0;

    ok = ok && writePngChunk(fp, "IHDR", ihdr, sizeof(ihdr));
    ok = ok && writePngChunk(fp, "IDAT", z.data(), z.size());
    ok = ok && writePngChunk(fp, "IEND", nullptr, 0);

    if (std::fclose(fp) != 0) {
        std::perror("writeRgbPng: fclose");
        ok = false;
    }
    return ok;
}

static int reductionBlockCount(size_t n)
{
    const size_t threads = 256u;
    const size_t min_blocks = (n + threads - 1u) / threads;
    const size_t capped = std::min<size_t>(4096u, std::max<size_t>(1u, min_blocks));
    return (int)capped;
}

static bool computeMinMaxOnDevice(const float* field,
                                  size_t n,
                                  float& minv,
                                  float& maxv)
{
    const int blocks = reductionBlockCount(n);
    MinMaxBlock* d_blocks = nullptr;
    cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&d_blocks),
                                 (size_t)blocks * sizeof(MinMaxBlock));
    if (err != cudaSuccess) {
        std::fprintf(stderr, "computeMinMaxOnDevice: cudaMalloc failed: %s\n",
                     cudaGetErrorString(err));
        return false;
    }

    reduceMinMaxKernel<<<blocks, 256>>>(field, n, d_blocks);
    err = cudaGetLastError();
    if (err == cudaSuccess) {
        err = cudaDeviceSynchronize();
    }
    if (err != cudaSuccess) {
        std::fprintf(stderr, "computeMinMaxOnDevice: reduce failed: %s\n",
                     cudaGetErrorString(err));
        cudaFree(d_blocks);
        return false;
    }

    std::vector<MinMaxBlock> h_blocks((size_t)blocks);
    err = cudaMemcpy(h_blocks.data(), d_blocks, h_blocks.size() * sizeof(MinMaxBlock),
                     cudaMemcpyDeviceToHost);
    cudaFree(d_blocks);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "computeMinMaxOnDevice: copy failed: %s\n",
                     cudaGetErrorString(err));
        return false;
    }

    minv = FLT_MAX;
    maxv = -FLT_MAX;
    unsigned long long count = 0;
    for (const MinMaxBlock& b : h_blocks) {
        if (b.count == 0) {
            continue;
        }
        minv = std::min(minv, b.minv);
        maxv = std::max(maxv, b.maxv);
        count += b.count;
    }

    if (count == 0) {
        minv = 0.0f;
        maxv = 0.0f;
    }
    return true;
}

static bool computePixelStatsOnDevice(const float* field,
                                      size_t n,
                                      float minv,
                                      float inv_range,
                                      bool is_constant,
                                      PixelStats& stats)
{
    const int blocks = reductionBlockCount(n);
    PixelStats* d_blocks = nullptr;
    cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&d_blocks),
                                 (size_t)blocks * sizeof(PixelStats));
    if (err != cudaSuccess) {
        std::fprintf(stderr, "computePixelStatsOnDevice: cudaMalloc failed: %s\n",
                     cudaGetErrorString(err));
        return false;
    }

    reducePixelStatsKernel<<<blocks, 256>>>(field, n, d_blocks,
                                            minv, inv_range, is_constant);
    err = cudaGetLastError();
    if (err == cudaSuccess) {
        err = cudaDeviceSynchronize();
    }
    if (err != cudaSuccess) {
        std::fprintf(stderr, "computePixelStatsOnDevice: reduce failed: %s\n",
                     cudaGetErrorString(err));
        cudaFree(d_blocks);
        return false;
    }

    std::vector<PixelStats> h_blocks((size_t)blocks);
    err = cudaMemcpy(h_blocks.data(), d_blocks, h_blocks.size() * sizeof(PixelStats),
                     cudaMemcpyDeviceToHost);
    cudaFree(d_blocks);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "computePixelStatsOnDevice: copy failed: %s\n",
                     cudaGetErrorString(err));
        return false;
    }

    stats = PixelStats{0.0, 0.0};
    for (const PixelStats& b : h_blocks) {
        stats.sum += b.sum;
        stats.sum2 += b.sum2;
    }
    return true;
}

}  // namespace

int dnsCudaMovieScaleF(int N)
{
    if (N == 9216) return 8;
    if (N == 16384) return 12;
    if (N == 29400) return 14;
    const int full_n = 3 * N / 2;
    return std::max(1, (full_n + 3839) / 3840);
}

bool dnsCudaSaveFieldInfernoPng(DnsDeviceState *S,
                                int comp,
                                const char *filename,
                                int scale_f)
{
    if (!S || !filename) {
        return false;
    }
    if (comp < 0 || comp > 2) {
        std::fprintf(stderr, "dnsCudaSaveFieldInfernoPng: invalid comp=%d\n", comp);
        return false;
    }
    if (scale_f <= 0) {
        scale_f = 1;
    }
    if (!ensureInfernoLut()) {
        return false;
    }

    const int nx = S->NX_full;
    const int nz = S->NZ_full;
    const int out_w = (nx + scale_f - 1) / scale_f;
    const int out_h = (nz + scale_f - 1) / scale_f;
    const size_t plane = (size_t)nx * (size_t)nz;
    const float* field = S->d_ur_full + plane * (size_t)comp;

    float minv = 0.0f;
    float maxv = 0.0f;
    if (!computeMinMaxOnDevice(field, plane, minv, maxv)) {
        return false;
    }

    const float range = maxv - minv;
    const bool is_constant = !(range > 1.0e-12f);
    const float inv_range = is_constant ? 1.0f : 1.0f / range;

    PixelStats stats{0.0, 0.0};
    if (!computePixelStatsOnDevice(field, plane, minv, inv_range, is_constant, stats)) {
        return false;
    }

    const double mean = stats.sum / (double)plane;
    const double var = std::max(0.0, stats.sum2 / (double)plane - mean * mean);
    const double sigma = std::sqrt(var);

    unsigned char* d_rgb = nullptr;
    const size_t rgb_bytes = (size_t)out_w * (size_t)out_h * 3u;
    cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&d_rgb), rgb_bytes);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "dnsCudaSaveFieldInfernoPng: cudaMalloc RGB failed: %s\n",
                     cudaGetErrorString(err));
        return false;
    }

    const dim3 block(16, 16);
    const dim3 grid((out_w + block.x - 1) / block.x,
                    (out_h + block.y - 1) / block.y);
    renderInfernoKernel<<<grid, block>>>(field,
                                         d_rgb,
                                         nx,
                                         nz,
                                         out_w,
                                         out_h,
                                         scale_f,
                                         minv,
                                         inv_range,
                                         is_constant,
                                         mean,
                                         sigma);
    err = cudaGetLastError();
    if (err == cudaSuccess) {
        err = cudaDeviceSynchronize();
    }
    if (err != cudaSuccess) {
        std::fprintf(stderr, "dnsCudaSaveFieldInfernoPng: render failed: %s\n",
                     cudaGetErrorString(err));
        cudaFree(d_rgb);
        return false;
    }

    std::vector<unsigned char> rgb(rgb_bytes);
    err = cudaMemcpy(rgb.data(), d_rgb, rgb_bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_rgb);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "dnsCudaSaveFieldInfernoPng: RGB copy failed: %s\n",
                     cudaGetErrorString(err));
        return false;
    }

    if (!writeRgbPng(filename, out_w, out_h, rgb)) {
        std::fprintf(stderr, "dnsCudaSaveFieldInfernoPng: failed to write %s\n", filename);
        return false;
    }

    std::printf("[MOV] Wrote %s (Inferno PNG, %dx%d, F=%d, comp=%d, sigma=%d)\n",
                filename, out_w, out_h, scale_f, comp, (int)sigma);
    return true;
}
