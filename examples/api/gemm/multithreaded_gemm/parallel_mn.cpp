//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#if !defined(__riscv)
#error These examples require a RISC-V target.
#endif

#include <torq_tile/tqt_compile_config.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <thread>
#include <vector>

#if TORQ_TILE_HAS_RVV || TORQ_TILE_HAS_ZTT_GEMM_F32
#include "torq_tile/gemm/gemm_f32_f32/gemm_1xnbias_clamp_f32_f32_f32/tqt_gemm_1xnbias_clamp_f32_f32_f32_interface.h"
#endif
#if TORQ_TILE_HAS_RVV
#include "torq_tile/gemm/gemm_f32_f32/gemm_1xnbias_clamp_f32_f32_f32/tqt_gemm_1xnbias_clamp_f32_f32_f32_8x3vl_rvv.h"
#endif
#if TORQ_TILE_HAS_RVV
#include "torq_tile/gemm/gemm_f32_f32/gemm_1xnbias_clamp_f32_f32_f32/tqt_gemm_1xnbias_clamp_f32_f32_f32_4x4vl_rvv.h"
#endif
#if TORQ_TILE_HAS_RVV
#include "torq_tile/gemm/gemm_f32_f32/gemm_1xnbias_clamp_f32_f32_f32/tqt_gemm_1xnbias_clamp_f32_f32_f32_2x8vl_rvv.h"
#endif
#if TORQ_TILE_HAS_ZTT_GEMM_F32
#include "torq_tile/gemm/gemm_f32_f32/gemm_1xnbias_clamp_f32_f32_f32/tqt_gemm_1xnbias_clamp_f32_f32_f32_1nlenxpnlen_ztt.h"
#endif

namespace
{
using Kernel = tqt_gemm_1xnbias_clamp_f32_f32_f32_ukernel;
struct Variant
{
    const char *name;
    Kernel kernel;
    bool aligned_shape;
};
const Variant variants[] = {
#if TORQ_TILE_HAS_RVV
    {"gemm_1xnbias_clamp_f32_f32_f32_8x3vl_rvv",
     {tqt_get_m_step_gemm_1xnbias_clamp_f32_f32_f32_8x3vl_rvv,
      tqt_get_n_step_gemm_1xnbias_clamp_f32_f32_f32_8x3vl_rvv,
      tqt_get_a_offset_gemm_1xnbias_clamp_f32_f32_f32_8x3vl_rvv,
      tqt_get_b_offset_gemm_1xnbias_clamp_f32_f32_f32_8x3vl_rvv,
      tqt_get_c_offset_gemm_1xnbias_clamp_f32_f32_f32_8x3vl_rvv,
      tqt_get_bias_offset_gemm_1xnbias_clamp_f32_f32_f32_8x3vl_rvv,
      tqt_get_d_offset_gemm_1xnbias_clamp_f32_f32_f32_8x3vl_rvv,
      tqt_get_d_size_gemm_1xnbias_clamp_f32_f32_f32_8x3vl_rvv,
      tqt_run_gemm_1xnbias_clamp_f32_f32_f32_8x3vl_rvv},
     false},
#endif
#if TORQ_TILE_HAS_RVV
    {"gemm_1xnbias_clamp_f32_f32_f32_4x4vl_rvv",
     {tqt_get_m_step_gemm_1xnbias_clamp_f32_f32_f32_4x4vl_rvv,
      tqt_get_n_step_gemm_1xnbias_clamp_f32_f32_f32_4x4vl_rvv,
      tqt_get_a_offset_gemm_1xnbias_clamp_f32_f32_f32_4x4vl_rvv,
      tqt_get_b_offset_gemm_1xnbias_clamp_f32_f32_f32_4x4vl_rvv,
      tqt_get_c_offset_gemm_1xnbias_clamp_f32_f32_f32_4x4vl_rvv,
      tqt_get_bias_offset_gemm_1xnbias_clamp_f32_f32_f32_4x4vl_rvv,
      tqt_get_d_offset_gemm_1xnbias_clamp_f32_f32_f32_4x4vl_rvv,
      tqt_get_d_size_gemm_1xnbias_clamp_f32_f32_f32_4x4vl_rvv,
      tqt_run_gemm_1xnbias_clamp_f32_f32_f32_4x4vl_rvv},
     false},
#endif
#if TORQ_TILE_HAS_RVV
    {"gemm_1xnbias_clamp_f32_f32_f32_2x8vl_rvv",
     {tqt_get_m_step_gemm_1xnbias_clamp_f32_f32_f32_2x8vl_rvv,
      tqt_get_n_step_gemm_1xnbias_clamp_f32_f32_f32_2x8vl_rvv,
      tqt_get_a_offset_gemm_1xnbias_clamp_f32_f32_f32_2x8vl_rvv,
      tqt_get_b_offset_gemm_1xnbias_clamp_f32_f32_f32_2x8vl_rvv,
      tqt_get_c_offset_gemm_1xnbias_clamp_f32_f32_f32_2x8vl_rvv,
      tqt_get_bias_offset_gemm_1xnbias_clamp_f32_f32_f32_2x8vl_rvv,
      tqt_get_d_offset_gemm_1xnbias_clamp_f32_f32_f32_2x8vl_rvv,
      tqt_get_d_size_gemm_1xnbias_clamp_f32_f32_f32_2x8vl_rvv,
      tqt_run_gemm_1xnbias_clamp_f32_f32_f32_2x8vl_rvv},
     false},
#endif
#if TORQ_TILE_HAS_ZTT_GEMM_F32
    {"gemm_1xnbias_clamp_f32_f32_f32_1nlenxpnlen_ztt",
     {tqt_get_m_step_gemm_1xnbias_clamp_f32_f32_f32_1nlenxpnlen_ztt,
      tqt_get_n_step_gemm_1xnbias_clamp_f32_f32_f32_1nlenxpnlen_ztt,
      tqt_get_a_offset_gemm_1xnbias_clamp_f32_f32_f32_1nlenxpnlen_ztt,
      tqt_get_b_offset_gemm_1xnbias_clamp_f32_f32_f32_1nlenxpnlen_ztt,
      tqt_get_c_offset_gemm_1xnbias_clamp_f32_f32_f32_1nlenxpnlen_ztt,
      tqt_get_bias_offset_gemm_1xnbias_clamp_f32_f32_f32_1nlenxpnlen_ztt,
      tqt_get_d_offset_gemm_1xnbias_clamp_f32_f32_f32_1nlenxpnlen_ztt,
      tqt_get_d_size_gemm_1xnbias_clamp_f32_f32_f32_1nlenxpnlen_ztt,
      tqt_run_gemm_1xnbias_clamp_f32_f32_f32_1nlenxpnlen_ztt},
     true},
#endif
};

// Keep reference computation and checks visible in this standalone teaching source.
size_t checked_product(size_t a, size_t b)
{
    if (b && a > std::numeric_limits<size_t>::max() / b)
        throw std::overflow_error("matrix size overflow");
    return a * b;
}
float input_value(size_t index)
{
    return (static_cast<int>(index % 17) - 8) * 0.0625f;
}
bool verify(size_t m, size_t n, const float *actual, size_t stride,
            const std::vector<float> &reference, float tolerance = 1e-4f)
{
    float max_error = 0;
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            const float a = actual[i * stride + j], r = reference[i * n + j];
            if (!std::isfinite(a) || !std::isfinite(r) ||
                std::fabs(a - r) > tolerance * (1.0f + std::fabs(r))) {
                std::fprintf(stderr, "Mismatch [%zu,%zu]: actual=%g reference=%g\n", i, j, a, r);
                return false;
            }
            max_error = std::max(max_error, std::fabs(a - r));
        }
    }
    std::printf("  maximum absolute error: %g\n", max_error);
    return true;
}
std::vector<float> reference_gemm(size_t m, size_t n, size_t k, const float *a, size_t lda,
                                  const float *b, size_t ldb, const float *c, size_t ldc,
                                  const float *bias, float lo, float hi)
{
    std::vector<float> out(checked_product(m, n));
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            float sum = 0;
            for (size_t q = 0; q < k; ++q)
                sum = std::fma(a[i * lda + q], b[q * ldb + j], sum);
            if (c)
                sum += c[i * ldc + j];
            if (bias)
                sum += bias[j];
            out[i * n + j] = std::min(hi, std::max(lo, sum));
        }
    }
    return out;
}

struct Options
{
    size_t m = 0, n = 0, k = 0, mc = 0, nc = 0, kc = 0, threads = 8;
};
size_t positive(const char *value)
{
    if (!value[0])
        throw std::invalid_argument("expected a positive integer");
    size_t result = 0;
    for (const char *p = value; *p; ++p) {
        if (*p < '0' || *p > '9' || result > (std::numeric_limits<size_t>::max() - (*p - '0')) / 10)
            throw std::invalid_argument("invalid or overflowing integer");
        result = result * 10 + (*p - '0');
    }
    if (!result || result > (1u << 20))
        throw std::invalid_argument("options must be between 1 and 1048576");
    return result;
}
Options parse(int argc, char **argv)
{
    Options o;
    for (int i = 1; i < argc; i += 2) {
        if (i + 1 == argc)
            throw std::invalid_argument("missing option value");
        const size_t value = positive(argv[i + 1]);
        if (!std::strcmp(argv[i], "--m"))
            o.m = value;
        else if (!std::strcmp(argv[i], "--n"))
            o.n = value;
        else if (!std::strcmp(argv[i], "--k"))
            o.k = value;
        else if (!std::strcmp(argv[i], "--mc"))
            o.mc = value;
        else if (!std::strcmp(argv[i], "--nc"))
            o.nc = value;
        else if (!std::strcmp(argv[i], "--threads"))
            o.threads = value;
        else
            throw std::invalid_argument("unknown option");
    }
    if (o.threads > 256)
        throw std::invalid_argument("at most 256 workers are supported");
    return o;
}

template <typename Function>
void parallel_tasks(size_t count, size_t workers, Function function)
{
    std::vector<std::thread> threads;
    std::atomic<bool> failed{false};
    threads.reserve(workers);
    try {
        for (size_t worker = 0; worker < workers; ++worker) {
            threads.emplace_back([&, worker] {
                try {
                    for (size_t task = worker; task < count; task += workers)
                        function(task);
                } catch (...) {
                    failed.store(true);
                }
            });
        }
    } catch (...) {
        for (auto &thread : threads)
            thread.join();
        throw;
    }
    for (auto &thread : threads)
        thread.join();
    if (failed.load())
        throw std::runtime_error("worker failed");
}

bool run_case(const Variant &variant, const Options &o)
{
    const auto &u = variant.kernel;
    const size_t ms = u.get_m_step(), ns = u.get_n_step();
    const size_t m = o.m ? o.m : 2 * ms + (variant.aligned_shape ? 0 : 1);
    const size_t n = o.n ? o.n : 2 * ns + (variant.aligned_shape ? 0 : 3);
    const size_t k = o.k ? o.k : (variant.aligned_shape ? 3 * ns : 37);
    if (variant.aligned_shape && (m % ms || n % ns || k % ns))
        throw std::invalid_argument(
            "Ztt ordinary GEMM requires aligned M/N/K; see packed_gemm for tails");
    const size_t lda = k + (variant.aligned_shape ? ns : 3);
    const size_t ldb = n + (variant.aligned_shape ? ns : 5);
    const size_t ldd = n + (variant.aligned_shape ? ns : 7);
    std::vector<float> a(checked_product(m, lda)), b(checked_product(k, ldb));
    std::vector<float> c(checked_product(m, ldd), 0.25f), d(c.size(), 12345.0f), bias(n);
    for (size_t i = 0; i < a.size(); ++i)
        a[i] = input_value(i);
    for (size_t i = 0; i < b.size(); ++i)
        b[i] = input_value(i + 5);
    for (size_t j = 0; j < n; ++j)
        bias[j] = input_value(j + 3);
    const float lo = -0.5f, hi = 0.75f;
    const auto ref =
        reference_gemm(m, n, k, a.data(), lda, b.data(), ldb, c.data(), ldd, bias.data(), lo, hi);
    std::printf("%s: M=%zu N=%zu K=%zu, strides=%zu/%zu/%zu\n", variant.name, m, n, k, lda, ldb,
                ldd);

    const size_t mc = o.mc ? o.mc : ms, nc = o.nc ? o.nc : ns;
    if (mc % ms || nc % ns)
        throw std::invalid_argument("MC/NC must align to the variant steps");
    const size_t m_tasks = 1 + (m - 1) / mc, n_tasks = 1 + (n - 1) / nc;
    const size_t tasks = checked_product(m_tasks, n_tasks);
    const auto start = std::chrono::steady_clock::now();
    parallel_tasks(tasks, o.threads, [&](size_t task) {
        const size_t mi = (task / n_tasks) * mc, ni = (task % n_tasks) * nc;
        const size_t rows = std::min(mc, m - mi), cols = std::min(nc, n - ni);
        const auto *ap = reinterpret_cast<const uint8_t *>(a.data()) + u.get_a_offset(mi, 0, lda);
        const auto *bp = reinterpret_cast<const uint8_t *>(b.data()) + u.get_b_offset(ni, 0, ldb);
        u.run_gemm(rows, cols, k, ap, lda, 0, bp, ldb, 0, c.data() + mi * ldd + ni, ldd,
                   d.data() + mi * ldd + ni, ldd, bias.data() + ni, lo, hi);
    });
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::printf("  MN tasks=%zu workers=%zu time_with_create_join=%g s\n", tasks, o.threads,
                seconds);

    if (!verify(m, n, d.data(), ldd, ref))
        return false;
    for (size_t i = 0; i < m; ++i)
        for (size_t j = n; j < ldd; ++j)
            if (d[i * ldd + j] != 12345.0f)
                return false;
    return true;
}

}  // namespace

int main(int argc, char **argv)
{
    try {
        const Options options = parse(argc, argv);
        bool ok = true;
        for (const auto &variant : variants)
            ok = run_case(variant, options) && ok;
        std::printf("Status: %s\n", ok ? "PASSED" : "FAILED");
        return ok ? 0 : 1;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "FAILED: %s\n", error.what());
        return 1;
    }
}
