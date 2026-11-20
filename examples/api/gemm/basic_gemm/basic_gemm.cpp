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
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

// Basic public GEMM call: element strides, byte offsets, residual, bias and clamp.
// Dyadic inputs keep the reference comparison exactly representable.
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
    size_t m = 0, n = 0, k = 0;
};

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

    u.run_gemm(m, n, k, a.data(), lda, 0, b.data(), ldb, 0, c.data(), ldd, d.data(), ldd,
               bias.data(), lo, hi);

    if (!verify(m, n, d.data(), ldd, ref))
        return false;
    for (size_t i = 0; i < m; ++i)
        for (size_t j = n; j < ldd; ++j)
            if (d[i * ldd + j] != 12345.0f)
                return false;
    return true;
}

}  // namespace

int main()
{
    try {
        const Options options{};
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
