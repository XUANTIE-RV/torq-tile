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

// Serial N/K/M panel traversal: one packed B panel is reused across M blocks.
// A/B panels are independently packed with local K origin zero, not sliced out of
// an unrelated packed allocation. Only the final epilogue adds C, bias and clamp.
#if TORQ_TILE_HAS_RVV
#include "torq_tile/gemm/gemm_f32_f32/gemm_1xnbias_clamp_f32_f32p_f32p/tqt_gemm_1xnbias_clamp_f32_f32p_f32p_interface.h"
#endif
#if TORQ_TILE_HAS_RVV
#include "torq_tile/gemm/gemm_f32_f32/gemm_1xnbias_clamp_f32_f32p_f32p/tqt_gemm_1xnbias_clamp_f32_f32p_f32p_8x3vl_rvv.h"
#endif
#if TORQ_TILE_HAS_RVV
#include "torq_tile/gemm/gemm_f32_f32/gemm_1xnbias_clamp_f32_f32p_f32p/tqt_gemm_1xnbias_clamp_f32_f32p_f32p_4x4vl_rvv.h"
#endif
#if TORQ_TILE_HAS_RVV
#include "torq_tile/gemm/gemm_f32_f32/gemm_1xnbias_clamp_f32_f32p_f32p/tqt_gemm_1xnbias_clamp_f32_f32p_f32p_2x8vl_rvv.h"
#endif

namespace
{
using Kernel = tqt_gemm_1xnbias_clamp_f32_f32p_f32p_ukernel;
struct Variant
{
    const char *name;
    Kernel kernel;
};
const Variant variants[] = {
#if TORQ_TILE_HAS_RVV
    {"gemm_1xnbias_clamp_f32_f32p_f32p_8x3vl_rvv",
     {tqt_get_m_step_gemm_1xnbias_clamp_f32_f32p_f32p_8x3vl_rvv,
      tqt_get_n_step_gemm_1xnbias_clamp_f32_f32p_f32p_8x3vl_rvv,
      tqt_get_a_packed_offset_gemm_1xnbias_clamp_f32_f32p_f32p_8x3vl_rvv,
      tqt_get_b_packed_offset_gemm_1xnbias_clamp_f32_f32p_f32p_8x3vl_rvv,
      tqt_get_c_offset_gemm_1xnbias_clamp_f32_f32p_f32p_8x3vl_rvv,
      tqt_get_bias_offset_gemm_1xnbias_clamp_f32_f32p_f32p_8x3vl_rvv,
      tqt_get_d_offset_gemm_1xnbias_clamp_f32_f32p_f32p_8x3vl_rvv,
      tqt_get_d_size_gemm_1xnbias_clamp_f32_f32p_f32p_8x3vl_rvv,
      tqt_get_a_packed_size_gemm_1xnbias_clamp_f32_f32p_f32p_8x3vl_rvv,
      tqt_get_b_packed_size_gemm_1xnbias_clamp_f32_f32p_f32p_8x3vl_rvv,
      tqt_run_a_pack_gemm_1xnbias_clamp_f32_f32p_f32p_8x3vl_rvv,
      tqt_run_b_pack_gemm_1xnbias_clamp_f32_f32p_f32p_8x3vl_rvv,
      tqt_run_bt_pack_gemm_1xnbias_clamp_f32_f32p_f32p_8x3vl_rvv,
      tqt_run_gemm_1xnbias_clamp_f32_f32p_f32p_8x3vl_rvv}},
#endif
#if TORQ_TILE_HAS_RVV
    {"gemm_1xnbias_clamp_f32_f32p_f32p_4x4vl_rvv",
     {tqt_get_m_step_gemm_1xnbias_clamp_f32_f32p_f32p_4x4vl_rvv,
      tqt_get_n_step_gemm_1xnbias_clamp_f32_f32p_f32p_4x4vl_rvv,
      tqt_get_a_packed_offset_gemm_1xnbias_clamp_f32_f32p_f32p_4x4vl_rvv,
      tqt_get_b_packed_offset_gemm_1xnbias_clamp_f32_f32p_f32p_4x4vl_rvv,
      tqt_get_c_offset_gemm_1xnbias_clamp_f32_f32p_f32p_4x4vl_rvv,
      tqt_get_bias_offset_gemm_1xnbias_clamp_f32_f32p_f32p_4x4vl_rvv,
      tqt_get_d_offset_gemm_1xnbias_clamp_f32_f32p_f32p_4x4vl_rvv,
      tqt_get_d_size_gemm_1xnbias_clamp_f32_f32p_f32p_4x4vl_rvv,
      tqt_get_a_packed_size_gemm_1xnbias_clamp_f32_f32p_f32p_4x4vl_rvv,
      tqt_get_b_packed_size_gemm_1xnbias_clamp_f32_f32p_f32p_4x4vl_rvv,
      tqt_run_a_pack_gemm_1xnbias_clamp_f32_f32p_f32p_4x4vl_rvv,
      tqt_run_b_pack_gemm_1xnbias_clamp_f32_f32p_f32p_4x4vl_rvv,
      tqt_run_bt_pack_gemm_1xnbias_clamp_f32_f32p_f32p_4x4vl_rvv,
      tqt_run_gemm_1xnbias_clamp_f32_f32p_f32p_4x4vl_rvv}},
#endif
#if TORQ_TILE_HAS_RVV
    {"gemm_1xnbias_clamp_f32_f32p_f32p_2x8vl_rvv",
     {tqt_get_m_step_gemm_1xnbias_clamp_f32_f32p_f32p_2x8vl_rvv,
      tqt_get_n_step_gemm_1xnbias_clamp_f32_f32p_f32p_2x8vl_rvv,
      tqt_get_a_packed_offset_gemm_1xnbias_clamp_f32_f32p_f32p_2x8vl_rvv,
      tqt_get_b_packed_offset_gemm_1xnbias_clamp_f32_f32p_f32p_2x8vl_rvv,
      tqt_get_c_offset_gemm_1xnbias_clamp_f32_f32p_f32p_2x8vl_rvv,
      tqt_get_bias_offset_gemm_1xnbias_clamp_f32_f32p_f32p_2x8vl_rvv,
      tqt_get_d_offset_gemm_1xnbias_clamp_f32_f32p_f32p_2x8vl_rvv,
      tqt_get_d_size_gemm_1xnbias_clamp_f32_f32p_f32p_2x8vl_rvv,
      tqt_get_a_packed_size_gemm_1xnbias_clamp_f32_f32p_f32p_2x8vl_rvv,
      tqt_get_b_packed_size_gemm_1xnbias_clamp_f32_f32p_f32p_2x8vl_rvv,
      tqt_run_a_pack_gemm_1xnbias_clamp_f32_f32p_f32p_2x8vl_rvv,
      tqt_run_b_pack_gemm_1xnbias_clamp_f32_f32p_f32p_2x8vl_rvv,
      tqt_run_bt_pack_gemm_1xnbias_clamp_f32_f32p_f32p_2x8vl_rvv,
      tqt_run_gemm_1xnbias_clamp_f32_f32p_f32p_2x8vl_rvv}},
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
        else if (!std::strcmp(argv[i], "--kc"))
            o.kc = value;
        else
            throw std::invalid_argument("unknown option");
    }
    if (o.threads > 256)
        throw std::invalid_argument("at most 256 workers are supported");
    return o;
}

bool run_case(const Variant &variant, const Options &o)
{
    const auto &u = variant.kernel;
    const size_t ms = u.get_m_step(), ns = u.get_n_step();
    const size_t mc = o.mc ? o.mc : 2 * ms, nc = o.nc ? o.nc : 2 * ns, kc = o.kc ? o.kc : 32;
    if (mc % ms || nc % ns)
        throw std::invalid_argument("MC/NC must align to micro-tile steps");
    const size_t m = o.m ? o.m : 2 * mc + 1, n = o.n ? o.n : 2 * nc + 3, k = o.k ? o.k : 2 * kc + 5;
    const size_t lda = k + 3, ldb = n + 5, ldd = n + 7;
    std::vector<float> a(checked_product(m, lda)), b(checked_product(k, ldb));
    std::vector<float> c(checked_product(m, ldd), 0.25f), d(c.size(), 12345.0f), bias(n, 0.125f);
    for (size_t i = 0; i < a.size(); ++i)
        a[i] = input_value(i);
    for (size_t i = 0; i < b.size(); ++i)
        b[i] = input_value(i + 5);
    const auto ref = reference_gemm(m, n, k, a.data(), lda, b.data(), ldb, c.data(), ldd,
                                    bias.data(), -0.5f, 0.75f);
    size_t b_packs = 0, a_packs = 0;
    for (size_t ni = 0; ni < n; ni += std::min(nc, n - ni)) {
        const size_t cols = std::min(nc, n - ni);
        for (size_t ki = 0; ki < k; ki += std::min(kc, k - ki)) {
            const size_t depth = std::min(kc, k - ki);
            std::vector<float> bp(u.get_b_packed_size(cols, depth) / sizeof(float));
            u.run_b_pack(cols, depth, ldb, depth, 0, b.data() + ki * ldb + ni, bp.data());
            ++b_packs;
            for (size_t mi = 0; mi < m; mi += std::min(mc, m - mi)) {
                const size_t rows = std::min(mc, m - mi);
                std::vector<float> ap(u.get_a_packed_size(rows, depth) / sizeof(float));
                u.run_a_pack(rows, depth, lda, depth, 0, a.data() + mi * lda + ki, ap.data());
                ++a_packs;
                float *out = d.data() + mi * ldd + ni;
                u.run_gemm(rows, cols, depth, ap.data(), depth, 0, bp.data(), depth, 0,
                           ki == 0 ? nullptr : out, ldd, out, ldd, nullptr, -INFINITY, INFINITY);
            }
        }
    }
    for (size_t i = 0; i < m; ++i)
        for (size_t j = 0; j < n; ++j)
            d[i * ldd + j] =
                std::min(0.75f, std::max(-0.5f, d[i * ldd + j] + c[i * ldd + j] + bias[j]));
    std::printf("%s M/N/K=%zu/%zu/%zu MC/NC/KC=%zu/%zu/%zu A_packs=%zu B_packs=%zu\n", variant.name,
                m, n, k, mc, nc, kc, a_packs, b_packs);

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
