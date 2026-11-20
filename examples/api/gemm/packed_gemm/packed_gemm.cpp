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

// B[K,N] and BT[N,K] pack into the same representation for one variant.
// Packing belongs to a variant and runtime geometry; never reuse it across layouts.
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
#if TORQ_TILE_HAS_ZTT_GEMM_F32
#include "torq_tile/gemm/gemm_f32_f32/gemm_1xnbias_clamp_f32p_f32p_f32p/tqt_gemm_1xnbias_clamp_f32p_f32p_f32p_interface.h"
#endif
#if TORQ_TILE_HAS_ZTT_GEMM_F32
#include "torq_tile/gemm/gemm_f32_f32/gemm_1xnbias_clamp_f32p_f32p_f32p/tqt_gemm_1xnbias_clamp_f32p_f32p_f32p_1nlenxpnlen_ztt.h"
#endif

namespace
{

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

#if TORQ_TILE_HAS_RVV
namespace ordinary_output
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

bool run_case(const Variant &variant)
{
    const auto &u = variant.kernel;
    const size_t m = 2 * u.get_m_step() + 1, n = 2 * u.get_n_step() + 3, k = 37;
    const size_t lda = k + 3, ldb = n + 5, ldbt = k + 7, ldd = n + 9;
    std::vector<float> a(m * lda), b(k * ldb), bt(n * ldbt), c(m * ldd, 0.25f);
    std::vector<float> d(m * ldd), bias(n, 0.125f);
    for (size_t q = 0; q < k; ++q)
        for (size_t j = 0; j < n; ++j)
            b[q * ldb + j] = bt[j * ldbt + q] = input_value(q * n + j);
    std::vector<float> ap(u.get_a_packed_size(m, k) / sizeof(float));
    std::vector<float> bp(u.get_b_packed_size(n, k) / sizeof(float));
    std::vector<float> btp(bp.size());

    u.run_b_pack(n, k, ldb, k, 0, b.data(), bp.data());
    u.run_bt_pack(n, k, ldbt, k, 0, bt.data(), btp.data());
    if (bp != btp)
        throw std::runtime_error("B and BT must produce identical packed weights");
    // The same B buffer is reused for two separately generated activation matrices.
    for (size_t batch = 0; batch < 2; ++batch) {
        for (size_t i = 0; i < a.size(); ++i)
            a[i] = input_value(i + batch * 3);
        u.run_a_pack(m, k, lda, k, 0, a.data(), ap.data());
        u.run_gemm(m, n, k, ap.data(), k, 0, bp.data(), k, 0, c.data(), ldd, d.data(), ldd,
                   bias.data(), -0.5f, 0.75f);
        const auto ref = reference_gemm(m, n, k, a.data(), lda, b.data(), ldb, c.data(), ldd,
                                        bias.data(), -0.5f, 0.75f);
        std::printf("%s batch=%zu B_bytes=%zu\n", variant.name, batch, bp.size() * sizeof(float));
        if (!verify(m, n, d.data(), ldd, ref))
            return false;
    }
    return true;
}
}  // namespace ordinary_output
#endif
#if TORQ_TILE_HAS_ZTT_GEMM_F32
namespace packed_output
{
using Kernel = tqt_gemm_1xnbias_clamp_f32p_f32p_f32p_ukernel;
struct Variant
{
    const char *name;
    Kernel kernel;
};
const Variant variants[] = {
#if TORQ_TILE_HAS_ZTT_GEMM_F32
    {"gemm_1xnbias_clamp_f32p_f32p_f32p_1nlenxpnlen_ztt",
     {tqt_get_m_step_gemm_1xnbias_clamp_f32p_f32p_f32p_1nlenxpnlen_ztt,
      tqt_get_n_step_gemm_1xnbias_clamp_f32p_f32p_f32p_1nlenxpnlen_ztt,
      tqt_get_a_packed_size_gemm_1xnbias_clamp_f32p_f32p_f32p_1nlenxpnlen_ztt,
      tqt_get_b_packed_size_gemm_1xnbias_clamp_f32p_f32p_f32p_1nlenxpnlen_ztt,
      tqt_get_d_packed_size_gemm_1xnbias_clamp_f32p_f32p_f32p_1nlenxpnlen_ztt,
      tqt_get_bias_packed_size_gemm_1xnbias_clamp_f32p_f32p_f32p_1nlenxpnlen_ztt,
      tqt_run_a_pack_gemm_1xnbias_clamp_f32p_f32p_f32p_1nlenxpnlen_ztt,
      tqt_run_b_pack_gemm_1xnbias_clamp_f32p_f32p_f32p_1nlenxpnlen_ztt,
      tqt_run_bt_pack_gemm_1xnbias_clamp_f32p_f32p_f32p_1nlenxpnlen_ztt,
      tqt_run_c_pack_gemm_1xnbias_clamp_f32p_f32p_f32p_1nlenxpnlen_ztt,
      tqt_run_bias_pack_gemm_1xnbias_clamp_f32p_f32p_f32p_1nlenxpnlen_ztt,
      tqt_run_d_unpack_gemm_1xnbias_clamp_f32p_f32p_f32p_1nlenxpnlen_ztt,
      tqt_run_gemm_1xnbias_clamp_f32p_f32p_f32p_1nlenxpnlen_ztt}},
#endif
};

bool run_case(const Variant &variant)
{
    const auto &u = variant.kernel;
    const size_t m = 2 * u.get_m_step() + 1, n = 2 * u.get_n_step() + 3, k = 37;
    const size_t lda = k + 3, ldb = n + 5, ldbt = k + 7, ldd = n + 9;
    std::vector<float> a(m * lda), b(k * ldb), bt(n * ldbt), c(m * ldd, 0.25f);
    std::vector<float> d(m * ldd), bias(n, 0.125f);
    for (size_t q = 0; q < k; ++q)
        for (size_t j = 0; j < n; ++j)
            b[q * ldb + j] = bt[j * ldbt + q] = input_value(q * n + j);
    std::vector<float> ap(u.get_a_packed_size(m, k) / sizeof(float));
    std::vector<float> bp(u.get_b_packed_size(n, k) / sizeof(float));
    std::vector<float> btp(bp.size());

    std::vector<float> cp(u.get_d_packed_size(m, n) / sizeof(float)), dp(cp.size());
    std::vector<float> biasp(u.get_bias_packed_size(n) / sizeof(float));
    u.run_b_pack(n, k, ldb, b.data(), bp.data());
    u.run_bt_pack(n, k, ldbt, bt.data(), btp.data());
    if (bp != btp)
        throw std::runtime_error("B and BT must produce identical packed weights");
    u.run_c_pack(m, n, ldd, c.data(), cp.data());
    u.run_bias_pack(n, bias.data(), biasp.data());
    for (size_t batch = 0; batch < 2; ++batch) {
        for (size_t i = 0; i < a.size(); ++i)
            a[i] = input_value(i + batch * 3);
        u.run_a_pack(m, k, lda, a.data(), ap.data());
        u.run_gemm(m, n, k, ap.data(), bp.data(), cp.data(), dp.data(), biasp.data(), -0.5f, 0.75f);
        u.run_d_unpack(m, n, ldd, dp.data(), d.data());
        const auto ref = reference_gemm(m, n, k, a.data(), lda, b.data(), ldb, c.data(), ldd,
                                        bias.data(), -0.5f, 0.75f);
        std::printf("%s batch=%zu packed_D_bytes=%zu\n", variant.name, batch,
                    dp.size() * sizeof(float));
        if (!verify(m, n, d.data(), ldd, ref))
            return false;
    }
    return true;
}
}  // namespace packed_output
#endif

}  // namespace

int main()
{
    try {
        bool ok = true;
#if TORQ_TILE_HAS_RVV
        for (const auto &variant : ordinary_output::variants)
            ok = ordinary_output::run_case(variant) && ok;
#endif
#if TORQ_TILE_HAS_ZTT_GEMM_F32
        for (const auto &variant : packed_output::variants)
            ok = packed_output::run_case(variant) && ok;
#endif
        std::printf("Status: %s\n", ok ? "PASSED" : "FAILED");
        return ok ? 0 : 1;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "FAILED: %s\n", error.what());
        return 1;
    }
}
