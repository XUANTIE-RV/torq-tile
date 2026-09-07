//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "src/gemm/gemm_i8_i2/gemm_1xnbias_clamp_f32_qai8dx_qsi2cxt/tqt_gemm_1xnbias_clamp_f32_qai8dx_qsi2cxt_4x1vl_rvv.h"
#include "src/gemm/gemm_i8_i2/gemm_1xnbias_clamp_f32_qai8dxp_qsi2cxp/tqt_gemm_1xnbias_clamp_f32_qai8dxp_qsi2cxp_4x1vl_rvv.h"
#include "src/gemm/gemm_i8_i2/tqt_params_i8_i2.h"
#include "src/gemm/gemm_i8_i2/tqt_utils_i8_i2.h"
#include "test/common/gemm_clamp_test_utils.h"

namespace
{

// ============================================================================
// Test parameter struct
// ============================================================================

struct GemmI8I2Params
{
    size_t m;
    size_t n;
    size_t k;
    bool has_bias;
    bool has_clamp;
    bool custom_lut;
    std::string label;
};

struct GemmI8I2ParamName
{
    std::string operator()(const testing::TestParamInfo<GemmI8I2Params> &info) const
    {
        const auto &p = info.param;
        std::string s =
            "M" + std::to_string(p.m) + "N" + std::to_string(p.n) + "K" + std::to_string(p.k);
        if (p.has_bias)
            s += "_Bias";
        if (p.has_clamp)
            s += "_Clamp";
        if (p.custom_lut)
            s += "_LUT";
        if (!p.label.empty())
            s += "_" + p.label;
        return s;
    }
};

// ============================================================================
// Quantization helpers
// ============================================================================

// Quantize fp32 [M, K] -> int8 (per-row asymmetric, qai8dx).
//   q = clamp(round(x*scale) + nudged_zp, -128, 127)
//   recip_scale = 1/scale, stored in params.scale_a
//   nudged_zp stored in params.zero_point_a
static void quantize_a_qai8dx(size_t m, size_t k, const float *A_fp32, int8_t *A_q,
                              float *recip_scale_a, int32_t *zp_a)
{
    const float qmin = static_cast<float>(INT8_MIN);
    const float qmax = static_cast<float>(INT8_MAX);

    for (size_t i = 0; i < m; ++i) {
        float row_min = A_fp32[i * k];
        float row_max = row_min;
        for (size_t j = 1; j < k; ++j) {
            float v = A_fp32[i * k + j];
            row_min = std::fmin(row_min, v);
            row_max = std::fmax(row_max, v);
        }

        float rmin = std::fmin(0.0f, row_min);
        float rmax = std::fmax(0.0f, row_max);
        float scale = (rmax == rmin) ? 1.0f : (qmax - qmin) / (rmax - rmin);
        float recip = (scale == 0.0f) ? 0.0f : 1.0f / scale;

        float zp_min_err = qmin + rmin * scale;
        float zp_max_err = qmax + rmax * scale;
        float zp_f =
            (zp_min_err + zp_max_err > 0.0f) ? (qmin - rmin * scale) : (qmax - rmax * scale);
        zp_f = std::fmax(zp_f, qmin);
        zp_f = std::fmin(zp_f, qmax);
        int32_t nudged_zp = static_cast<int32_t>(std::nearbyint(zp_f));

        for (size_t j = 0; j < k; ++j) {
            float q = std::nearbyint(A_fp32[i * k + j] * scale) + static_cast<float>(nudged_zp);
            q = std::fmax(q, qmin);
            q = std::fmin(q, qmax);
            A_q[i * k + j] = static_cast<int8_t>(q);
        }
        recip_scale_a[i] = recip;
        zp_a[i] = nudged_zp;
    }
}

// Quantize fp32 [N, K] -> int2 (per-channel symmetric, qsi2cx). Stored as packed u2 bytes
// in [N, K/4] layout (s3 s2 s1 s0 per byte). Returns scale_b[n] and sum_b[n] for each
// output channel.
//
// Symmetric int2 range used: {-2, -1, 0, 1}. We pick scale_b = max_abs / 2 so that the
// largest magnitude maps near -2 (the saturating endpoint). This matches KleidiAI's
// rhs_zero_point=2 (= u2 - 2) convention.
static void quantize_b_qsi2cx(size_t n, size_t k, const float *B_fp32, uint8_t *B_u2,
                              float *scale_b, int32_t *sum_b)
{
    const size_t k_bytes = k / 4;
    for (size_t i = 0; i < n; ++i) {
        float max_abs = 0.0f;
        for (size_t j = 0; j < k; ++j) {
            max_abs = std::fmax(max_abs, std::fabs(B_fp32[i * k + j]));
        }
        float scale = (max_abs == 0.0f) ? 1.0f : max_abs / 2.0f;
        scale_b[i] = scale;
        float recip = 1.0f / scale;

        int32_t row_sum = 0;
        for (size_t kb = 0; kb < k_bytes; ++kb) {
            uint8_t byte = 0;
            for (size_t s = 0; s < 4; ++s) {
                size_t kk = kb * 4 + s;
                float q_f = std::nearbyint(B_fp32[i * k + kk] * recip);
                q_f = std::fmax(q_f, -2.0f);
                q_f = std::fmin(q_f, 1.0f);
                int32_t q_i = static_cast<int32_t>(q_f);
                row_sum += q_i;
                uint8_t u2 = static_cast<uint8_t>(q_i + 2) & 0x3u;
                byte |= static_cast<uint8_t>(u2 << (s * 2));
            }
            B_u2[i * k_bytes + kb] = byte;
        }
        sum_b[i] = row_sum;
    }
}

// Reference: D[m][n] = clamp(bias[n] + recip_scale_a[m] * scale_b[n] *
//                              (Σ_k (q_a[m][k] - zp_a[m]) * lut[u2_b[n][k]]),
//                            min, max)
static void reference_gemm(size_t m, size_t n, size_t k, const int8_t *A_q,
                           const float *recip_scale_a, const int32_t *zp_a, const uint8_t *B_u2,
                           const float *scale_b, const int8_t *lut, const float *bias, float *D,
                           float clamp_min, float clamp_max)
{
    const size_t k_bytes = k / 4;
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            int32_t iacc = 0;
            for (size_t kb = 0; kb < k_bytes; ++kb) {
                uint8_t byte = B_u2[j * k_bytes + kb];
                for (size_t s = 0; s < 4; ++s) {
                    size_t kk = kb * 4 + s;
                    uint8_t u2 = static_cast<uint8_t>((byte >> (s * 2)) & 0x3u);
                    int32_t qa = A_q[i * k + kk];
                    int32_t qb = lut[u2];
                    iacc += (qa - zp_a[i]) * qb;
                }
            }
            float acc = recip_scale_a[i] * scale_b[j] * static_cast<float>(iacc);
            if (bias != nullptr)
                acc += bias[j];
            acc = std::fmax(acc, clamp_min);
            acc = std::fmin(acc, clamp_max);
            D[i * n + j] = acc;
        }
    }
}

static void verify(const float *ref, const float *act, size_t size, const GemmI8I2Params &p)
{
    float max_abs_err = 0.0f;
    size_t err_idx = 0;
    for (size_t i = 0; i < size; ++i) {
        float err = std::fabs(ref[i] - act[i]);
        if (err > max_abs_err) {
            max_abs_err = err;
            err_idx = i;
        }
    }
    EXPECT_LT(max_abs_err, 5e-3f) << "M=" << p.m << " N=" << p.n << " K=" << p.k
                                  << " has_bias=" << p.has_bias << " custom_lut=" << p.custom_lut
                                  << " | max err at [" << (err_idx / p.n) << "," << (err_idx % p.n)
                                  << "]: ref=" << ref[err_idx] << " act=" << act[err_idx];
}

static std::vector<int8_t> copy_a_window(const std::vector<int8_t> &src, size_t m, size_t k,
                                         size_t k_origin, size_t window)
{
    std::vector<int8_t> dst(m * window);
    for (size_t i = 0; i < m; ++i) {
        std::copy_n(src.data() + i * k + k_origin, window, dst.data() + i * window);
    }
    return dst;
}

static std::vector<uint8_t> copy_b_window(const std::vector<uint8_t> &src, size_t n, size_t k,
                                          size_t k_origin, size_t window)
{
    std::vector<uint8_t> dst(n * (window / 4));
    for (size_t i = 0; i < n; ++i) {
        std::copy_n(src.data() + i * (k / 4) + k_origin / 4, window / 4,
                    dst.data() + i * (window / 4));
    }
    return dst;
}

static std::vector<int32_t> sum_b_window(const std::vector<uint8_t> &src, size_t n, size_t window,
                                         const std::vector<int8_t> &lut)
{
    std::vector<int32_t> sums(n, 0);
    for (size_t i = 0; i < n; ++i) {
        for (size_t kb = 0; kb < window / 4; ++kb) {
            const uint8_t byte = src[i * (window / 4) + kb];
            for (size_t slot = 0; slot < 4; ++slot) {
                sums[i] += lut[(byte >> (slot * 2)) & 0x3u];
            }
        }
    }
    return sums;
}

// Generate inputs and quantize. Returns vectors and params struct ready for kernel use.
struct GeneratedData
{
    std::vector<int8_t> A_q;    // [M, K]
    std::vector<uint8_t> B_u2;  // [N, K/4]
    std::vector<float> recip_scale_a;
    std::vector<int32_t> zp_a;
    std::vector<float> scale_b;
    std::vector<int32_t> sum_b;
    std::vector<float> bias;
    std::vector<int8_t> lut;  // 4 entries
    float clamp_min;
    float clamp_max;
    std::vector<float> D_ref;
};

static GeneratedData generate_inputs(const GemmI8I2Params &p)
{
    std::mt19937 rng(0x4242 + static_cast<uint32_t>(p.m * 31 + p.n * 17 + p.k));
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> A_fp32(p.m * p.k);
    std::vector<float> B_fp32(p.n * p.k);
    for (auto &x : A_fp32)
        x = dist(rng);
    for (auto &x : B_fp32)
        x = dist(rng);

    GeneratedData g;
    g.A_q.resize(p.m * p.k);
    g.B_u2.resize(p.n * (p.k / 4));
    g.recip_scale_a.resize(p.m);
    g.zp_a.resize(p.m);
    g.scale_b.resize(p.n);
    g.sum_b.resize(p.n);

    quantize_a_qai8dx(p.m, p.k, A_fp32.data(), g.A_q.data(), g.recip_scale_a.data(), g.zp_a.data());
    quantize_b_qsi2cx(p.n, p.k, B_fp32.data(), g.B_u2.data(), g.scale_b.data(), g.sum_b.data());

    if (p.has_bias) {
        g.bias.resize(p.n);
        std::uniform_real_distribution<float> bias_dist(-0.5f, 0.5f);
        for (auto &x : g.bias)
            x = bias_dist(rng);
    }

    if (p.custom_lut) {
        // A non-monotonic permutation of the default LUT to exercise the vrgather path.
        g.lut = {-1, 1, -2, 0};
        // sum_b must match the active LUT: recompute Σ_k lut[u2[n,k]] for each row.
        const size_t k_bytes = p.k / 4;
        for (size_t i = 0; i < p.n; ++i) {
            int32_t s = 0;
            for (size_t kb = 0; kb < k_bytes; ++kb) {
                uint8_t byte = g.B_u2[i * k_bytes + kb];
                for (size_t slot = 0; slot < 4; ++slot) {
                    uint8_t u2 = static_cast<uint8_t>((byte >> (slot * 2)) & 0x3u);
                    s += g.lut[u2];
                }
            }
            g.sum_b[i] = s;
        }
    } else {
        g.lut.assign(tqt_qsi2_default_lut, tqt_qsi2_default_lut + 4);
    }

    g.clamp_min = p.has_clamp ? -0.5f : -std::numeric_limits<float>::max();
    g.clamp_max = p.has_clamp ? 0.5f : std::numeric_limits<float>::max();

    g.D_ref.resize(p.m * p.n);
    reference_gemm(p.m, p.n, p.k, g.A_q.data(), g.recip_scale_a.data(), g.zp_a.data(),
                   g.B_u2.data(), g.scale_b.data(), g.lut.data(),
                   p.has_bias ? g.bias.data() : nullptr, g.D_ref.data(), g.clamp_min, g.clamp_max);

    return g;
}

// ============================================================================
// Common test parameter set
// ============================================================================

static auto common_test_values()
{
    return testing::Values(GemmI8I2Params{1, 16, 32, false, false, false, ""},
                           GemmI8I2Params{4, 16, 32, false, false, false, ""},
                           GemmI8I2Params{4, 16, 32, true, false, false, ""},
                           GemmI8I2Params{4, 16, 32, true, true, false, ""},
                           GemmI8I2Params{1, 1, 32, false, false, true, "Tail"},
                           GemmI8I2Params{2, 3, 32, true, false, true, "Tail"},
                           GemmI8I2Params{3, 5, 32, true, true, true, "Tail"},
                           GemmI8I2Params{4, 16, 32, false, false, true, ""},
                           GemmI8I2Params{8, 32, 64, true, false, false, ""},
                           GemmI8I2Params{5, 19, 64, true, false, false, ""},
                           GemmI8I2Params{16, 64, 128, true, true, false, ""},
                           GemmI8I2Params{16, 64, 128, true, false, true, ""},
                           GemmI8I2Params{32, 128, 256, true, false, false, ""});
}

// ============================================================================
// gemm_1xnbias_clamp_f32_qai8dx_qsi2cxt_4x1vl_rvv  (a_bt)
// ============================================================================

class Test_gemm_1xnbias_clamp_f32_qai8dx_qsi2cxt_4x1vl_rvv
    : public testing::TestWithParam<GemmI8I2Params>
{
};

TEST_P(Test_gemm_1xnbias_clamp_f32_qai8dx_qsi2cxt_4x1vl_rvv, Correctness)
{
    const auto &p = GetParam();
    auto g = generate_inputs(p);

    tqt_qai8dx_qsi2cx_f32_params params{};
    params.scale_a = g.recip_scale_a.data();
    params.zero_point_a = g.zp_a.data();
    params.scale_b = g.scale_b.data();
    params.sum_b = g.sum_b.data();

    std::vector<float> D_act(p.m * p.n, 0.0f);
    torq_tile::test::clamp_test_run<tqt_run_gemm_1xnbias_clamp_f32_qai8dx_qsi2cxt_4x1vl_rvv, float,
                                    11, 9, 10, 12, torq_tile::test::ClampSource::kColumnBias, false,
                                    5, false>()(
        p.m, p.n, p.k, g.A_q.data(), p.k, 0, g.B_u2.data(), p.k, 0, D_act.data(), p.n,
        p.has_bias ? g.bias.data() : nullptr, g.clamp_min, g.clamp_max,
        p.custom_lut ? g.lut.data() : nullptr, &params);

    verify(g.D_ref.data(), D_act.data(), p.m * p.n, p);

    const size_t k_origin = (p.k / 8) * 4;
    const size_t window = p.k - k_origin;
    auto A_window = copy_a_window(g.A_q, p.m, p.k, k_origin, window);
    auto B_window = copy_b_window(g.B_u2, p.n, p.k, k_origin, window);
    auto window_sum_b = sum_b_window(B_window, p.n, window, g.lut);
    params.sum_b = window_sum_b.data();
    std::vector<float> D_zero(p.m * p.n);
    std::vector<float> D_nonzero(p.m * p.n);
    torq_tile::test::clamp_test_run<tqt_run_gemm_1xnbias_clamp_f32_qai8dx_qsi2cxt_4x1vl_rvv, float,
                                    11, 9, 10, 12, torq_tile::test::ClampSource::kColumnBias, false,
                                    5, false>()(
        p.m, p.n, window, A_window.data(), window, 0, B_window.data(), window, 0, D_zero.data(),
        p.n, nullptr, -std::numeric_limits<float>::max(), std::numeric_limits<float>::max(),
        p.custom_lut ? g.lut.data() : nullptr, &params);
    torq_tile::test::clamp_test_run<tqt_run_gemm_1xnbias_clamp_f32_qai8dx_qsi2cxt_4x1vl_rvv, float,
                                    11, 9, 10, 12, torq_tile::test::ClampSource::kColumnBias, false,
                                    5, false>()(
        p.m, p.n, window, g.A_q.data() + k_origin, p.k, k_origin, g.B_u2.data() + k_origin / 4, p.k,
        k_origin, D_nonzero.data(), p.n, nullptr, -std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max(), p.custom_lut ? g.lut.data() : nullptr, &params);
    verify(D_zero.data(), D_nonzero.data(), p.m * p.n, p);
}

TEST_P(Test_gemm_1xnbias_clamp_f32_qai8dx_qsi2cxt_4x1vl_rvv, ClampSpecialValues)
{
    torq_tile::test::ScopedGemmClampProbe probe;
    const auto &p = GetParam();
    auto g = generate_inputs(p);

    tqt_qai8dx_qsi2cx_f32_params params{};
    params.scale_a = g.recip_scale_a.data();
    params.zero_point_a = g.zp_a.data();
    params.scale_b = g.scale_b.data();
    params.sum_b = g.sum_b.data();

    std::vector<float> D_act(p.m * p.n, 0.0f);
    torq_tile::test::clamp_test_run<tqt_run_gemm_1xnbias_clamp_f32_qai8dx_qsi2cxt_4x1vl_rvv, float,
                                    11, 9, 10, 12, torq_tile::test::ClampSource::kColumnBias, false,
                                    5, false>()(
        p.m, p.n, p.k, g.A_q.data(), p.k, 0, g.B_u2.data(), p.k, 0, D_act.data(), p.n,
        p.has_bias ? g.bias.data() : nullptr, g.clamp_min, g.clamp_max,
        p.custom_lut ? g.lut.data() : nullptr, &params);

    verify(g.D_ref.data(), D_act.data(), p.m * p.n, p);

    const size_t k_origin = (p.k / 8) * 4;
    const size_t window = p.k - k_origin;
    auto A_window = copy_a_window(g.A_q, p.m, p.k, k_origin, window);
    auto B_window = copy_b_window(g.B_u2, p.n, p.k, k_origin, window);
    auto window_sum_b = sum_b_window(B_window, p.n, window, g.lut);
    params.sum_b = window_sum_b.data();
    std::vector<float> D_zero(p.m * p.n);
    std::vector<float> D_nonzero(p.m * p.n);
    torq_tile::test::clamp_test_run<tqt_run_gemm_1xnbias_clamp_f32_qai8dx_qsi2cxt_4x1vl_rvv, float,
                                    11, 9, 10, 12, torq_tile::test::ClampSource::kColumnBias, false,
                                    5, false>()(
        p.m, p.n, window, A_window.data(), window, 0, B_window.data(), window, 0, D_zero.data(),
        p.n, nullptr, -std::numeric_limits<float>::max(), std::numeric_limits<float>::max(),
        p.custom_lut ? g.lut.data() : nullptr, &params);
    torq_tile::test::clamp_test_run<tqt_run_gemm_1xnbias_clamp_f32_qai8dx_qsi2cxt_4x1vl_rvv, float,
                                    11, 9, 10, 12, torq_tile::test::ClampSource::kColumnBias, false,
                                    5, false>()(
        p.m, p.n, window, g.A_q.data() + k_origin, p.k, k_origin, g.B_u2.data() + k_origin / 4, p.k,
        k_origin, D_nonzero.data(), p.n, nullptr, -std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max(), p.custom_lut ? g.lut.data() : nullptr, &params);
    verify(D_zero.data(), D_nonzero.data(), p.m * p.n, p);
}

INSTANTIATE_TEST_SUITE_P(GemmTests, Test_gemm_1xnbias_clamp_f32_qai8dx_qsi2cxt_4x1vl_rvv,
                         common_test_values(), GemmI8I2ParamName());

// ============================================================================
// gemm_1xnbias_clamp_f32_qai8dxp_qsi2cxp_4x1vl_rvv  (ap_bp)
// ============================================================================

class Test_gemm_1xnbias_clamp_f32_qai8dxp_qsi2cxp_4x1vl_rvv
    : public testing::TestWithParam<GemmI8I2Params>
{
};

TEST_P(Test_gemm_1xnbias_clamp_f32_qai8dxp_qsi2cxp_4x1vl_rvv, Correctness)
{
    const auto &p = GetParam();
    auto g = generate_inputs(p);

    tqt_qai8dx_qsi2cx_f32_params params{};
    params.scale_a = g.recip_scale_a.data();
    params.zero_point_a = g.zp_a.data();
    params.scale_b = g.scale_b.data();
    params.sum_b = g.sum_b.data();

    // Pack A: [M, K] -> [M/4, K, 4]
    const size_t a_packed_size =
        tqt_get_a_packed_size_gemm_1xnbias_clamp_f32_qai8dxp_qsi2cxp_4x1vl_rvv(p.m, p.k);
    std::vector<int8_t> A_packed(a_packed_size);
    tqt_run_a_pack_gemm_1xnbias_clamp_f32_qai8dxp_qsi2cxp_4x1vl_rvv(p.m, p.k, p.k, p.k, 0,
                                                                    g.A_q.data(), A_packed.data());

    // Pack B (transposed): [N, K/4] -> [N/vl, K/4, vl]
    const size_t b_packed_size =
        tqt_get_b_packed_size_gemm_1xnbias_clamp_f32_qai8dxp_qsi2cxp_4x1vl_rvv(p.n, p.k);
    std::vector<uint8_t> B_packed(b_packed_size);
    tqt_run_bt_pack_gemm_1xnbias_clamp_f32_qai8dxp_qsi2cxp_4x1vl_rvv(
        p.n, p.k, p.k, p.k, 0, g.B_u2.data(), B_packed.data());

    std::vector<float> D_act(p.m * p.n, 0.0f);
    torq_tile::test::clamp_test_run<tqt_run_gemm_1xnbias_clamp_f32_qai8dxp_qsi2cxp_4x1vl_rvv, float,
                                    11, 9, 10, 12, torq_tile::test::ClampSource::kColumnBias, false,
                                    5, false>()(
        p.m, p.n, p.k, A_packed.data(), p.k, 0, B_packed.data(), p.k, 0, D_act.data(), p.n,
        p.has_bias ? g.bias.data() : nullptr, g.clamp_min, g.clamp_max,
        p.custom_lut ? g.lut.data() : nullptr, &params);

    verify(g.D_ref.data(), D_act.data(), p.m * p.n, p);

    const size_t k_origin = (p.k / 8) * 4;
    const size_t window = p.k - k_origin;
    auto A_window = copy_a_window(g.A_q, p.m, p.k, k_origin, window);
    auto B_window = copy_b_window(g.B_u2, p.n, p.k, k_origin, window);
    auto window_sum_b = sum_b_window(B_window, p.n, window, g.lut);
    params.sum_b = window_sum_b.data();

    std::vector<int8_t> A_window_packed(
        tqt_get_a_packed_size_gemm_1xnbias_clamp_f32_qai8dxp_qsi2cxp_4x1vl_rvv(p.m, window));
    std::vector<uint8_t> B_window_packed(
        tqt_get_b_packed_size_gemm_1xnbias_clamp_f32_qai8dxp_qsi2cxp_4x1vl_rvv(p.n, window));
    tqt_run_a_pack_gemm_1xnbias_clamp_f32_qai8dxp_qsi2cxp_4x1vl_rvv(
        p.m, window, window, window, 0, A_window.data(), A_window_packed.data());
    tqt_run_bt_pack_gemm_1xnbias_clamp_f32_qai8dxp_qsi2cxp_4x1vl_rvv(
        p.n, window, window, window, 0, B_window.data(), B_window_packed.data());

    const size_t a_offset =
        tqt_get_a_packed_offset_gemm_1xnbias_clamp_f32_qai8dxp_qsi2cxp_4x1vl_rvv(0, k_origin, p.k,
                                                                                 p.m);
    const size_t b_offset =
        tqt_get_b_packed_offset_gemm_1xnbias_clamp_f32_qai8dxp_qsi2cxp_4x1vl_rvv(0, k_origin, p.k,
                                                                                 p.n);
    std::vector<float> D_zero(p.m * p.n);
    std::vector<float> D_nonzero(p.m * p.n);
    torq_tile::test::clamp_test_run<tqt_run_gemm_1xnbias_clamp_f32_qai8dxp_qsi2cxp_4x1vl_rvv, float,
                                    11, 9, 10, 12, torq_tile::test::ClampSource::kColumnBias, false,
                                    5, false>()(
        p.m, p.n, window, A_window_packed.data(), window, 0, B_window_packed.data(), window, 0,
        D_zero.data(), p.n, nullptr, -std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max(), p.custom_lut ? g.lut.data() : nullptr, &params);
    torq_tile::test::clamp_test_run<tqt_run_gemm_1xnbias_clamp_f32_qai8dxp_qsi2cxp_4x1vl_rvv, float,
                                    11, 9, 10, 12, torq_tile::test::ClampSource::kColumnBias, false,
                                    5, false>()(
        p.m, p.n, window, A_packed.data() + a_offset, p.k, k_origin, B_packed.data() + b_offset,
        p.k, k_origin, D_nonzero.data(), p.n, nullptr, -std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max(), p.custom_lut ? g.lut.data() : nullptr, &params);
    verify(D_zero.data(), D_nonzero.data(), p.m * p.n, p);
}

TEST_P(Test_gemm_1xnbias_clamp_f32_qai8dxp_qsi2cxp_4x1vl_rvv, ClampSpecialValues)
{
    torq_tile::test::ScopedGemmClampProbe probe;
    const auto &p = GetParam();
    auto g = generate_inputs(p);

    tqt_qai8dx_qsi2cx_f32_params params{};
    params.scale_a = g.recip_scale_a.data();
    params.zero_point_a = g.zp_a.data();
    params.scale_b = g.scale_b.data();
    params.sum_b = g.sum_b.data();

    // Pack A: [M, K] -> [M/4, K, 4]
    const size_t a_packed_size =
        tqt_get_a_packed_size_gemm_1xnbias_clamp_f32_qai8dxp_qsi2cxp_4x1vl_rvv(p.m, p.k);
    std::vector<int8_t> A_packed(a_packed_size);
    tqt_run_a_pack_gemm_1xnbias_clamp_f32_qai8dxp_qsi2cxp_4x1vl_rvv(p.m, p.k, p.k, p.k, 0,
                                                                    g.A_q.data(), A_packed.data());

    // Pack B (transposed): [N, K/4] -> [N/vl, K/4, vl]
    const size_t b_packed_size =
        tqt_get_b_packed_size_gemm_1xnbias_clamp_f32_qai8dxp_qsi2cxp_4x1vl_rvv(p.n, p.k);
    std::vector<uint8_t> B_packed(b_packed_size);
    tqt_run_bt_pack_gemm_1xnbias_clamp_f32_qai8dxp_qsi2cxp_4x1vl_rvv(
        p.n, p.k, p.k, p.k, 0, g.B_u2.data(), B_packed.data());

    std::vector<float> D_act(p.m * p.n, 0.0f);
    torq_tile::test::clamp_test_run<tqt_run_gemm_1xnbias_clamp_f32_qai8dxp_qsi2cxp_4x1vl_rvv, float,
                                    11, 9, 10, 12, torq_tile::test::ClampSource::kColumnBias, false,
                                    5, false>()(
        p.m, p.n, p.k, A_packed.data(), p.k, 0, B_packed.data(), p.k, 0, D_act.data(), p.n,
        p.has_bias ? g.bias.data() : nullptr, g.clamp_min, g.clamp_max,
        p.custom_lut ? g.lut.data() : nullptr, &params);

    verify(g.D_ref.data(), D_act.data(), p.m * p.n, p);

    const size_t k_origin = (p.k / 8) * 4;
    const size_t window = p.k - k_origin;
    auto A_window = copy_a_window(g.A_q, p.m, p.k, k_origin, window);
    auto B_window = copy_b_window(g.B_u2, p.n, p.k, k_origin, window);
    auto window_sum_b = sum_b_window(B_window, p.n, window, g.lut);
    params.sum_b = window_sum_b.data();

    std::vector<int8_t> A_window_packed(
        tqt_get_a_packed_size_gemm_1xnbias_clamp_f32_qai8dxp_qsi2cxp_4x1vl_rvv(p.m, window));
    std::vector<uint8_t> B_window_packed(
        tqt_get_b_packed_size_gemm_1xnbias_clamp_f32_qai8dxp_qsi2cxp_4x1vl_rvv(p.n, window));
    tqt_run_a_pack_gemm_1xnbias_clamp_f32_qai8dxp_qsi2cxp_4x1vl_rvv(
        p.m, window, window, window, 0, A_window.data(), A_window_packed.data());
    tqt_run_bt_pack_gemm_1xnbias_clamp_f32_qai8dxp_qsi2cxp_4x1vl_rvv(
        p.n, window, window, window, 0, B_window.data(), B_window_packed.data());

    const size_t a_offset =
        tqt_get_a_packed_offset_gemm_1xnbias_clamp_f32_qai8dxp_qsi2cxp_4x1vl_rvv(0, k_origin, p.k,
                                                                                 p.m);
    const size_t b_offset =
        tqt_get_b_packed_offset_gemm_1xnbias_clamp_f32_qai8dxp_qsi2cxp_4x1vl_rvv(0, k_origin, p.k,
                                                                                 p.n);
    std::vector<float> D_zero(p.m * p.n);
    std::vector<float> D_nonzero(p.m * p.n);
    torq_tile::test::clamp_test_run<tqt_run_gemm_1xnbias_clamp_f32_qai8dxp_qsi2cxp_4x1vl_rvv, float,
                                    11, 9, 10, 12, torq_tile::test::ClampSource::kColumnBias, false,
                                    5, false>()(
        p.m, p.n, window, A_window_packed.data(), window, 0, B_window_packed.data(), window, 0,
        D_zero.data(), p.n, nullptr, -std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max(), p.custom_lut ? g.lut.data() : nullptr, &params);
    torq_tile::test::clamp_test_run<tqt_run_gemm_1xnbias_clamp_f32_qai8dxp_qsi2cxp_4x1vl_rvv, float,
                                    11, 9, 10, 12, torq_tile::test::ClampSource::kColumnBias, false,
                                    5, false>()(
        p.m, p.n, window, A_packed.data() + a_offset, p.k, k_origin, B_packed.data() + b_offset,
        p.k, k_origin, D_nonzero.data(), p.n, nullptr, -std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max(), p.custom_lut ? g.lut.data() : nullptr, &params);
    verify(D_zero.data(), D_nonzero.data(), p.m * p.n, p);
}

INSTANTIATE_TEST_SUITE_P(GemmTests, Test_gemm_1xnbias_clamp_f32_qai8dxp_qsi2cxp_4x1vl_rvv,
                         common_test_values(), GemmI8I2ParamName());

}  // namespace
