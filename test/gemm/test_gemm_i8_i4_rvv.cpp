//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "src/gemm/gemm_i8_i4/gemm_1xnbias_clamp_f32_qai8dx_qsi4cxt/tqt_gemm_1xnbias_clamp_f32_qai8dx_qsi4cxt_4x1vl_rvv.h"
#include "src/gemm/gemm_i8_i4/gemm_1xnbias_clamp_f32_qai8dxp_qsi4cxp/tqt_gemm_1xnbias_clamp_f32_qai8dxp_qsi4cxp_4x1vl_rvv.h"
#include "src/gemm/gemm_i8_i4/tqt_params_i8_i4.h"
#if defined(__riscv_zfh) && defined(__riscv_zvfh)
#include "src/gemm/gemm_i8_i4/gemm_1xnbias_clamp_f16_qai8dx_qsi4cxt/tqt_gemm_1xnbias_clamp_f16_qai8dx_qsi4cxt_4x1vl_rvv.h"
#include "src/gemm/gemm_i8_i4/gemm_1xnbias_clamp_f16_qai8dxp_qsi4cxp/tqt_gemm_1xnbias_clamp_f16_qai8dxp_qsi4cxp_4x1vl_rvv.h"
#endif  // __riscv_zfh && __riscv_zvfh
#include "test/common/data_generator.h"
#include "test/common/gemm_test_runner.h"
#include "test/common/quantize_utils.h"
#include "test/common/test_common.h"
#include "test/common/verify.h"

using namespace torq_tile::test;

// ============================================================================
// W4A8 quantization helpers
// ============================================================================

namespace
{

/// Per-row asymmetric int8 quantization for matrix A [m, k].
/// Computes scale_a[m] and zero_point_a[m], then quantizes to A_i8.
inline void quantize_a_per_row_asym(const float *A_fp32, size_t m, size_t k, int8_t *A_i8,
                                    float *scale_a, int32_t *zp_a)
{
    for (size_t row = 0; row < m; ++row) {
        const float *row_ptr = A_fp32 + row * k;
        float fmin = FLT_MAX;
        float fmax = -FLT_MAX;
        for (size_t i = 0; i < k; ++i) {
            if (row_ptr[i] < fmin)
                fmin = row_ptr[i];
            if (row_ptr[i] > fmax)
                fmax = row_ptr[i];
        }
        fmax = fmax > 0.0f ? fmax : 0.0f;
        fmin = fmin < 0.0f ? fmin : 0.0f;

        float s = (fmax - fmin) / 255.0f;
        if (s == 0.0f) {
            scale_a[row] = 1.0f;
            zp_a[row] = 0;
            for (size_t i = 0; i < k; ++i) {
                A_i8[row * k + i] = 0;
            }
            continue;
        }
        float zp_tmp = -128.0f - fmin / s;
        int32_t zp = static_cast<int32_t>(std::nearbyintf(zp_tmp));
        if (zp < -128)
            zp = -128;
        if (zp > 127)
            zp = 127;
        scale_a[row] = s;
        zp_a[row] = zp;
        for (size_t i = 0; i < k; ++i) {
            A_i8[row * k + i] = quantize_i8(row_ptr[i], s, zp);
        }
    }
}

/// Per-channel symmetric int4 quantization for transposed B [n, k].
/// Computes scale_b[n] (one per N row), and packs int4 into uint8 (2 nibbles/byte).
/// Encoding: signed [-8, 7] stored as unsigned [0, 15] (ZP=8 offset).
inline void quantize_b_per_channel_sym_int4(const float *B_fp32, size_t n, size_t k,
                                            uint8_t *B_packed, float *scale_b)
{
    for (size_t row = 0; row < n; ++row) {
        const float *row_ptr = B_fp32 + row * k;
        float abs_max = 0.0f;
        for (size_t i = 0; i < k; ++i) {
            float v = std::fabsf(row_ptr[i]);
            if (v > abs_max)
                abs_max = v;
        }
        // Symmetric int4 in [-8, 7]; scale = abs_max / 8 (max signed magnitude).
        // Mirror KleidiAI/W4A16 convention: values in [-8, 7] cover ±abs_max approximately.
        float s = abs_max / 8.0f;
        if (s == 0.0f)
            s = 1.0f;
        scale_b[row] = s;

        for (size_t kp = 0; kp < k / 2; ++kp) {
            float v0 = row_ptr[2 * kp + 0];
            float v1 = row_ptr[2 * kp + 1];
            int32_t q0 = static_cast<int32_t>(std::nearbyintf(v0 / s));
            int32_t q1 = static_cast<int32_t>(std::nearbyintf(v1 / s));
            if (q0 < -8)
                q0 = -8;
            if (q0 > 7)
                q0 = 7;
            if (q1 < -8)
                q1 = -8;
            if (q1 > 7)
                q1 = 7;
            uint8_t lo = static_cast<uint8_t>(q0 + 8);
            uint8_t hi = static_cast<uint8_t>(q1 + 8);
            B_packed[row * (k / 2) + kp] = lo | (hi << 4);
        }
    }
}

/// W4A8 reference computation in fp32.
/// D[m,n] = clamp(C[m,n] + dequant(A_i8) @ dequant(B_i4)^T + bias[n], min, max)
inline void w4a8_reference_fp32(size_t m, size_t n, size_t k, const int8_t *A_i8,
                                const uint8_t *B_packed, const float *scale_a, const int32_t *zp_a,
                                const float *scale_b, const float *C, const float *bias, float *D,
                                float clamp_min, float clamp_max)
{
    for (size_t mi = 0; mi < m; ++mi) {
        for (size_t ni = 0; ni < n; ++ni) {
            int32_t acc = 0;
            int32_t b_sum = 0;
            for (size_t kp = 0; kp < k / 2; ++kp) {
                uint8_t byte = B_packed[ni * (k / 2) + kp];
                int32_t b_lo = (int32_t)(byte & 0xf) - 8;
                int32_t b_hi = (int32_t)(byte >> 4) - 8;
                int32_t a_lo = (int32_t)A_i8[mi * k + 2 * kp + 0];
                int32_t a_hi = (int32_t)A_i8[mi * k + 2 * kp + 1];
                acc += a_lo * b_lo + a_hi * b_hi;
                b_sum += b_lo + b_hi;
            }
            // Subtract zp_a[mi] * sum_k(b_signed[k, ni])
            acc -= zp_a[mi] * b_sum;
            float result = static_cast<float>(acc) * scale_a[mi] * scale_b[ni];
            if (C) {
                result += C[mi * n + ni];
            }
            if (bias) {
                result += bias[ni];
            }
            if (result < clamp_min)
                result = clamp_min;
            if (result > clamp_max)
                result = clamp_max;
            D[mi * n + ni] = result;
        }
    }
}

}  // namespace

// ============================================================================
// Common test values
// ============================================================================

static auto common_w4a8_test_values()
{
    return testing::Values(
        GemmTestParams(8, 8, 8, "Basic"), GemmTestParams(16, 16, 16, "Small"),
        GemmTestParams(32, 32, 32, "Medium"), GemmTestParams(5, 11, 16, "NonAligned"),
        GemmTestParams(7, 3, 8, "SmallNonAligned"), GemmTestParams(17, 19, 24, "Odd"),
        GemmTestParams(31, 35, 64, "Large"), GemmTestParams(60, 61, 128, "LargerK"),
        GemmTestParams(32, 32, 32, "WithBias").set_bias(true),
        GemmTestParams(17, 19, 24, "WithBias_Odd").set_bias(true),
        GemmTestParams(32, 32, 32, "WithClamp").set_clamp(-0.5f, 0.5f),
        GemmTestParams(32, 32, 32, "WithC").set_c(true),
        GemmTestParams(32, 32, 32, "WithC_Bias_Clamp")
            .set_c(true)
            .set_bias(true)
            .set_clamp(-0.5f, 0.5f));
}

// ============================================================================
// Test runner — non-packed (qai8dx × qsi4cxt)
// ============================================================================

template <typename TD>
struct W4A8GemmFuncs
{
    using GetMStepFunc = size_t (*)(void);
    using GetNStepFunc = size_t (*)(void);
    using RunGemmFunc = void (*)(size_t m, size_t n, size_t k, const void *A, size_t lda,
                                 size_t k_idx_a, const void *B, size_t ldb, size_t k_idx_b,
                                 const void *C, size_t ldc, void *D, size_t ldd, const void *bias,
                                 TD clamp_min, TD clamp_max,
                                 const struct tqt_qai8dx_qsi4cx_params *params);

    GetMStepFunc get_m_step;
    GetNStepFunc get_n_step;
    RunGemmFunc run_gemm;
};

template <typename TD>
struct W4A8PackedGemmFuncs
{
    using GetMStepFunc = size_t (*)(void);
    using GetNStepFunc = size_t (*)(void);
    using GetAPackedSizeFunc = size_t (*)(size_t m, size_t k);
    using GetBPackedSizeFunc = size_t (*)(size_t n, size_t k);
    using RunAPackFunc = void (*)(size_t m, size_t k, size_t lda, size_t lda_packed, size_t k_idx,
                                  const void *A, void *A_packed);
    using RunBtPackFunc = void (*)(size_t n, size_t k, size_t ldb, size_t ldb_packed, size_t k_idx,
                                   const void *B, void *B_packed);
    using RunGemmFunc = void (*)(size_t m, size_t n, size_t k, const void *A_packed, size_t lda,
                                 size_t k_idx_a, const void *B_packed, size_t ldb, size_t k_idx_b,
                                 const void *C, size_t ldc, void *D, size_t ldd, const void *bias,
                                 TD clamp_min, TD clamp_max,
                                 const struct tqt_qai8dx_qsi4cx_params *params);

    GetMStepFunc get_m_step;
    GetNStepFunc get_n_step;
    GetAPackedSizeFunc get_a_packed_size;
    GetBPackedSizeFunc get_b_packed_size;
    RunAPackFunc run_a_pack;
    RunBtPackFunc run_bt_pack;
    RunGemmFunc run_gemm;
};

template <typename TD>
static void run_w4a8_gemm_test(const GemmTestParams &params, const W4A8GemmFuncs<TD> &funcs)
{
    const size_t m = params.m;
    const size_t n = params.n;
    const size_t k = params.k;
    const size_t lda = k;
    const size_t ldb = k;
    const size_t ldc = n;
    const size_t ldd = n;

    UniformRandomGenerator<float> gen(-1.0f, 1.0f, 42);
    std::vector<float> A_fp32(m * k);
    std::vector<float> B_fp32(n * k);
    gen.fill_matrix(A_fp32.data(), m, k);
    gen.fill_matrix(B_fp32.data(), n, k);

    // Quantize A per-row asymmetric int8 and B per-channel symmetric int4.
    std::vector<int8_t> A_i8(m * k);
    std::vector<float> scale_a(m);
    std::vector<int32_t> zp_a(m);
    quantize_a_per_row_asym(A_fp32.data(), m, k, A_i8.data(), scale_a.data(), zp_a.data());

    std::vector<uint8_t> B_packed(n * (k / 2));
    std::vector<float> scale_b(n);
    quantize_b_per_channel_sym_int4(B_fp32.data(), n, k, B_packed.data(), scale_b.data());

    // C [m, n] in TD type (optional)
    std::vector<TD> C_data;
    if (params.has_c) {
        C_data.resize(m * n);
        UniformRandomGenerator<TD> c_gen(-2.0f, 2.0f, 42);
        c_gen.fill_matrix(C_data.data(), m, n);
    }
    // Bias [n] in TD type (optional)
    std::vector<TD> bias_data;
    if (params.has_bias) {
        bias_data.resize(n);
        UniformRandomGenerator<TD> bias_gen(-0.5f, 0.5f, 7);
        bias_gen.fill_matrix(bias_data.data(), 1, n);
    }

    const float clamp_min_f = params.clamp_min_f;
    const float clamp_max_f = params.clamp_max_f;
    const TD clamp_min_td = to_target_type<TD>(clamp_min_f);
    const TD clamp_max_td = to_target_type<TD>(clamp_max_f);

    // Reference (fp32 from quantized inputs).
    std::vector<float> C_fp32;
    std::vector<float> bias_fp32;
    if (params.has_c) {
        C_fp32.resize(m * n);
        for (size_t i = 0; i < m * n; ++i) {
            C_fp32[i] = to_float(C_data[i]);
        }
    }
    if (params.has_bias) {
        bias_fp32.resize(n);
        for (size_t i = 0; i < n; ++i) {
            bias_fp32[i] = to_float(bias_data[i]);
        }
    }

    std::vector<float> D_ref_fp32(m * n);
    w4a8_reference_fp32(m, n, k, A_i8.data(), B_packed.data(), scale_a.data(), zp_a.data(),
                        scale_b.data(), params.has_c ? C_fp32.data() : nullptr,
                        params.has_bias ? bias_fp32.data() : nullptr, D_ref_fp32.data(),
                        clamp_min_f, clamp_max_f);
    // Round reference through TD precision for fair comparison with kernel output.
    std::vector<TD> D_ref(m * n);
    for (size_t i = 0; i < m * n; ++i) {
        D_ref[i] = to_target_type<TD>(D_ref_fp32[i]);
    }

    // Set up params struct.
    struct tqt_qai8dx_qsi4cx_params qparams;
    qparams.scale_a = scale_a.data();
    qparams.zero_point_a = zp_a.data();
    qparams.scale_b = scale_b.data();

    // Run kernel.
    std::vector<TD> D_act(m * n, to_target_type<TD>(0.0f));
    funcs.run_gemm(m, n, k, A_i8.data(), lda, 0, B_packed.data(), ldb, 0,
                   params.has_c ? C_data.data() : nullptr, ldc, D_act.data(), ldd,
                   params.has_bias ? bias_data.data() : nullptr, clamp_min_td, clamp_max_td,
                   &qparams);

    // Verify (use TD's tolerance — same precision as output).
    auto result =
        verify_gemm_result<TD>(D_ref.data(), D_act.data(), m * n, DefaultThreshold<TD>::abs_error,
                               DefaultThreshold<TD>::cosine_sim);
    EXPECT_TRUE(result.passed) << result.error_message << "\n  Shape: M=" << m << " N=" << n
                               << " K=" << k << "\n  Config: " << params.name
                               << "\n  Max abs error: " << result.max_abs_error
                               << "\n  Cosine similarity: " << result.cosine_similarity;
}

template <typename TD>
static void run_w4a8_packed_gemm_test(const GemmTestParams &params,
                                      const W4A8PackedGemmFuncs<TD> &funcs)
{
    const size_t m = params.m;
    const size_t n = params.n;
    const size_t k = params.k;
    const size_t lda = k;
    const size_t ldb = k;
    const size_t ldc = n;
    const size_t ldd = n;

    UniformRandomGenerator<float> gen(-1.0f, 1.0f, 42);
    std::vector<float> A_fp32(m * k);
    std::vector<float> B_fp32(n * k);
    gen.fill_matrix(A_fp32.data(), m, k);
    gen.fill_matrix(B_fp32.data(), n, k);

    std::vector<int8_t> A_i8(m * k);
    std::vector<float> scale_a(m);
    std::vector<int32_t> zp_a(m);
    quantize_a_per_row_asym(A_fp32.data(), m, k, A_i8.data(), scale_a.data(), zp_a.data());

    std::vector<uint8_t> B_int4(n * (k / 2));
    std::vector<float> scale_b(n);
    quantize_b_per_channel_sym_int4(B_fp32.data(), n, k, B_int4.data(), scale_b.data());

    std::vector<TD> C_data;
    if (params.has_c) {
        C_data.resize(m * n);
        UniformRandomGenerator<TD> c_gen(-2.0f, 2.0f, 42);
        c_gen.fill_matrix(C_data.data(), m, n);
    }
    std::vector<TD> bias_data;
    if (params.has_bias) {
        bias_data.resize(n);
        UniformRandomGenerator<TD> bias_gen(-0.5f, 0.5f, 7);
        bias_gen.fill_matrix(bias_data.data(), 1, n);
    }

    const float clamp_min_f = params.clamp_min_f;
    const float clamp_max_f = params.clamp_max_f;
    const TD clamp_min_td = to_target_type<TD>(clamp_min_f);
    const TD clamp_max_td = to_target_type<TD>(clamp_max_f);

    std::vector<float> C_fp32;
    std::vector<float> bias_fp32;
    if (params.has_c) {
        C_fp32.resize(m * n);
        for (size_t i = 0; i < m * n; ++i) {
            C_fp32[i] = to_float(C_data[i]);
        }
    }
    if (params.has_bias) {
        bias_fp32.resize(n);
        for (size_t i = 0; i < n; ++i) {
            bias_fp32[i] = to_float(bias_data[i]);
        }
    }

    std::vector<float> D_ref_fp32(m * n);
    w4a8_reference_fp32(m, n, k, A_i8.data(), B_int4.data(), scale_a.data(), zp_a.data(),
                        scale_b.data(), params.has_c ? C_fp32.data() : nullptr,
                        params.has_bias ? bias_fp32.data() : nullptr, D_ref_fp32.data(),
                        clamp_min_f, clamp_max_f);
    std::vector<TD> D_ref(m * n);
    for (size_t i = 0; i < m * n; ++i) {
        D_ref[i] = to_target_type<TD>(D_ref_fp32[i]);
    }

    struct tqt_qai8dx_qsi4cx_params qparams;
    qparams.scale_a = scale_a.data();
    qparams.zero_point_a = zp_a.data();
    qparams.scale_b = scale_b.data();

    // Pack A and B.
    const size_t a_packed_size = funcs.get_a_packed_size(m, k);
    const size_t b_packed_size = funcs.get_b_packed_size(n, k);
    std::vector<uint8_t> A_packed(a_packed_size);
    std::vector<uint8_t> B_packed(b_packed_size);

    funcs.run_a_pack(m, k, lda, k, 0, A_i8.data(), A_packed.data());
    funcs.run_bt_pack(n, k, ldb, k, 0, B_int4.data(), B_packed.data());

    std::vector<TD> D_act(m * n, to_target_type<TD>(0.0f));
    funcs.run_gemm(m, n, k, A_packed.data(), lda, 0, B_packed.data(), ldb, 0,
                   params.has_c ? C_data.data() : nullptr, ldc, D_act.data(), ldd,
                   params.has_bias ? bias_data.data() : nullptr, clamp_min_td, clamp_max_td,
                   &qparams);

    auto result =
        verify_gemm_result<TD>(D_ref.data(), D_act.data(), m * n, DefaultThreshold<TD>::abs_error,
                               DefaultThreshold<TD>::cosine_sim);
    EXPECT_TRUE(result.passed) << result.error_message << "\n  Shape: M=" << m << " N=" << n
                               << " K=" << k << "\n  Config: " << params.name
                               << "\n  Max abs error: " << result.max_abs_error
                               << "\n  Cosine similarity: " << result.cosine_similarity;
}

#if defined(__riscv_zfh) && defined(__riscv_zvfh)
// ============================================================================
// gemm_1xnbias_clamp_f16_qai8dx_qsi4cxt_4x1vl_rvv (D=fp16, a_bt)
// ============================================================================

static W4A8GemmFuncs<float16_t> make_gemm_1xnbias_clamp_f16_qai8dx_qsi4cxt_4x1vl_rvv_funcs()
{
    W4A8GemmFuncs<float16_t> funcs;
    funcs.get_m_step = tqt_get_m_step_gemm_1xnbias_clamp_f16_qai8dx_qsi4cxt_4x1vl_rvv;
    funcs.get_n_step = tqt_get_n_step_gemm_1xnbias_clamp_f16_qai8dx_qsi4cxt_4x1vl_rvv;
    funcs.run_gemm = tqt_run_gemm_1xnbias_clamp_f16_qai8dx_qsi4cxt_4x1vl_rvv;
    return funcs;
}

class Test_gemm_1xnbias_clamp_f16_qai8dx_qsi4cxt_4x1vl_rvv
    : public testing::TestWithParam<GemmTestParams>
{
};

TEST_P(Test_gemm_1xnbias_clamp_f16_qai8dx_qsi4cxt_4x1vl_rvv, Correctness)
{
    auto funcs = make_gemm_1xnbias_clamp_f16_qai8dx_qsi4cxt_4x1vl_rvv_funcs();
    run_w4a8_gemm_test<float16_t>(GetParam(), funcs);
}

INSTANTIATE_TEST_SUITE_P(GemmTests, Test_gemm_1xnbias_clamp_f16_qai8dx_qsi4cxt_4x1vl_rvv,
                         common_w4a8_test_values(), GemmTestParamNameGenerator());

// ============================================================================
// gemm_1xnbias_clamp_f16_qai8dxp_qsi4cxp_4x1vl_rvv (D=fp16, ap_bp)
// ============================================================================

static W4A8PackedGemmFuncs<float16_t> make_gemm_1xnbias_clamp_f16_qai8dxp_qsi4cxp_4x1vl_rvv_funcs()
{
    W4A8PackedGemmFuncs<float16_t> funcs;
    funcs.get_m_step = tqt_get_m_step_gemm_1xnbias_clamp_f16_qai8dxp_qsi4cxp_4x1vl_rvv;
    funcs.get_n_step = tqt_get_n_step_gemm_1xnbias_clamp_f16_qai8dxp_qsi4cxp_4x1vl_rvv;
    funcs.get_a_packed_size =
        tqt_get_a_packed_size_gemm_1xnbias_clamp_f16_qai8dxp_qsi4cxp_4x1vl_rvv;
    funcs.get_b_packed_size =
        tqt_get_b_packed_size_gemm_1xnbias_clamp_f16_qai8dxp_qsi4cxp_4x1vl_rvv;
    funcs.run_a_pack = tqt_run_a_pack_gemm_1xnbias_clamp_f16_qai8dxp_qsi4cxp_4x1vl_rvv;
    funcs.run_bt_pack = tqt_run_bt_pack_gemm_1xnbias_clamp_f16_qai8dxp_qsi4cxp_4x1vl_rvv;
    funcs.run_gemm = tqt_run_gemm_1xnbias_clamp_f16_qai8dxp_qsi4cxp_4x1vl_rvv;
    return funcs;
}

class Test_gemm_1xnbias_clamp_f16_qai8dxp_qsi4cxp_4x1vl_rvv
    : public testing::TestWithParam<GemmTestParams>
{
};

TEST_P(Test_gemm_1xnbias_clamp_f16_qai8dxp_qsi4cxp_4x1vl_rvv, Correctness)
{
    auto funcs = make_gemm_1xnbias_clamp_f16_qai8dxp_qsi4cxp_4x1vl_rvv_funcs();
    run_w4a8_packed_gemm_test<float16_t>(GetParam(), funcs);
}

INSTANTIATE_TEST_SUITE_P(GemmTests, Test_gemm_1xnbias_clamp_f16_qai8dxp_qsi4cxp_4x1vl_rvv,
                         common_w4a8_test_values(), GemmTestParamNameGenerator());
#endif  // __riscv_zfh && __riscv_zvfh

// ============================================================================
// gemm_1xnbias_clamp_f32_qai8dx_qsi4cxt_4x1vl_rvv (D=fp32, a_bt)
// ============================================================================

static W4A8GemmFuncs<float> make_gemm_1xnbias_clamp_f32_qai8dx_qsi4cxt_4x1vl_rvv_funcs()
{
    W4A8GemmFuncs<float> funcs;
    funcs.get_m_step = tqt_get_m_step_gemm_1xnbias_clamp_f32_qai8dx_qsi4cxt_4x1vl_rvv;
    funcs.get_n_step = tqt_get_n_step_gemm_1xnbias_clamp_f32_qai8dx_qsi4cxt_4x1vl_rvv;
    funcs.run_gemm = tqt_run_gemm_1xnbias_clamp_f32_qai8dx_qsi4cxt_4x1vl_rvv;
    return funcs;
}

class Test_gemm_1xnbias_clamp_f32_qai8dx_qsi4cxt_4x1vl_rvv
    : public testing::TestWithParam<GemmTestParams>
{
};

TEST_P(Test_gemm_1xnbias_clamp_f32_qai8dx_qsi4cxt_4x1vl_rvv, Correctness)
{
    auto funcs = make_gemm_1xnbias_clamp_f32_qai8dx_qsi4cxt_4x1vl_rvv_funcs();
    run_w4a8_gemm_test<float>(GetParam(), funcs);
}

INSTANTIATE_TEST_SUITE_P(GemmTests, Test_gemm_1xnbias_clamp_f32_qai8dx_qsi4cxt_4x1vl_rvv,
                         common_w4a8_test_values(), GemmTestParamNameGenerator());

// ============================================================================
// gemm_1xnbias_clamp_f32_qai8dxp_qsi4cxp_4x1vl_rvv (D=fp32, ap_bp)
// ============================================================================

static W4A8PackedGemmFuncs<float> make_gemm_1xnbias_clamp_f32_qai8dxp_qsi4cxp_4x1vl_rvv_funcs()
{
    W4A8PackedGemmFuncs<float> funcs;
    funcs.get_m_step = tqt_get_m_step_gemm_1xnbias_clamp_f32_qai8dxp_qsi4cxp_4x1vl_rvv;
    funcs.get_n_step = tqt_get_n_step_gemm_1xnbias_clamp_f32_qai8dxp_qsi4cxp_4x1vl_rvv;
    funcs.get_a_packed_size =
        tqt_get_a_packed_size_gemm_1xnbias_clamp_f32_qai8dxp_qsi4cxp_4x1vl_rvv;
    funcs.get_b_packed_size =
        tqt_get_b_packed_size_gemm_1xnbias_clamp_f32_qai8dxp_qsi4cxp_4x1vl_rvv;
    funcs.run_a_pack = tqt_run_a_pack_gemm_1xnbias_clamp_f32_qai8dxp_qsi4cxp_4x1vl_rvv;
    funcs.run_bt_pack = tqt_run_bt_pack_gemm_1xnbias_clamp_f32_qai8dxp_qsi4cxp_4x1vl_rvv;
    funcs.run_gemm = tqt_run_gemm_1xnbias_clamp_f32_qai8dxp_qsi4cxp_4x1vl_rvv;
    return funcs;
}

class Test_gemm_1xnbias_clamp_f32_qai8dxp_qsi4cxp_4x1vl_rvv
    : public testing::TestWithParam<GemmTestParams>
{
};

TEST_P(Test_gemm_1xnbias_clamp_f32_qai8dxp_qsi4cxp_4x1vl_rvv, Correctness)
{
    auto funcs = make_gemm_1xnbias_clamp_f32_qai8dxp_qsi4cxp_4x1vl_rvv_funcs();
    run_w4a8_packed_gemm_test<float>(GetParam(), funcs);
}

INSTANTIATE_TEST_SUITE_P(GemmTests, Test_gemm_1xnbias_clamp_f32_qai8dxp_qsi4cxp_4x1vl_rvv,
                         common_w4a8_test_values(), GemmTestParamNameGenerator());

// ============================================================================
// BF16 output variants (require zvfbfwma).
// Compiled into this TU only when the toolchain enables Zvfbfwma.
// ============================================================================

#if defined(__riscv_zvfbfwma)
#include "src/gemm/gemm_i8_i4/gemm_1xnbias_clamp_bf16_qai8dx_qsi4cxt/tqt_gemm_1xnbias_clamp_bf16_qai8dx_qsi4cxt_4x1vl_rvv.h"
#include "src/gemm/gemm_i8_i4/gemm_1xnbias_clamp_bf16_qai8dxp_qsi4cxp/tqt_gemm_1xnbias_clamp_bf16_qai8dxp_qsi4cxp_4x1vl_rvv.h"

// ============================================================================
// gemm_1xnbias_clamp_bf16_qai8dx_qsi4cxt_4x1vl_rvv (D=bf16, a_bt)
// ============================================================================

static W4A8GemmFuncs<bfloat16_t> make_gemm_1xnbias_clamp_bf16_qai8dx_qsi4cxt_4x1vl_rvv_funcs()
{
    W4A8GemmFuncs<bfloat16_t> funcs;
    funcs.get_m_step = tqt_get_m_step_gemm_1xnbias_clamp_bf16_qai8dx_qsi4cxt_4x1vl_rvv;
    funcs.get_n_step = tqt_get_n_step_gemm_1xnbias_clamp_bf16_qai8dx_qsi4cxt_4x1vl_rvv;
    funcs.run_gemm = tqt_run_gemm_1xnbias_clamp_bf16_qai8dx_qsi4cxt_4x1vl_rvv;
    return funcs;
}

class Test_gemm_1xnbias_clamp_bf16_qai8dx_qsi4cxt_4x1vl_rvv
    : public testing::TestWithParam<GemmTestParams>
{
};

TEST_P(Test_gemm_1xnbias_clamp_bf16_qai8dx_qsi4cxt_4x1vl_rvv, Correctness)
{
    auto funcs = make_gemm_1xnbias_clamp_bf16_qai8dx_qsi4cxt_4x1vl_rvv_funcs();
    run_w4a8_gemm_test<bfloat16_t>(GetParam(), funcs);
}

INSTANTIATE_TEST_SUITE_P(GemmTests, Test_gemm_1xnbias_clamp_bf16_qai8dx_qsi4cxt_4x1vl_rvv,
                         common_w4a8_test_values(), GemmTestParamNameGenerator());

// ============================================================================
// gemm_1xnbias_clamp_bf16_qai8dxp_qsi4cxp_4x1vl_rvv (D=bf16, ap_bp)
// ============================================================================

static W4A8PackedGemmFuncs<bfloat16_t>
make_gemm_1xnbias_clamp_bf16_qai8dxp_qsi4cxp_4x1vl_rvv_funcs()
{
    W4A8PackedGemmFuncs<bfloat16_t> funcs;
    funcs.get_m_step = tqt_get_m_step_gemm_1xnbias_clamp_bf16_qai8dxp_qsi4cxp_4x1vl_rvv;
    funcs.get_n_step = tqt_get_n_step_gemm_1xnbias_clamp_bf16_qai8dxp_qsi4cxp_4x1vl_rvv;
    funcs.get_a_packed_size =
        tqt_get_a_packed_size_gemm_1xnbias_clamp_bf16_qai8dxp_qsi4cxp_4x1vl_rvv;
    funcs.get_b_packed_size =
        tqt_get_b_packed_size_gemm_1xnbias_clamp_bf16_qai8dxp_qsi4cxp_4x1vl_rvv;
    funcs.run_a_pack = tqt_run_a_pack_gemm_1xnbias_clamp_bf16_qai8dxp_qsi4cxp_4x1vl_rvv;
    funcs.run_bt_pack = tqt_run_bt_pack_gemm_1xnbias_clamp_bf16_qai8dxp_qsi4cxp_4x1vl_rvv;
    funcs.run_gemm = tqt_run_gemm_1xnbias_clamp_bf16_qai8dxp_qsi4cxp_4x1vl_rvv;
    return funcs;
}

class Test_gemm_1xnbias_clamp_bf16_qai8dxp_qsi4cxp_4x1vl_rvv
    : public testing::TestWithParam<GemmTestParams>
{
};

TEST_P(Test_gemm_1xnbias_clamp_bf16_qai8dxp_qsi4cxp_4x1vl_rvv, Correctness)
{
    auto funcs = make_gemm_1xnbias_clamp_bf16_qai8dxp_qsi4cxp_4x1vl_rvv_funcs();
    run_w4a8_packed_gemm_test<bfloat16_t>(GetParam(), funcs);
}

INSTANTIATE_TEST_SUITE_P(GemmTests, Test_gemm_1xnbias_clamp_bf16_qai8dxp_qsi4cxp_4x1vl_rvv,
                         common_w4a8_test_values(), GemmTestParamNameGenerator());
#endif  // __riscv_zvfbfwma
