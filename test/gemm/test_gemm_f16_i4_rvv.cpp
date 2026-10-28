//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#include "src/gemm/gemm_f16_i4/gemm_1xnbias_clamp_f16_f16_qsi4c2t/tqt_gemm_1xnbias_clamp_f16_f16_qsi4c2t_8x2vl_rvv.h"
#include "src/gemm/gemm_f16_i4/gemm_1xnbias_clamp_f16_f16p_qsi4c2p/tqt_gemm_1xnbias_clamp_f16_f16p_qsi4c2p_8x2vl_rvv.h"
#include "src/gemm/gemm_f16_i4/gemm_1xnbias_clamp_f32_f16_qsi4c2t/tqt_gemm_1xnbias_clamp_f32_f16_qsi4c2t_8x2vl_rvv.h"
#include "src/gemm/gemm_f16_i4/gemm_1xnbias_clamp_f32_f16p_qsi4c2p/tqt_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x2vl_rvv.h"
#include "test/common/gemm_clamp_test_utils.h"
#include "test/common/gemm_test_runner.h"
#include "test/common/quantized_gemm_test_runner.h"

using namespace torq_tile::test;

// ============================================================================
// Common test values shared by all W4A16 GEMM variants in this file.
// ============================================================================

static auto common_quantized_weight_test_values()
{
    return testing::Values(
        // Basic shapes with bl=2 (minimum)
        GemmTestParams(8, 8, 4, "Basic").set_bl(2), GemmTestParams(16, 16, 8, "Small").set_bl(2),
        GemmTestParams(32, 32, 32, "Medium_BL2").set_bl(2),
        // bl=4
        GemmTestParams(32, 32, 32, "Medium_BL4").set_bl(4),
        // bl=32 (typical LLM group_size)
        GemmTestParams(32, 32, 32, "Medium_BL32").set_bl(32),
        // Non-aligned M/N
        GemmTestParams(5, 11, 10, "NonAligned_BL2").set_bl(2),
        GemmTestParams(7, 3, 10, "SmallNonAligned").set_bl(2),
        GemmTestParams(17, 19, 24, "Odd_BL4").set_bl(4),
        // Larger shapes
        GemmTestParams(31, 35, 64, "Large_BL32").set_bl(32),
        GemmTestParams(60, 61, 128, "LargerK_BL32").set_bl(32),
        // With bias
        GemmTestParams(32, 32, 32, "WithBias").set_bl(2).set_bias(true),
        GemmTestParams(17, 19, 24, "WithBias_BL4").set_bl(4).set_bias(true),
        // With clamp
        GemmTestParams(32, 32, 32, "WithClamp").set_bl(4).set_clamp(-0.5f, 0.5f),
        // With C
        GemmTestParams(32, 32, 32, "WithC").set_bl(4).set_c(true),
        // Combined
        GemmTestParams(32, 32, 32, "WithC_WithBias_WithClamp")
            .set_bl(4)
            .set_c(true)
            .set_bias(true)
            .set_clamp(-0.5f, 0.5f));
}

// ============================================================================
// gemm_1xnbias_clamp_f16_f16_qsi4c2t_8x2vl_rvv (D=fp16)
// ============================================================================

static QuantizedWeightGemmFuncs<float16_t, float16_t>
make_gemm_1xnbias_clamp_f16_f16_qsi4c2t_8x2vl_rvv_funcs()
{
    QuantizedWeightGemmFuncs<float16_t, float16_t> funcs;
    funcs.get_m_step = tqt_get_m_step_gemm_1xnbias_clamp_f16_f16_qsi4c2t_8x2vl_rvv;
    funcs.get_n_step = tqt_get_n_step_gemm_1xnbias_clamp_f16_f16_qsi4c2t_8x2vl_rvv;
    funcs.get_a_offset = tqt_get_a_offset_gemm_1xnbias_clamp_f16_f16_qsi4c2t_8x2vl_rvv;
    funcs.get_b_offset = tqt_get_b_offset_gemm_1xnbias_clamp_f16_f16_qsi4c2t_8x2vl_rvv;
    funcs.get_scale_b_offset = tqt_get_scale_b_offset_gemm_1xnbias_clamp_f16_f16_qsi4c2t_8x2vl_rvv;
    funcs.get_c_offset = tqt_get_c_offset_gemm_1xnbias_clamp_f16_f16_qsi4c2t_8x2vl_rvv;
    funcs.get_bias_offset = tqt_get_bias_offset_gemm_1xnbias_clamp_f16_f16_qsi4c2t_8x2vl_rvv;
    funcs.get_d_offset = tqt_get_d_offset_gemm_1xnbias_clamp_f16_f16_qsi4c2t_8x2vl_rvv;
    funcs.get_d_size = tqt_get_d_size_gemm_1xnbias_clamp_f16_f16_qsi4c2t_8x2vl_rvv;
    funcs.run_gemm = clamp_test_run<tqt_run_gemm_1xnbias_clamp_f16_f16_qsi4c2t_8x2vl_rvv, float16_t,
                                    16, 14, 15, 17, ClampSource::kColumnBias, false, 6, true>();
    return funcs;
}

class Test_gemm_1xnbias_clamp_f16_f16_qsi4c2t_8x2vl_rvv
    : public testing::TestWithParam<GemmTestParams>
{
};

TEST_P(Test_gemm_1xnbias_clamp_f16_f16_qsi4c2t_8x2vl_rvv, Correctness)
{
    auto funcs = make_gemm_1xnbias_clamp_f16_f16_qsi4c2t_8x2vl_rvv_funcs();
    run_quantized_weight_gemm_test<float16_t, float16_t, BiasMode::k1xN>(GetParam(), funcs);
}

TEST_P(Test_gemm_1xnbias_clamp_f16_f16_qsi4c2t_8x2vl_rvv, ClampSpecialValues)
{
    ScopedGemmClampProbe probe;
    auto funcs = make_gemm_1xnbias_clamp_f16_f16_qsi4c2t_8x2vl_rvv_funcs();
    run_quantized_weight_gemm_test<float16_t, float16_t, BiasMode::k1xN>(GetParam(), funcs);
}

INSTANTIATE_TEST_SUITE_P(
    GemmTests, Test_gemm_1xnbias_clamp_f16_f16_qsi4c2t_8x2vl_rvv,
    with_clamp_dimensions(common_quantized_weight_test_values(),
                          tqt_get_m_step_gemm_1xnbias_clamp_f16_f16_qsi4c2t_8x2vl_rvv(),
                          tqt_get_n_step_gemm_1xnbias_clamp_f16_f16_qsi4c2t_8x2vl_rvv(), 2, true,
                          false),
    GemmTestParamNameGenerator());

// ============================================================================
// gemm_1xnbias_clamp_f16_f16p_qsi4c2p_8x2vl_rvv (D=fp16, packed A and B)
// ============================================================================

static PackedQuantizedWeightGemmFuncs<float16_t, float16_t>
make_gemm_1xnbias_clamp_f16_f16p_qsi4c2p_8x2vl_rvv_funcs()
{
    PackedQuantizedWeightGemmFuncs<float16_t, float16_t> funcs;
    funcs.get_m_step = tqt_get_m_step_gemm_1xnbias_clamp_f16_f16p_qsi4c2p_8x2vl_rvv;
    funcs.get_n_step = tqt_get_n_step_gemm_1xnbias_clamp_f16_f16p_qsi4c2p_8x2vl_rvv;
    funcs.get_a_packed_offset =
        tqt_get_a_packed_offset_gemm_1xnbias_clamp_f16_f16p_qsi4c2p_8x2vl_rvv;
    funcs.get_b_packed_offset =
        tqt_get_b_packed_offset_gemm_1xnbias_clamp_f16_f16p_qsi4c2p_8x2vl_rvv;
    funcs.get_scale_b_offset = tqt_get_scale_b_offset_gemm_1xnbias_clamp_f16_f16p_qsi4c2p_8x2vl_rvv;
    funcs.get_c_offset = tqt_get_c_offset_gemm_1xnbias_clamp_f16_f16p_qsi4c2p_8x2vl_rvv;
    funcs.get_bias_offset = tqt_get_bias_offset_gemm_1xnbias_clamp_f16_f16p_qsi4c2p_8x2vl_rvv;
    funcs.get_d_offset = tqt_get_d_offset_gemm_1xnbias_clamp_f16_f16p_qsi4c2p_8x2vl_rvv;
    funcs.get_d_size = tqt_get_d_size_gemm_1xnbias_clamp_f16_f16p_qsi4c2p_8x2vl_rvv;
    funcs.get_a_packed_size = tqt_get_a_packed_size_gemm_1xnbias_clamp_f16_f16p_qsi4c2p_8x2vl_rvv;
    funcs.get_b_packed_size = tqt_get_b_packed_size_gemm_1xnbias_clamp_f16_f16p_qsi4c2p_8x2vl_rvv;
    funcs.run_a_pack = tqt_run_a_pack_gemm_1xnbias_clamp_f16_f16p_qsi4c2p_8x2vl_rvv;
    funcs.run_bt_pack = tqt_run_bt_pack_gemm_1xnbias_clamp_f16_f16p_qsi4c2p_8x2vl_rvv;
    funcs.run_gemm =
        clamp_test_run<tqt_run_gemm_1xnbias_clamp_f16_f16p_qsi4c2p_8x2vl_rvv, float16_t, 16, 14, 15,
                       17, ClampSource::kColumnBias, false, 6, true>();
    return funcs;
}

class Test_gemm_1xnbias_clamp_f16_f16p_qsi4c2p_8x2vl_rvv
    : public testing::TestWithParam<GemmTestParams>
{
};

TEST_P(Test_gemm_1xnbias_clamp_f16_f16p_qsi4c2p_8x2vl_rvv, Correctness)
{
    auto funcs = make_gemm_1xnbias_clamp_f16_f16p_qsi4c2p_8x2vl_rvv_funcs();
    run_packed_quantized_weight_gemm_test<float16_t, float16_t, BiasMode::k1xN>(GetParam(), funcs);
}

TEST_P(Test_gemm_1xnbias_clamp_f16_f16p_qsi4c2p_8x2vl_rvv, ClampSpecialValues)
{
    ScopedGemmClampProbe probe;
    auto funcs = make_gemm_1xnbias_clamp_f16_f16p_qsi4c2p_8x2vl_rvv_funcs();
    run_packed_quantized_weight_gemm_test<float16_t, float16_t, BiasMode::k1xN>(GetParam(), funcs);
}

INSTANTIATE_TEST_SUITE_P(
    GemmTests, Test_gemm_1xnbias_clamp_f16_f16p_qsi4c2p_8x2vl_rvv,
    with_clamp_dimensions(common_quantized_weight_test_values(),
                          tqt_get_m_step_gemm_1xnbias_clamp_f16_f16p_qsi4c2p_8x2vl_rvv(),
                          tqt_get_n_step_gemm_1xnbias_clamp_f16_f16p_qsi4c2p_8x2vl_rvv(), 2, true,
                          false),
    GemmTestParamNameGenerator());

// ============================================================================
// gemm_1xnbias_clamp_f32_f16_qsi4c2t_8x2vl_rvv (D=fp32, A=fp16)
// ============================================================================

static QuantizedWeightGemmFuncs<float16_t, float>
make_gemm_1xnbias_clamp_f32_f16_qsi4c2t_8x2vl_rvv_funcs()
{
    QuantizedWeightGemmFuncs<float16_t, float> funcs;
    funcs.get_m_step = tqt_get_m_step_gemm_1xnbias_clamp_f32_f16_qsi4c2t_8x2vl_rvv;
    funcs.get_n_step = tqt_get_n_step_gemm_1xnbias_clamp_f32_f16_qsi4c2t_8x2vl_rvv;
    funcs.get_a_offset = tqt_get_a_offset_gemm_1xnbias_clamp_f32_f16_qsi4c2t_8x2vl_rvv;
    funcs.get_b_offset = tqt_get_b_offset_gemm_1xnbias_clamp_f32_f16_qsi4c2t_8x2vl_rvv;
    funcs.get_scale_b_offset = tqt_get_scale_b_offset_gemm_1xnbias_clamp_f32_f16_qsi4c2t_8x2vl_rvv;
    funcs.get_c_offset = tqt_get_c_offset_gemm_1xnbias_clamp_f32_f16_qsi4c2t_8x2vl_rvv;
    funcs.get_bias_offset = tqt_get_bias_offset_gemm_1xnbias_clamp_f32_f16_qsi4c2t_8x2vl_rvv;
    funcs.get_d_offset = tqt_get_d_offset_gemm_1xnbias_clamp_f32_f16_qsi4c2t_8x2vl_rvv;
    funcs.get_d_size = tqt_get_d_size_gemm_1xnbias_clamp_f32_f16_qsi4c2t_8x2vl_rvv;
    funcs.run_gemm = clamp_test_run<tqt_run_gemm_1xnbias_clamp_f32_f16_qsi4c2t_8x2vl_rvv, float, 16,
                                    14, 15, 17, ClampSource::kColumnBias, true, 6, true>();
    return funcs;
}

class Test_gemm_1xnbias_clamp_f32_f16_qsi4c2t_8x2vl_rvv
    : public testing::TestWithParam<GemmTestParams>
{
};

TEST_P(Test_gemm_1xnbias_clamp_f32_f16_qsi4c2t_8x2vl_rvv, Correctness)
{
    auto funcs = make_gemm_1xnbias_clamp_f32_f16_qsi4c2t_8x2vl_rvv_funcs();
    run_quantized_weight_gemm_test<float16_t, float, BiasMode::k1xN>(GetParam(), funcs);
}

TEST_P(Test_gemm_1xnbias_clamp_f32_f16_qsi4c2t_8x2vl_rvv, ClampSpecialValues)
{
    ScopedGemmClampProbe probe;
    auto funcs = make_gemm_1xnbias_clamp_f32_f16_qsi4c2t_8x2vl_rvv_funcs();
    run_quantized_weight_gemm_test<float16_t, float, BiasMode::k1xN>(GetParam(), funcs);
}

INSTANTIATE_TEST_SUITE_P(
    GemmTests, Test_gemm_1xnbias_clamp_f32_f16_qsi4c2t_8x2vl_rvv,
    with_clamp_dimensions(common_quantized_weight_test_values(),
                          tqt_get_m_step_gemm_1xnbias_clamp_f32_f16_qsi4c2t_8x2vl_rvv(),
                          tqt_get_n_step_gemm_1xnbias_clamp_f32_f16_qsi4c2t_8x2vl_rvv(), 2, true,
                          false),
    GemmTestParamNameGenerator());

// ============================================================================
// gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x2vl_rvv (D=fp32, A=fp16, packed A and B)
// ============================================================================

static PackedQuantizedWeightGemmFuncs<float16_t, float>
make_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x2vl_rvv_funcs()
{
    PackedQuantizedWeightGemmFuncs<float16_t, float> funcs;
    funcs.get_m_step = tqt_get_m_step_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x2vl_rvv;
    funcs.get_n_step = tqt_get_n_step_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x2vl_rvv;
    funcs.get_a_packed_offset =
        tqt_get_a_packed_offset_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x2vl_rvv;
    funcs.get_b_packed_offset =
        tqt_get_b_packed_offset_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x2vl_rvv;
    funcs.get_scale_b_offset = tqt_get_scale_b_offset_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x2vl_rvv;
    funcs.get_c_offset = tqt_get_c_offset_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x2vl_rvv;
    funcs.get_bias_offset = tqt_get_bias_offset_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x2vl_rvv;
    funcs.get_d_offset = tqt_get_d_offset_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x2vl_rvv;
    funcs.get_d_size = tqt_get_d_size_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x2vl_rvv;
    funcs.get_a_packed_size = tqt_get_a_packed_size_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x2vl_rvv;
    funcs.get_b_packed_size = tqt_get_b_packed_size_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x2vl_rvv;
    funcs.run_a_pack = tqt_run_a_pack_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x2vl_rvv;
    funcs.run_bt_pack = tqt_run_bt_pack_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x2vl_rvv;
    funcs.run_gemm = clamp_test_run<tqt_run_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x2vl_rvv, float,
                                    16, 14, 15, 17, ClampSource::kColumnBias, true, 6, true>();
    return funcs;
}

class Test_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x2vl_rvv
    : public testing::TestWithParam<GemmTestParams>
{
};

TEST_P(Test_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x2vl_rvv, Correctness)
{
    auto funcs = make_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x2vl_rvv_funcs();
    run_packed_quantized_weight_gemm_test<float16_t, float, BiasMode::k1xN>(GetParam(), funcs);
}

TEST_P(Test_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x2vl_rvv, ClampSpecialValues)
{
    ScopedGemmClampProbe probe;
    auto funcs = make_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x2vl_rvv_funcs();
    run_packed_quantized_weight_gemm_test<float16_t, float, BiasMode::k1xN>(GetParam(), funcs);
}

INSTANTIATE_TEST_SUITE_P(
    GemmTests, Test_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x2vl_rvv,
    with_clamp_dimensions(common_quantized_weight_test_values(),
                          tqt_get_m_step_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x2vl_rvv(),
                          tqt_get_n_step_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x2vl_rvv(), 2, true,
                          false),
    GemmTestParamNameGenerator());

TEST(W4Contract, gemm_1xnbias_clamp_f16_f16_qsi4c2t_8x2vl_rvv)
{
    check_w4_contract<float16_t, float16_t, false, false, false>(
        tqt_get_m_step_gemm_1xnbias_clamp_f16_f16_qsi4c2t_8x2vl_rvv(),
        tqt_get_n_step_gemm_1xnbias_clamp_f16_f16_qsi4c2t_8x2vl_rvv(), 0.125f,
        [&](size_t m, size_t n, size_t k, size_t bl, size_t fm, size_t fn, size_t fk, size_t mi,
            size_t ni, size_t ki, size_t ld, size_t lds, const auto &a, const auto &b,
            const auto &sa, const auto &za, const auto &sb, const auto &ob, const float *bias,
            const float16_t *c, float16_t *d, size_t ldd, float lo, float hi,
            const float16_t *lut16, const int8_t *lut8) {
            const void *aw =
                (const uint8_t *)a.data() +
                tqt_get_a_offset_gemm_1xnbias_clamp_f16_f16_qsi4c2t_8x2vl_rvv(mi, ki, ld, bl);
            const void *bw =
                b.data() +
                tqt_get_b_offset_gemm_1xnbias_clamp_f16_f16_qsi4c2t_8x2vl_rvv(ni, ki, ld, bl);
            std::vector<float16_t> halfbias(fn);
            if (bias)
                for (size_t j = 0; j < fn; ++j)
                    halfbias[j] = (float16_t)bias[j];
            const float16_t *biasw = bias ? halfbias.data() + ni : nullptr;
            std::vector<float16_t> scales(sb.size());
            for (size_t j = 0; j < sb.size(); ++j)
                scales[j] = (float16_t)sb[j];
            tqt_run_gemm_1xnbias_clamp_f16_f16_qsi4c2t_8x2vl_rvv(
                m, n, k, bl, aw, ld, ki, bw, ld, ki,
                scales.data() + tqt_get_scale_b_offset_gemm_1xnbias_clamp_f16_f16_qsi4c2t_8x2vl_rvv(
                                    ni, ki / bl, lds) /
                                    sizeof(float16_t),
                lds, c, ldd, d, ldd, biasw, (float16_t)lo, (float16_t)hi, lut16);
        });
}

TEST(W4Contract, gemm_1xnbias_clamp_f16_f16p_qsi4c2p_8x2vl_rvv)
{
    check_w4_contract<float16_t, float16_t, false, false, false>(
        tqt_get_m_step_gemm_1xnbias_clamp_f16_f16p_qsi4c2p_8x2vl_rvv(),
        tqt_get_n_step_gemm_1xnbias_clamp_f16_f16p_qsi4c2p_8x2vl_rvv(), 0.125f,
        [&](size_t m, size_t n, size_t k, size_t bl, size_t fm, size_t fn, size_t fk, size_t mi,
            size_t ni, size_t ki, size_t ld, size_t lds, const auto &a, const auto &b,
            const auto &sa, const auto &za, const auto &sb, const auto &ob, const float *bias,
            const float16_t *c, float16_t *d, size_t ldd, float lo, float hi,
            const float16_t *lut16, const int8_t *lut8) {
            std::vector<uint8_t> ap(
                tqt_get_a_packed_size_gemm_1xnbias_clamp_f16_f16p_qsi4c2p_8x2vl_rvv(fm, fk)),
                bp(tqt_get_b_packed_size_gemm_1xnbias_clamp_f16_f16p_qsi4c2p_8x2vl_rvv(fn, fk));
            tqt_run_a_pack_gemm_1xnbias_clamp_f16_f16p_qsi4c2p_8x2vl_rvv(fm, fk, ld, fk, 0,
                                                                         a.data(), ap.data());
            tqt_run_bt_pack_gemm_1xnbias_clamp_f16_f16p_qsi4c2p_8x2vl_rvv(fn, fk, ld, fk, 0,
                                                                          b.data(), bp.data());
            const void *aw =
                ap.data() + tqt_get_a_packed_offset_gemm_1xnbias_clamp_f16_f16p_qsi4c2p_8x2vl_rvv(
                                mi, ki, fk, m);
            const void *bw =
                bp.data() + tqt_get_b_packed_offset_gemm_1xnbias_clamp_f16_f16p_qsi4c2p_8x2vl_rvv(
                                ni, ki, fk, n);
            std::vector<float16_t> halfbias(fn);
            if (bias)
                for (size_t j = 0; j < fn; ++j)
                    halfbias[j] = (float16_t)bias[j];
            const float16_t *biasw = bias ? halfbias.data() + ni : nullptr;
            std::vector<float16_t> scales(sb.size());
            for (size_t j = 0; j < sb.size(); ++j)
                scales[j] = (float16_t)sb[j];
            tqt_run_gemm_1xnbias_clamp_f16_f16p_qsi4c2p_8x2vl_rvv(
                m, n, k, bl, aw, fk, ki, bw, fk, ki,
                scales.data() +
                    tqt_get_scale_b_offset_gemm_1xnbias_clamp_f16_f16p_qsi4c2p_8x2vl_rvv(
                        ni, ki / bl, lds) /
                        sizeof(float16_t),
                lds, c, ldd, d, ldd, biasw, (float16_t)lo, (float16_t)hi, lut16);
        });
}

TEST(W4Contract, gemm_1xnbias_clamp_f32_f16_qsi4c2t_8x2vl_rvv)
{
    check_w4_contract<float16_t, float, false, false, false>(
        tqt_get_m_step_gemm_1xnbias_clamp_f32_f16_qsi4c2t_8x2vl_rvv(),
        tqt_get_n_step_gemm_1xnbias_clamp_f32_f16_qsi4c2t_8x2vl_rvv(), 0.125f,
        [&](size_t m, size_t n, size_t k, size_t bl, size_t fm, size_t fn, size_t fk, size_t mi,
            size_t ni, size_t ki, size_t ld, size_t lds, const auto &a, const auto &b,
            const auto &sa, const auto &za, const auto &sb, const auto &ob, const float *bias,
            const float *c, float *d, size_t ldd, float lo, float hi, const float16_t *lut16,
            const int8_t *lut8) {
            const void *aw =
                (const uint8_t *)a.data() +
                tqt_get_a_offset_gemm_1xnbias_clamp_f32_f16_qsi4c2t_8x2vl_rvv(mi, ki, ld, bl);
            const void *bw =
                b.data() +
                tqt_get_b_offset_gemm_1xnbias_clamp_f32_f16_qsi4c2t_8x2vl_rvv(ni, ki, ld, bl);
            const float *biasw =
                bias ? bias + tqt_get_bias_offset_gemm_1xnbias_clamp_f32_f16_qsi4c2t_8x2vl_rvv(ni) /
                                  sizeof(float)
                     : nullptr;
            std::vector<float16_t> scales(sb.size());
            for (size_t j = 0; j < sb.size(); ++j)
                scales[j] = (float16_t)sb[j];
            tqt_run_gemm_1xnbias_clamp_f32_f16_qsi4c2t_8x2vl_rvv(
                m, n, k, bl, aw, ld, ki, bw, ld, ki,
                scales.data() + tqt_get_scale_b_offset_gemm_1xnbias_clamp_f32_f16_qsi4c2t_8x2vl_rvv(
                                    ni, ki / bl, lds) /
                                    sizeof(float16_t),
                lds, c, ldd, d, ldd, biasw, (float)lo, (float)hi, lut16);
        });
}
#include "src/gemm/gemm_f16_i4/gemm_1xnbias_clamp_f32_f16_qsi4c2t/tqt_gemm_1xnbias_clamp_f32_f16_qsi4c2t_8x1vl_rvv.h"

TEST(W4Contract, gemm_1xnbias_clamp_f32_f16_qsi4c2t_8x1vl_rvv)
{
    check_w4_contract<float16_t, float, false, false, false>(
        tqt_get_m_step_gemm_1xnbias_clamp_f32_f16_qsi4c2t_8x1vl_rvv(),
        tqt_get_n_step_gemm_1xnbias_clamp_f32_f16_qsi4c2t_8x1vl_rvv(), 0.00001f,
        [&](size_t m, size_t n, size_t k, size_t bl, size_t fm, size_t fn, size_t fk, size_t mi,
            size_t ni, size_t ki, size_t ld, size_t lds, const auto &a, const auto &b,
            const auto &sa, const auto &za, const auto &sb, const auto &ob, const float *bias,
            const float *c, float *d, size_t ldd, float lo, float hi, const float16_t *lut16,
            const int8_t *lut8) {
            const void *aw =
                (const uint8_t *)a.data() +
                tqt_get_a_offset_gemm_1xnbias_clamp_f32_f16_qsi4c2t_8x1vl_rvv(mi, ki, ld, bl);
            const void *bw =
                b.data() +
                tqt_get_b_offset_gemm_1xnbias_clamp_f32_f16_qsi4c2t_8x1vl_rvv(ni, ki, ld, bl);
            const float *biasw =
                bias ? bias + tqt_get_bias_offset_gemm_1xnbias_clamp_f32_f16_qsi4c2t_8x1vl_rvv(ni) /
                                  sizeof(float)
                     : nullptr;
            std::vector<float16_t> scales(sb.size());
            for (size_t j = 0; j < sb.size(); ++j)
                scales[j] = (float16_t)sb[j];
            tqt_run_gemm_1xnbias_clamp_f32_f16_qsi4c2t_8x1vl_rvv(
                m, n, k, bl, aw, ld, ki, bw, ld, ki,
                scales.data() + tqt_get_scale_b_offset_gemm_1xnbias_clamp_f32_f16_qsi4c2t_8x1vl_rvv(
                                    ni, ki / bl, lds) /
                                    sizeof(float16_t),
                lds, c, ldd, d, ldd, biasw, (float)lo, (float)hi, lut16);
        });
}

TEST(W4Contract, gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x2vl_rvv)
{
    check_w4_contract<float16_t, float, false, false, false>(
        tqt_get_m_step_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x2vl_rvv(),
        tqt_get_n_step_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x2vl_rvv(), 0.125f,
        [&](size_t m, size_t n, size_t k, size_t bl, size_t fm, size_t fn, size_t fk, size_t mi,
            size_t ni, size_t ki, size_t ld, size_t lds, const auto &a, const auto &b,
            const auto &sa, const auto &za, const auto &sb, const auto &ob, const float *bias,
            const float *c, float *d, size_t ldd, float lo, float hi, const float16_t *lut16,
            const int8_t *lut8) {
            std::vector<uint8_t> ap(
                tqt_get_a_packed_size_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x2vl_rvv(fm, fk)),
                bp(tqt_get_b_packed_size_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x2vl_rvv(fn, fk));
            tqt_run_a_pack_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x2vl_rvv(fm, fk, ld, fk, 0,
                                                                         a.data(), ap.data());
            tqt_run_bt_pack_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x2vl_rvv(fn, fk, ld, fk, 0,
                                                                          b.data(), bp.data());
            const void *aw =
                ap.data() + tqt_get_a_packed_offset_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x2vl_rvv(
                                mi, ki, fk, m);
            const void *bw =
                bp.data() + tqt_get_b_packed_offset_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x2vl_rvv(
                                ni, ki, fk, n);
            const float *biasw =
                bias
                    ? bias + tqt_get_bias_offset_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x2vl_rvv(ni) /
                                 sizeof(float)
                    : nullptr;
            std::vector<float16_t> scales(sb.size());
            for (size_t j = 0; j < sb.size(); ++j)
                scales[j] = (float16_t)sb[j];
            tqt_run_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x2vl_rvv(
                m, n, k, bl, aw, fk, ki, bw, fk, ki,
                scales.data() +
                    tqt_get_scale_b_offset_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x2vl_rvv(
                        ni, ki / bl, lds) /
                        sizeof(float16_t),
                lds, c, ldd, d, ldd, biasw, (float)lo, (float)hi, lut16);
        });
}
#include "src/gemm/gemm_f16_i4/gemm_1xnbias_clamp_f32_f16p_qsi4c2p/tqt_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x1vl_rvv.h"

TEST(W4Contract, gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x1vl_rvv)
{
    check_w4_contract<float16_t, float, false, false, false>(
        tqt_get_m_step_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x1vl_rvv(),
        tqt_get_n_step_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x1vl_rvv(), 0.00001f,
        [&](size_t m, size_t n, size_t k, size_t bl, size_t fm, size_t fn, size_t fk, size_t mi,
            size_t ni, size_t ki, size_t ld, size_t lds, const auto &a, const auto &b,
            const auto &sa, const auto &za, const auto &sb, const auto &ob, const float *bias,
            const float *c, float *d, size_t ldd, float lo, float hi, const float16_t *lut16,
            const int8_t *lut8) {
            std::vector<uint8_t> ap(
                tqt_get_a_packed_size_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x1vl_rvv(fm, fk)),
                bp(tqt_get_b_packed_size_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x1vl_rvv(fn, fk));
            tqt_run_a_pack_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x1vl_rvv(fm, fk, ld, fk, 0,
                                                                         a.data(), ap.data());
            tqt_run_bt_pack_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x1vl_rvv(fn, fk, ld, fk, 0,
                                                                          b.data(), bp.data());
            const void *aw =
                ap.data() + tqt_get_a_packed_offset_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x1vl_rvv(
                                mi, ki, fk, m);
            const void *bw =
                bp.data() + tqt_get_b_packed_offset_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x1vl_rvv(
                                ni, ki, fk, n);
            const float *biasw =
                bias
                    ? bias + tqt_get_bias_offset_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x1vl_rvv(ni) /
                                 sizeof(float)
                    : nullptr;
            std::vector<float16_t> scales(sb.size());
            for (size_t j = 0; j < sb.size(); ++j)
                scales[j] = (float16_t)sb[j];
            tqt_run_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x1vl_rvv(
                m, n, k, bl, aw, fk, ki, bw, fk, ki,
                scales.data() +
                    tqt_get_scale_b_offset_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x1vl_rvv(
                        ni, ki / bl, lds) /
                        sizeof(float16_t),
                lds, c, ldd, d, ldd, biasw, (float)lo, (float)hi, lut16);
        });
}
#include "src/gemm/gemm_f16_i4/gemm_1xnbias_clamp_f32_f16_qai4c2t/tqt_gemm_1xnbias_clamp_f32_f16_qai4c2t_8x1vl_rvv.h"

TEST(W4Contract, gemm_1xnbias_clamp_f32_f16_qai4c2t_8x1vl_rvv)
{
    check_w4_contract<float16_t, float, true, false, false>(
        tqt_get_m_step_gemm_1xnbias_clamp_f32_f16_qai4c2t_8x1vl_rvv(),
        tqt_get_n_step_gemm_1xnbias_clamp_f32_f16_qai4c2t_8x1vl_rvv(), 0.00001f,
        [&](size_t m, size_t n, size_t k, size_t bl, size_t fm, size_t fn, size_t fk, size_t mi,
            size_t ni, size_t ki, size_t ld, size_t lds, const auto &a, const auto &b,
            const auto &sa, const auto &za, const auto &sb, const auto &ob, const float *bias,
            const float *c, float *d, size_t ldd, float lo, float hi, const float16_t *lut16,
            const int8_t *lut8) {
            const void *aw =
                (const uint8_t *)a.data() +
                tqt_get_a_offset_gemm_1xnbias_clamp_f32_f16_qai4c2t_8x1vl_rvv(mi, ki, ld, bl);
            const void *bw =
                b.data() +
                tqt_get_b_offset_gemm_1xnbias_clamp_f32_f16_qai4c2t_8x1vl_rvv(ni, ki, ld, bl);
            const float *biasw =
                bias ? bias + tqt_get_bias_offset_gemm_1xnbias_clamp_f32_f16_qai4c2t_8x1vl_rvv(ni) /
                                  sizeof(float)
                     : nullptr;
            tqt_run_gemm_1xnbias_clamp_f32_f16_qai4c2t_8x1vl_rvv(
                m, n, k, bl, aw, ld, ki, bw, ld, ki,
                sb.data() + tqt_get_scale_b_offset_gemm_1xnbias_clamp_f32_f16_qai4c2t_8x1vl_rvv(
                                ni, ki / bl, lds) /
                                sizeof(float),
                lds,
                ob.data() + tqt_get_offset_b_offset_gemm_1xnbias_clamp_f32_f16_qai4c2t_8x1vl_rvv(
                                ni, ki / bl, lds) /
                                sizeof(float),
                lds, c, ldd, d, ldd, biasw, lo, hi);
        });
}
#include "src/gemm/gemm_f16_i4/gemm_1xnbias_clamp_f32_f16p_qai4c2p/tqt_gemm_1xnbias_clamp_f32_f16p_qai4c2p_8x1vl_rvv.h"

TEST(W4Contract, gemm_1xnbias_clamp_f32_f16p_qai4c2p_8x1vl_rvv)
{
    check_w4_contract<float16_t, float, true, false, false>(
        tqt_get_m_step_gemm_1xnbias_clamp_f32_f16p_qai4c2p_8x1vl_rvv(),
        tqt_get_n_step_gemm_1xnbias_clamp_f32_f16p_qai4c2p_8x1vl_rvv(), 0.00001f,
        [&](size_t m, size_t n, size_t k, size_t bl, size_t fm, size_t fn, size_t fk, size_t mi,
            size_t ni, size_t ki, size_t ld, size_t lds, const auto &a, const auto &b,
            const auto &sa, const auto &za, const auto &sb, const auto &ob, const float *bias,
            const float *c, float *d, size_t ldd, float lo, float hi, const float16_t *lut16,
            const int8_t *lut8) {
            std::vector<uint8_t> ap(
                tqt_get_a_packed_size_gemm_1xnbias_clamp_f32_f16p_qai4c2p_8x1vl_rvv(fm, fk)),
                bp(tqt_get_b_packed_size_gemm_1xnbias_clamp_f32_f16p_qai4c2p_8x1vl_rvv(fn, fk));
            tqt_run_a_pack_gemm_1xnbias_clamp_f32_f16p_qai4c2p_8x1vl_rvv(fm, fk, ld, fk, 0,
                                                                         a.data(), ap.data());
            tqt_run_bt_pack_gemm_1xnbias_clamp_f32_f16p_qai4c2p_8x1vl_rvv(fn, fk, ld, fk, 0,
                                                                          b.data(), bp.data());
            const void *aw =
                ap.data() + tqt_get_a_packed_offset_gemm_1xnbias_clamp_f32_f16p_qai4c2p_8x1vl_rvv(
                                mi, ki, fk, m);
            const void *bw =
                bp.data() + tqt_get_b_packed_offset_gemm_1xnbias_clamp_f32_f16p_qai4c2p_8x1vl_rvv(
                                ni, ki, fk, n);
            const float *biasw =
                bias
                    ? bias + tqt_get_bias_offset_gemm_1xnbias_clamp_f32_f16p_qai4c2p_8x1vl_rvv(ni) /
                                 sizeof(float)
                    : nullptr;
            tqt_run_gemm_1xnbias_clamp_f32_f16p_qai4c2p_8x1vl_rvv(
                m, n, k, bl, aw, fk, ki, bw, fk, ki,
                sb.data() + tqt_get_scale_b_offset_gemm_1xnbias_clamp_f32_f16p_qai4c2p_8x1vl_rvv(
                                ni, ki / bl, lds) /
                                sizeof(float),
                lds,
                ob.data() + tqt_get_offset_b_offset_gemm_1xnbias_clamp_f32_f16p_qai4c2p_8x1vl_rvv(
                                ni, ki / bl, lds) /
                                sizeof(float),
                lds, c, ldd, d, ldd, biasw, lo, hi);
        });
}

TEST(F32BiasContract, gemm_1xnbias_clamp_f32_f16_qsi4c2t_8x1vl_rvv)
{
    const size_t m = 9, n = tqt_get_n_step_gemm_1xnbias_clamp_f32_f16_qsi4c2t_8x1vl_rvv() + 1,
                 k = 32, bl = 32;
    std::vector<float16_t> a(m * k, (float16_t)64), hs(n, (float16_t)8);
    std::vector<uint8_t> b(n * k / 2, 0xff);
    std::vector<float> scale(n, 8), offset(n, -64);
    check_f32_bias_epilogue<float>(
        m, n, 114688.0f, true,
        [&](const float *c, size_t stride, const float *bias, float *d, float lo, float hi) {
            tqt_run_gemm_1xnbias_clamp_f32_f16_qsi4c2t_8x1vl_rvv(
                m, n, k, bl, a.data(), k, 0, b.data(), k, 0, hs.data(), 1, c, stride, d, stride,
                bias, lo, hi, nullptr);
        });
}

TEST(F32BiasContract, gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x1vl_rvv)
{
    const size_t m = 9, n = tqt_get_n_step_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x1vl_rvv() + 1,
                 k = 32, bl = 32;
    std::vector<float16_t> a(m * k, (float16_t)64), hs(n, (float16_t)8);
    std::vector<uint8_t> b(n * k / 2, 0xff);
    std::vector<float> scale(n, 8), offset(n, -64);
    std::vector<uint8_t> ap(
        tqt_get_a_packed_size_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x1vl_rvv(m, k)),
        bp(tqt_get_b_packed_size_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x1vl_rvv(n, k));
    tqt_run_a_pack_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x1vl_rvv(m, k, k, k, 0, a.data(),
                                                                 ap.data());
    tqt_run_bt_pack_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x1vl_rvv(n, k, k, k, 0, b.data(),
                                                                  bp.data());
    check_f32_bias_epilogue<float>(
        m, n, 114688.0f, true,
        [&](const float *c, size_t stride, const float *bias, float *d, float lo, float hi) {
            tqt_run_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x1vl_rvv(
                m, n, k, bl, ap.data(), k, 0, bp.data(), k, 0, hs.data(), 1, c, stride, d, stride,
                bias, lo, hi, nullptr);
        });
}

TEST(F32BiasContract, gemm_1xnbias_clamp_f32_f16_qai4c2t_8x1vl_rvv)
{
    const size_t m = 9, n = tqt_get_n_step_gemm_1xnbias_clamp_f32_f16_qai4c2t_8x1vl_rvv() + 1,
                 k = 32, bl = 32;
    std::vector<float16_t> a(m * k, (float16_t)64), hs(n, (float16_t)8);
    std::vector<uint8_t> b(n * k / 2, 0xff);
    std::vector<float> scale(n, 8), offset(n, -64);
    check_f32_bias_epilogue<float>(
        m, n, 114688.0f, true,
        [&](const float *c, size_t stride, const float *bias, float *d, float lo, float hi) {
            tqt_run_gemm_1xnbias_clamp_f32_f16_qai4c2t_8x1vl_rvv(
                m, n, k, bl, a.data(), k, 0, b.data(), k, 0, scale.data(), 1, offset.data(), 1, c,
                stride, d, stride, bias, lo, hi);
        });
}

TEST(F32BiasContract, gemm_1xnbias_clamp_f32_f16p_qai4c2p_8x1vl_rvv)
{
    const size_t m = 9, n = tqt_get_n_step_gemm_1xnbias_clamp_f32_f16p_qai4c2p_8x1vl_rvv() + 1,
                 k = 32, bl = 32;
    std::vector<float16_t> a(m * k, (float16_t)64), hs(n, (float16_t)8);
    std::vector<uint8_t> b(n * k / 2, 0xff);
    std::vector<float> scale(n, 8), offset(n, -64);
    std::vector<uint8_t> ap(
        tqt_get_a_packed_size_gemm_1xnbias_clamp_f32_f16p_qai4c2p_8x1vl_rvv(m, k)),
        bp(tqt_get_b_packed_size_gemm_1xnbias_clamp_f32_f16p_qai4c2p_8x1vl_rvv(n, k));
    tqt_run_a_pack_gemm_1xnbias_clamp_f32_f16p_qai4c2p_8x1vl_rvv(m, k, k, k, 0, a.data(),
                                                                 ap.data());
    tqt_run_bt_pack_gemm_1xnbias_clamp_f32_f16p_qai4c2p_8x1vl_rvv(n, k, k, k, 0, b.data(),
                                                                  bp.data());
    check_f32_bias_epilogue<float>(
        m, n, 114688.0f, true,
        [&](const float *c, size_t stride, const float *bias, float *d, float lo, float hi) {
            tqt_run_gemm_1xnbias_clamp_f32_f16p_qai4c2p_8x1vl_rvv(
                m, n, k, bl, ap.data(), k, 0, bp.data(), k, 0, scale.data(), 1, offset.data(), 1, c,
                stride, d, stride, bias, lo, hi);
        });
}
